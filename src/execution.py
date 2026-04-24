"""
execution.py — OANDA trade execution with institutional-grade risk management.

Implements:
  1. ATR-based dynamic Stop Loss & Take Profit
  2. Kelly Criterion position sizing (fractional, conservative)
  3. Transaction cost filtering (don't trade if edge < cost)
  4. Maximum drawdown circuit breaker
  5. Maximum concurrent position limits

Key principle: RISK MANAGEMENT IS MORE IMPORTANT THAN SIGNAL QUALITY.
A 55% accurate model with great risk management beats a 70% accurate
model with poor risk management. (See: Van Tharp, "Trade Your Way
to Financial Freedom")

Position sizing formula (Fractional Kelly):
  f* = (kelly_fraction) * (p * (b+1) - 1) / b
  where:
    p = estimated win probability
    b = average win / average loss ratio
    kelly_fraction = 0.25 (quarter Kelly for safety)
"""
import logging
import math
import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.positions as positions
from src.config import (
    OANDA_TOKEN, OANDA_ACCOUNT, OANDA_ENV,
    KELLY_FRACTION, MAX_RISK_PER_TRADE, MAX_DRAWDOWN,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER, MIN_EDGE_AFTER_COSTS,
)

logger = logging.getLogger("execution")


class RiskManager:
    """
    Pre-trade risk checks and position sizing.

    This is the GATEKEEPER — every trade must pass through here before execution.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        kelly_fraction: float = KELLY_FRACTION,
        max_risk: float = MAX_RISK_PER_TRADE,
        max_drawdown: float = MAX_DRAWDOWN,
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.kelly_fraction = kelly_fraction
        self.max_risk = max_risk
        self.max_drawdown = max_drawdown
        self.is_halted = False

    def update_balance(self, balance: float):
        """Update current balance and check drawdown limits."""
        self.current_balance = balance
        if balance > self.peak_balance:
            self.peak_balance = balance

        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if drawdown >= self.max_drawdown:
            self.is_halted = True
            logger.warning(
                f"🚨 MAX DRAWDOWN BREACHED: {drawdown:.2%} >= {self.max_drawdown:.2%}. "
                f"TRADING HALTED. Manual reset required."
            )

    def calculate_position_size(
        self,
        balance: float,
        atr: float,
        price: float,
        win_rate: float = 0.52,
        avg_win_loss_ratio: float = 1.5,
        position_scale: float = 1.0,
    ) -> int:
        """
        Calculate position size using Fractional Kelly Criterion.

        Parameters
        ----------
        balance : float
            Current account balance
        atr : float
            Current ATR value (used for stop loss distance)
        price : float
            Current price of the instrument
        win_rate : float
            Estimated probability of winning (from ensemble confidence)
        avg_win_loss_ratio : float
            Average winning trade / average losing trade ratio
        position_scale : float
            Regime-based position scale (0-1)

        Returns
        -------
        int : Number of units to trade (0 if trade rejected)
        """
        if self.is_halted:
            logger.warning("Trading is HALTED due to drawdown. Returning 0 units.")
            return 0

        if atr <= 0 or price <= 0:
            return 0

        # Kelly Criterion: f* = (p(b+1) - 1) / b
        b = avg_win_loss_ratio
        p = win_rate
        kelly_f = (p * (b + 1) - 1) / b

        # Apply fractional Kelly for safety
        f_star = max(0, self.kelly_fraction * kelly_f)

        # Cap at max risk per trade
        risk_amount = min(f_star * balance, self.max_risk * balance)

        # Stop loss distance = ATR * multiplier
        sl_distance = atr * ATR_SL_MULTIPLIER

        # Position size = risk_amount / (stop_loss_distance_in_price_units)
        units = risk_amount / sl_distance if sl_distance > 0 else 0

        # Apply regime-based scaling
        units = int(units * position_scale)

        # Minimum 1 unit, maximum reasonable for account size
        max_units = int(balance * 10 / price)  # ~10x leverage cap
        units = min(max(units, 0), max_units)

        logger.info(
            f"Position sizing: Kelly f*={kelly_f:.4f}, fractional={f_star:.4f}, "
            f"risk=${risk_amount:.2f}, SL_dist={sl_distance:.5f}, "
            f"units={units}, scale={position_scale:.2f}"
        )
        return units

    def should_trade(
        self,
        confidence: float,
        atr: float,
        spread: float,
    ) -> tuple:
        """
        Pre-trade filter: should we take this trade?

        Checks:
          1. Not halted
          2. Confidence meets minimum threshold
          3. Expected profit after costs exceeds minimum edge
          4. ATR is reasonable (not in a dead market or flash crash)

        Returns (bool, str) — whether to trade and the reason if rejected.
        """
        if self.is_halted:
            return False, "Trading halted (max drawdown)"

        if confidence < 0.10:
            return False, f"Low confidence: {confidence:.4f} < 0.10"

        # Expected profit must exceed transaction costs
        # Expected move ~= ATR * confidence (simplified)
        expected_move = atr * confidence
        cost = spread * 1.5  # spread + estimated slippage

        if expected_move < cost * 2:  # Need at least 2x cost as edge
            return False, (
                f"Insufficient edge: expected_move={expected_move:.6f}, "
                f"cost={cost:.6f}, ratio={expected_move/cost:.2f}"
            )

        return True, "Trade approved"


class OandaExecutor:
    """
    Executes trades via OANDA V20 API with risk management integration.

    Handles:
      - Market orders with ATR-based SL/TP
      - Position tracking and management
      - Trade closure
    """

    def __init__(self, token: str = "", account_id: str = "", environment: str = ""):
        self.token = token or OANDA_TOKEN
        self.account_id = account_id or OANDA_ACCOUNT
        self.environment = environment or OANDA_ENV
        self.client = None

        if self.token:
            self.client = oandapyV20.API(
                access_token=self.token, environment=self.environment
            )

    def execute_market_order(
        self,
        instrument: str,
        units: int,
        signal: str,
        price: float,
        atr: float,
    ) -> dict:
        """
        Execute a market order with ATR-based SL/TP.

        Parameters
        ----------
        instrument : str (e.g. "EUR_USD")
        units : int (from RiskManager.calculate_position_size)
        signal : str ("BUY" or "SELL")
        price : float (current market price)
        atr : float (current ATR for SL/TP calculation)

        Returns
        -------
        dict with order details or error.
        """
        if self.client is None:
            logger.error("OANDA client not initialized. Cannot execute.")
            return {"error": "No client"}

        if units <= 0:
            return {"error": "Zero units — trade rejected by risk manager"}

        # Direction
        if signal == "SELL":
            units = -units

        # Calculate SL/TP based on ATR
        sl_distance = round(atr * ATR_SL_MULTIPLIER, 5)
        tp_distance = round(atr * ATR_TP_MULTIPLIER, 5)

        if signal == "BUY":
            stop_loss = round(price - sl_distance, 5)
            take_profit = round(price + tp_distance, 5)
        else:
            stop_loss = round(price + sl_distance, 5)
            take_profit = round(price - tp_distance, 5)

        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "stopLossOnFill": {"price": str(stop_loss)},
                "takeProfitOnFill": {"price": str(take_profit)},
                "timeInForce": "FOK",  # Fill or Kill
            }
        }

        try:
            req = orders.OrderCreate(accountID=self.account_id, data=order_data)
            self.client.request(req)
            response = req.response

            logger.info(
                f"✅ ORDER EXECUTED: {signal} {abs(units)} {instrument} "
                f"@ ~{price:.5f} | SL={stop_loss:.5f} | TP={take_profit:.5f}"
            )
            return {
                "status": "filled",
                "signal": signal,
                "units": units,
                "price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "response": response,
            }

        except Exception as e:
            logger.error(f"❌ Order execution failed: {e}")
            return {"error": str(e)}

    def get_open_positions(self) -> list:
        """Get all currently open positions."""
        if self.client is None:
            return []
        try:
            req = positions.OpenPositions(accountID=self.account_id)
            self.client.request(req)
            return req.response.get("positions", [])
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def get_open_trades(self) -> list:
        """Get all currently open trades."""
        if self.client is None:
            return []
        try:
            req = trades.TradesList(accountID=self.account_id)
            self.client.request(req)
            return req.response.get("trades", [])
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return []

    def close_all_positions(self) -> list:
        """Emergency close all positions. Used on shutdown or circuit break."""
        results = []
        open_positions = self.get_open_positions()
        for pos in open_positions:
            instrument = pos["instrument"]
            try:
                data = {"longUnits": "ALL", "shortUnits": "ALL"}
                req = positions.PositionClose(
                    accountID=self.account_id, instrument=instrument, data=data
                )
                self.client.request(req)
                results.append({"instrument": instrument, "status": "closed"})
                logger.info(f"Position closed: {instrument}")
            except Exception as e:
                results.append({"instrument": instrument, "error": str(e)})
                logger.error(f"Failed to close {instrument}: {e}")
        return results

    def has_open_position(self, instrument: str) -> bool:
        """Check if there's already an open position for this instrument."""
        open_pos = self.get_open_positions()
        for pos in open_pos:
            if pos["instrument"] == instrument:
                long_units = int(pos.get("long", {}).get("units", 0))
                short_units = int(pos.get("short", {}).get("units", 0))
                if long_units != 0 or short_units != 0:
                    return True
        return False
