"""
execution.py — Trade execution and risk management via Capital.com API.

Handles:
  - Market order execution (POST /api/v1/positions)
  - Position management (GET/DELETE /api/v1/positions)
  - Deal confirmation (GET /api/v1/confirms/{dealReference})
  - Fractional Kelly position sizing
  - ATR-based stop loss / take profit
  - Maximum drawdown circuit breaker
"""
import os
import time
import logging
import numpy as np

from src.config import (
    RISK_PER_TRADE, MAX_DRAWDOWN, KELLY_FRACTION,
    SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER,
    MIN_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger("execution")


# ──────────────────────────────────────────────────────────
# Trade Executor — Capital.com
# ──────────────────────────────────────────────────────────

class CapitalExecutor:
    """
    Executes trades via Capital.com REST API.
    Uses the CapitalClient from data_fetcher for session management.
    """

    def __init__(self, client=None):
        """
        Parameters
        ----------
        client : CapitalClient, optional
            If not provided, creates a new one from env vars.
        """
        if client is None:
            from src.data_fetcher import CapitalClient
            self.client = CapitalClient(demo=True)
        else:
            self.client = client

    def execute_market_order(
        self,
        instrument: str,
        size: float,
        signal: str,
        price: float,
        atr: float,
    ) -> dict:
        """
        Execute a market order on Capital.com.

        POST /api/v1/positions
        {
            "epic": "EURUSD",
            "direction": "BUY" | "SELL",
            "size": 1,
            "guaranteedStop": false,
            "stopLevel": ...,
            "profitLevel": ...
        }
        """
        epic = self.client.to_epic(instrument)
        direction = signal.upper()

        if direction not in ("BUY", "SELL"):
            return {"error": f"Invalid signal: {signal}"}

        # Calculate SL/TP based on ATR
        if direction == "BUY":
            stop_level = round(price - (atr * SL_ATR_MULTIPLIER), 5)
            profit_level = round(price + (atr * TP_ATR_MULTIPLIER), 5)
        else:
            stop_level = round(price + (atr * SL_ATR_MULTIPLIER), 5)
            profit_level = round(price - (atr * TP_ATR_MULTIPLIER), 5)

        order_data = {
            "epic": epic,
            "direction": direction,
            "size": max(round(size, 2), 0.01),  # Capital.com min size
            "guaranteedStop": False,
            "stopLevel": stop_level,
            "profitLevel": profit_level,
        }

        logger.info(f"Executing {direction} {size} {instrument} @ ~{price:.5f}")
        result = self.client._post("/api/v1/positions", order_data)

        if result is None:
            return {"error": "API request failed"}

        if "error" in result:
            logger.error(f"Order failed: {result['error']}")
            return result

        deal_ref = result.get("dealReference", "")
        if deal_ref:
            # Wait briefly and confirm the deal
            time.sleep(0.5)
            confirmation = self._confirm_deal(deal_ref)
            return {
                "dealReference": deal_ref,
                "stop_loss": stop_level,
                "take_profit": profit_level,
                "confirmation": confirmation,
            }

        return {
            "dealReference": "unknown",
            "stop_loss": stop_level,
            "take_profit": profit_level,
            "raw_response": result,
        }

    def _confirm_deal(self, deal_reference: str) -> dict:
        """GET /api/v1/confirms/{dealReference} — verify order was filled."""
        result = self.client._get(f"/api/v1/confirms/{deal_reference}")
        if result:
            status = result.get("dealStatus", "UNKNOWN")
            logger.info(f"Deal {deal_reference}: {status}")
            return result
        return {"dealStatus": "UNKNOWN"}

    def has_open_position(self, instrument: str) -> bool:
        """Check if there's an existing open position for this instrument."""
        epic = self.client.to_epic(instrument)
        result = self.client._get("/api/v1/positions")
        if result and "positions" in result:
            for pos in result["positions"]:
                if pos.get("market", {}).get("epic") == epic:
                    return True
        return False

    def close_position(self, deal_id: str) -> dict:
        """DELETE /api/v1/positions/{dealId} — close a specific position."""
        result = self.client._delete(f"/api/v1/positions/{deal_id}")
        if result:
            logger.info(f"Position {deal_id} closed.")
            return result
        return {"error": f"Failed to close position {deal_id}"}

    def get_all_positions(self) -> list:
        """GET /api/v1/positions — list all open positions."""
        result = self.client._get("/api/v1/positions")
        if result and "positions" in result:
            return result["positions"]
        return []


# ──────────────────────────────────────────────────────────
# Risk Manager (unchanged from OANDA version)
# ──────────────────────────────────────────────────────────

class RiskManager:
    """
    Institutional-grade risk management.

    Implements:
      - Fractional Kelly Criterion position sizing
      - ATR-based dynamic stop loss / take profit
      - Maximum drawdown circuit breaker
      - Pre-trade risk gateway (edge-after-costs filter)
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = RISK_PER_TRADE,
        max_drawdown: float = MAX_DRAWDOWN,
        kelly_fraction: float = KELLY_FRACTION,
    ):
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.kelly_fraction = kelly_fraction
        self.is_halted = False

    def update_balance(self, balance: float):
        """Update the current balance and check for drawdown breach."""
        self.current_balance = balance
        if balance > self.peak_balance:
            self.peak_balance = balance

        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if drawdown >= self.max_drawdown:
            self.is_halted = True
            logger.critical(
                f"CIRCUIT BREAKER: Drawdown {drawdown:.2%} >= {self.max_drawdown:.2%}. "
                f"Trading HALTED."
            )

    def calculate_position_size(
        self,
        balance: float,
        atr: float,
        price: float,
        win_rate: float = 0.52,
        avg_win_loss_ratio: float = 1.5,
        position_scale: float = 1.0,
    ) -> float:
        """
        Calculate position size using Fractional Kelly Criterion.

        Kelly % = W - (1-W)/R
        where W = win rate, R = avg win/loss ratio

        We use FRACTIONAL Kelly (default 1/4) for safety.

        For Capital.com CFDs, size is in contract units (e.g., 1 = 1 contract).
        """
        if atr <= 0 or price <= 0 or balance <= 0:
            return 0

        # Kelly Criterion
        kelly_pct = win_rate - (1 - win_rate) / max(avg_win_loss_ratio, 0.01)
        kelly_pct = max(kelly_pct, 0)  # Don't bet if negative edge
        fractional_kelly = kelly_pct * self.kelly_fraction

        # Risk amount in currency
        risk_amount = balance * min(fractional_kelly, self.risk_per_trade)

        # Position size: risk_amount / (ATR * SL multiplier)
        stop_distance = atr * SL_ATR_MULTIPLIER
        if stop_distance <= 0:
            return 0

        # For CFDs: size in contracts
        size = risk_amount / (stop_distance * price)
        size *= position_scale

        # Capital.com minimum size
        size = max(round(size, 2), 0.01)

        logger.info(
            f"Position sizing: Kelly={kelly_pct:.4f}, Frac={fractional_kelly:.4f}, "
            f"Risk=${risk_amount:.2f}, Size={size:.2f} contracts"
        )
        return size

    def should_trade(
        self, confidence: float, atr: float, spread: float
    ) -> tuple[bool, str]:
        """
        Pre-trade risk gateway.

        Checks:
        1. Confidence threshold
        2. Edge-after-costs (expected profit > transaction cost)
        3. Drawdown halt status
        """
        if self.is_halted:
            return False, "Trading halted — max drawdown breached"

        if confidence < MIN_CONFIDENCE_THRESHOLD:
            return False, f"Low confidence: {confidence:.4f} < {MIN_CONFIDENCE_THRESHOLD}"

        # Edge-after-costs: expected move should be > 2x the spread
        expected_move = atr * 0.5 * confidence
        cost = spread * 2  # Round-trip cost
        if expected_move < cost:
            return False, (
                f"Insufficient edge: expected_move={expected_move:.6f} < "
                f"cost={cost:.6f}"
            )

        return True, "APPROVED"
