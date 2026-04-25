"""
utils.py — Shared utilities: notifications, state management, logging.
"""
import os
import json
import requests
import logging
from datetime import datetime


def setup_logger(name: str, log_dir: str = None) -> logging.Logger:
    """Create a logger that writes to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on Colab cell re-runs
    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (if log_dir provided)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class Notifier:
    """Send trade alerts and system notifications via Discord/Slack webhook."""

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url
        self.logger = logging.getLogger("notifier")

    def send(self, message: str, level: str = "info"):
        """Send a notification. level can be info, warning, error, trade."""
        icons = {"info": "ℹ️", "warning": "⚠️", "error": "🚨", "trade": "💹"}
        icon = icons.get(level, "🤖")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"{icon} **[{timestamp}]** {message}"

        self.logger.info(formatted)

        if self.webhook_url:
            try:
                payload = {"content": formatted}
                requests.post(
                    self.webhook_url,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
            except Exception as e:
                self.logger.error(f"Webhook send failed: {e}")


class StateManager:
    """
    Manages persistent state across Colab sessions.

    Saves/loads model weights, trade logs, and configuration to either
    Google Drive (in Colab) or a local directory (for testing).
    """

    def __init__(self):
        self.is_colab = os.path.exists("/content")

        if self.is_colab:
            from src.config import DRIVE_STATE_DIR, DRIVE_MODELS_DIR, DRIVE_LOGS_DIR, DRIVE_DATA_DIR
            self.base = DRIVE_STATE_DIR
            self.models_dir = DRIVE_MODELS_DIR
            self.logs_dir = DRIVE_LOGS_DIR
            self.data_dir = DRIVE_DATA_DIR
        else:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "local_state"))
            self.base = base
            self.models_dir = os.path.join(base, "models")
            self.logs_dir = os.path.join(base, "logs")
            self.data_dir = os.path.join(base, "data")

        for d in [self.models_dir, self.logs_dir, self.data_dir]:
            os.makedirs(d, exist_ok=True)

    def model_path(self, filename: str) -> str:
        return os.path.join(self.models_dir, filename)

    def log_path(self, filename: str) -> str:
        return os.path.join(self.logs_dir, filename)

    def data_path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)


class TradeJournal:
    """
    Logs every trade decision for walk-forward analysis and self-improvement.

    Each entry records: timestamp, instrument, signal, confidence, regime,
    entry price, SL, TP, position size, and (later) the actual outcome.
    """

    def __init__(self, state_mgr: StateManager):
        self.state = state_mgr
        self.journal_file = state_mgr.log_path("trade_journal.json")
        self.trades = self._load()

    def _load(self) -> list:
        if os.path.exists(self.journal_file):
            with open(self.journal_file, "r") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.journal_file, "w") as f:
            json.dump(self.trades, f, indent=2, default=str)

    def log_trade(self, entry: dict):
        """Log a trade decision."""
        entry["timestamp"] = datetime.now().isoformat()
        self.trades.append(entry)
        self._save()

    def log_outcome(self, trade_id: int, outcome: dict):
        """Update a trade with its actual outcome (for retraining feedback)."""
        if 0 <= trade_id < len(self.trades):
            self.trades[trade_id].update(outcome)
            self._save()

    def get_recent(self, n: int = 50) -> list:
        """Get the last N trades for analysis."""
        return self.trades[-n:]

    def get_performance_stats(self) -> dict:
        """Calculate running performance statistics."""
        closed = [t for t in self.trades if "pnl" in t]
        if not closed:
            return {"total_trades": 0}

        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "total_trades": len(closed),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "total_pnl": sum(pnls),
            "max_drawdown": self._calc_max_drawdown(pnls),
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
        }

    def _calc_max_drawdown(self, pnls: list) -> float:
        """Calculate maximum drawdown from PnL series."""
        cumulative = 0
        peak = 0
        max_dd = 0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd
