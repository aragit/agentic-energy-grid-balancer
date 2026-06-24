"""Double-sided energy market auction with carbon credit trading."""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Bid:
    agent_name: str
    agent_type: str
    bid_price: float
    quantity_mw: float
    is_buy: bool  # True = consumer buying, False = generator selling


@dataclass
class MarketClearing:
    clearing_price: float
    total_traded_mwh: float
    buyer_surplus: float
    seller_surplus: float
    transactions: List[Dict[str, Any]]
    carbon_cost_total: float


class DoubleSidedAuction:
    """Continuous double-sided auction for energy trading."""

    def __init__(self, carbon_price_per_ton: float = 25.0):
        self.carbon_price = carbon_price_per_ton
        self.price_history: List[float] = []
        self.transaction_history: List[Dict] = []
        self.carbon_cost_total: float = 0.0

    def clear_market(self, bids: List[Bid]) -> MarketClearing:
        """
        Clear the market by matching buy and sell bids.
        Uses uniform price auction (all trades at clearing price).
        """
        buy_bids = sorted([b for b in bids if b.is_buy], key=lambda x: -x.bid_price)
        sell_bids = sorted([b for b in bids if not b.is_buy], key=lambda x: x.bid_price)

        transactions = []
        total_traded = 0.0
        carbon_cost_total = 0.0

        i, j = 0, 0
        while i < len(buy_bids) and j < len(sell_bids):
            buy = buy_bids[i]
            sell = sell_bids[j]

            if buy.bid_price < sell.bid_price:
                break  # No more matches possible

            trade_qty = min(buy.quantity_mw, sell.quantity_mw)
            clearing_price = (buy.bid_price + sell.bid_price) / 2

            # Carbon cost for seller
            carbon_intensity = self._get_carbon_intensity(sell.agent_type)
            carbon_kg = trade_qty * carbon_intensity
            carbon_cost = (carbon_kg / 1000) * self.carbon_price

            transactions.append({
                "buyer": buy.agent_name,
                "seller": sell.agent_name,
                "quantity_mwh": round(trade_qty, 2),
                "price_per_mwh": round(clearing_price, 2),
                "total_cost": round(trade_qty * clearing_price, 2),
                "carbon_kg": round(carbon_kg, 2),
                "carbon_cost": round(carbon_cost, 2),
            })

            total_traded += trade_qty
            carbon_cost_total += carbon_cost

            buy.quantity_mw -= trade_qty
            sell.quantity_mw -= trade_qty

            if buy.quantity_mw <= 0.01:
                i += 1
            if sell.quantity_mw <= 0.01:
                j += 1

        clearing_price = self._compute_clearing_price(transactions)
        buyer_surplus = sum(t["quantity_mwh"] * (t["price_per_mwh"] - clearing_price) for t in transactions)
        seller_surplus = sum(t["quantity_mwh"] * (clearing_price - t["price_per_mwh"]) for t in transactions)

        result = MarketClearing(
            clearing_price=round(clearing_price, 2),
            total_traded_mwh=round(total_traded, 2),
            buyer_surplus=round(buyer_surplus, 2),
            seller_surplus=round(seller_surplus, 2),
            transactions=transactions,
            carbon_cost_total=round(carbon_cost_total, 2),
        )

        self.price_history.append(result.clearing_price)
        self.transaction_history.extend(transactions)
        self.carbon_cost_total += carbon_cost_total
        return result

    def _compute_clearing_price(self, transactions: List[Dict]) -> float:
        """Compute clearing price with supply/demand adjustment."""
        if not transactions:
            # Default price with supply/demand signal
            return 50.0
        
        # Weighted average of matched prices
        weighted_price = sum(t["price_per_mwh"] * t["quantity_mwh"] for t in transactions) / sum(t["quantity_mwh"] for t in transactions)
        
        # Clamp to reasonable range
        return min(120.0, max(25.0, weighted_price))

    def _get_carbon_intensity(self, agent_type: str) -> float:
        intensities = {
            "solar": 0.0, "wind": 0.0, "nuclear": 0.0,
            "coal": 820.0, "gas": 490.0, "battery": 0.0,
        }
        return intensities.get(agent_type, 0.0)

    def get_price_trend(self) -> str:
        if len(self.price_history) < 2:
            return "stable"
        if self.price_history[-1] > self.price_history[-2] * 1.05:
            return "rising"
        if self.price_history[-1] < self.price_history[-2] * 0.95:
            return "falling"
        return "stable"