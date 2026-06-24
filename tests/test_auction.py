"""Tests for double-sided auction and market clearing."""

import pytest
from core.auction import DoubleSidedAuction, Bid, MarketClearing


class TestDoubleSidedAuction:
    def test_empty_bids_returns_default_price(self, auction):
        clearing = auction.clear_market([])
        assert clearing.clearing_price == 50.0
        assert clearing.total_traded_mwh == 0

    def test_single_buy_no_sell_no_trade(self, auction):
        bids = [Bid("Buyer", "consumer", 60.0, 100.0, True)]
        clearing = auction.clear_market(bids)
        assert clearing.total_traded_mwh == 0

    def test_single_sell_no_buy_no_trade(self, auction):
        bids = [Bid("Seller", "solar", 10.0, 100.0, False)]
        clearing = auction.clear_market(bids)
        assert clearing.total_traded_mwh == 0

    def test_simple_trade(self, auction):
        bids = [
            Bid("Buyer", "consumer", 60.0, 100.0, True),
            Bid("Seller", "solar", 10.0, 100.0, False),
        ]
        clearing = auction.clear_market(bids)
        assert clearing.total_traded_mwh == 100.0
        assert clearing.clearing_price == 35.0  # (60+10)/2

    def test_partial_fill(self, auction):
        bids = [
            Bid("Buyer", "consumer", 60.0, 50.0, True),
            Bid("Seller", "solar", 10.0, 100.0, False),
        ]
        clearing = auction.clear_market(bids)
        assert clearing.total_traded_mwh == 50.0

    def test_no_match_price_gap(self, auction):
        bids = [
            Bid("Buyer", "consumer", 10.0, 100.0, True),
            Bid("Seller", "solar", 60.0, 100.0, False),
        ]
        clearing = auction.clear_market(bids)
        assert clearing.total_traded_mwh == 0

    def test_multiple_buyers_priority(self, auction):
        bids = [
            Bid("Buyer1", "consumer", 80.0, 50.0, True),
            Bid("Buyer2", "consumer", 40.0, 50.0, True),
            Bid("Seller", "solar", 10.0, 100.0, False),
        ]
        clearing = auction.clear_market(bids)
        # Seller has 100 MW, Buyer1 takes 50, Buyer2 takes 50
        assert clearing.total_traded_mwh == 100.0
        assert clearing.transactions[0]["buyer"] == "Buyer1"

    def test_carbon_cost_applied(self, auction):
        bids = [
            Bid("Buyer", "consumer", 60.0, 100.0, True),
            Bid("Coal", "coal", 20.0, 100.0, False),
        ]
        clearing = auction.clear_market(bids)
        assert clearing.carbon_cost_total > 0

    def test_price_history_tracked(self, auction):
        bids = [
            Bid("Buyer", "consumer", 60.0, 100.0, True),
            Bid("Seller", "solar", 10.0, 100.0, False),
        ]
        auction.clear_market(bids)
        assert len(auction.price_history) == 1
        assert auction.price_history[0] == 35.0

    def test_price_trend_detection(self, auction):
        auction.price_history = [30.0, 32.0]
        assert auction.get_price_trend() == "rising"
        auction.price_history = [30.0, 28.0]
        assert auction.get_price_trend() == "falling"
        auction.price_history = [30.0, 30.5]
        assert auction.get_price_trend() == "stable"

    def test_clearing_price_clamped(self, auction):
        bids = [
            Bid("Buyer", "consumer", 200.0, 100.0, True),
            Bid("Seller", "solar", 150.0, 100.0, False),
        ]
        clearing = auction.clear_market(bids)
        assert clearing.clearing_price <= 120.0

    def test_transaction_details(self, auction):
        bids = [
            Bid("Buyer", "consumer", 60.0, 100.0, True),
            Bid("Seller", "solar", 10.0, 100.0, False),
        ]
        clearing = auction.clear_market(bids)
        tx = clearing.transactions[0]
        assert tx["buyer"] == "Buyer"
        assert tx["seller"] == "Seller"
        assert tx["quantity_mwh"] == 100.0
        assert "total_cost" in tx
        assert "carbon_kg" in tx