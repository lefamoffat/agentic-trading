"""Unit tests for PortfolioTracker core financial logic."""

import pytest

from src.environment.state.portfolio import PortfolioTracker
from src.environment.config import FeeStructure


class TestPortfolioTracker:
    """Ensure transaction-cost and balance logic behaves correctly."""

    def test_initialisation_requires_positive_balance(self):
        """Constructor must reject non-positive initial balances."""
        with pytest.raises(ValueError):
            PortfolioTracker(initial_balance=0.0, fee_structure=FeeStructure.SPREAD_BASED)

    @pytest.mark.parametrize(
        "fee_structure,spread,commission,trade_value,expected_cost",
        [
            (FeeStructure.SPREAD_BASED, 0.0002, 0.0, 100_000, 100_000 * 0.0002 / 2),
            (FeeStructure.COMMISSION, 0.0, 0.001, 50_000, 50_000 * 0.001),
            (FeeStructure.COMBINED, 0.0001, 0.0005, 80_000, (80_000 * 0.0001 / 2) + (80_000 * 0.0005)),
        ],
    )
    def test_calculate_transaction_cost(self, fee_structure, spread, commission, trade_value, expected_cost):
        tracker = PortfolioTracker(
            initial_balance=10_000,
            fee_structure=fee_structure,
            spread=spread,
            commission_rate=commission,
        )
        cost = tracker.calculate_transaction_cost(trade_value)
        assert abs(cost - expected_cost) < 1e-6

    def test_apply_trade_result_updates_balance_and_stats(self):
        tracker = PortfolioTracker(initial_balance=10_000, fee_structure=FeeStructure.SPREAD_BASED, spread=0.0001)

        initial_balance = tracker.balance
        profit = 500.0  # gross profit in quote currency
        trade_value = 100_000.0
        expected_fee = trade_value * tracker.spread / 2
        net_profit = profit - expected_fee

        returned_profit = tracker.apply_trade_result(profit, trade_value)

        # Returned value should match net profit
        assert abs(returned_profit - net_profit) < 1e-6
        # Balance updated
        assert abs(tracker.balance - (initial_balance + net_profit)) < 1e-6
        # Stats updated
        assert tracker.total_trades == 1
        assert abs(tracker.total_fees_paid - expected_fee) < 1e-6

    def test_position_size_fixed_method(self):
        tracker = PortfolioTracker(initial_balance=10_000, fee_structure=FeeStructure.SPREAD_BASED)
        price = 1.2500
        size = tracker.calculate_position_size(price)
        assert abs(size - (tracker.balance / price)) < 1e-6

    def test_total_return_property(self):
        tracker = PortfolioTracker(initial_balance=10_000, fee_structure=FeeStructure.SPREAD_BASED)

        # Artificially modify balance to simulate returns
        tracker.balance = 11_000
        assert abs(tracker.total_return - 10.0) < 1e-6  # 10% return

        tracker.balance = 9_000
        assert abs(tracker.total_return - (-10.0)) < 1e-6 