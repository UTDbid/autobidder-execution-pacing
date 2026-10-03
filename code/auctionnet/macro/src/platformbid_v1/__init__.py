"""Auditable synchronized-market evaluation for delegated auto-bidding."""

from .auction import AuctionOutcome, uniform_third_price_top2
from .design import DesignCell, make_design_cells, select_adopters

__all__ = [
    "AuctionOutcome",
    "DesignCell",
    "make_design_cells",
    "select_adopters",
    "uniform_third_price_top2",
]
