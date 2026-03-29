import math
from typing import Callable, Dict
from .interfaces import IBiddingStrategy, BidRequest, StrategyMetadata

class DynamicStrategy(IBiddingStrategy):
    """
    A strategy that executes dynamically generated Python code.
    """
    def __init__(self, metadata: StrategyMetadata):
        self.metadata = metadata
        self._strategy_func: Callable = self._compile_code(metadata.code)

    def _compile_code(self, code: str) -> Callable:
        """Compiles the code and extracts the 'bidding_strategy' function."""
        # execution context with allowed libraries
        local_scope = {}
        global_scope = {"math": math}
        
        try:
            exec(code, global_scope, local_scope)
        except Exception as e:
            raise ValueError(f"Failed to compile strategy code: {e}")

        if "bidding_strategy" not in local_scope:
            raise ValueError("Code must define a 'bidding_strategy' function.")

        return local_scope["bidding_strategy"]

    def bid(self, request: BidRequest) -> float:
        try:
            # map BidRequest to function arguments
            return float(self._strategy_func(
                initial_budget=request.initial_budget,
                total_duration=request.total_duration,
                remaining_budget=request.remaining_budget,
                remaining_time=request.remaining_time,
                winner_price_percentiles=request.winner_price_percentiles,
                conversion_rate=request.conversion_rate
            ))
        except Exception as e:
            # Fallback or error handling strategy
            print(f"Error in strategy execution: {e}")
            return 0.0

    def get_metadata(self) -> StrategyMetadata:
        return self.metadata

