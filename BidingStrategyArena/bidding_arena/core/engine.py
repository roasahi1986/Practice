import pandas as pd
from typing import Dict, Any, List
from .interfaces import IReplayEngine, IBiddingStrategy, BidRequest, BidResult
from ..data.generator import MockDataGenerator

class ReplayEngine(IReplayEngine):
    """
    Simulates the bidding process against historical data.
    """
    
    def __init__(self, initial_budget: float = 1000.0):
        self.initial_budget = initial_budget

    def run(self, strategy: IBiddingStrategy, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs the simulation.
        
        Args:
            strategy: The strategy to evaluate.
            data: A DataFrame containing 'timestamp', 'winner_price', 'is_conversion', etc.
        """
        
        # Prepare simulation state
        remaining_budget = self.initial_budget
        
        if data.empty:
            return {"error": "No data provided"}

        start_time = data['timestamp'].min()
        end_time = data['timestamp'].max()
        total_duration = end_time - start_time
        if total_duration == 0: total_duration = 1 # Avoid div by zero

        # Metrics
        win_count = 0
        conversion_count = 0
        total_spend = 0.0
        bids_placed = 0
        
        # New: Track moving average bid
        interval_bid_sum = 0.0
        interval_bid_count = 0
        
        history: List[Dict[str, Any]] = []

        # Pre-calculate global stats for the strategy context (simplified for now)
        # In a real scenario, this might be segmented or moving average.
        # We'll use the whole dataset stats as "historical knowledge" for the strategy
        # to simplify the 'training' phase assumption.
        percentiles = MockDataGenerator.get_percentiles(data)
        conversion_rate = MockDataGenerator.get_conversion_rate(data)

        for index, row in data.iterrows():
            current_time = row['timestamp']
            elapsed = current_time - start_time
            remaining_time = total_duration - elapsed
            
            # Stop if budget exhausted
            if remaining_budget <= 0:
                break

            request = BidRequest(
                initial_budget=self.initial_budget,
                total_duration=int(total_duration),
                remaining_budget=remaining_budget,
                remaining_time=int(remaining_time),
                winner_price_percentiles=percentiles,
                conversion_rate=conversion_rate 
            )

            # Execute Strategy
            bid_price = strategy.bid(request)
            bids_placed += 1
            
            # Accumulate for moving average
            interval_bid_sum += bid_price
            interval_bid_count += 1
            
            # Logic: First Price Auction
            # We assume bid_price is the actual cost for this single impression.
            # Winner price in the data is the market floor price (others' max bid).
            # Win condition: My Bid >= Market Winner Price
            # Cost: My Bid
            
            is_win = bid_price >= row['winner_price']
            cost = 0.0
            
            if is_win:
                # First Price Auction: Pay what you bid
                # (Or Second Price: Pay winner_price + epsilon? But usually simulation uses First Price or just Pay Bid for simplicity/risk)
                # Given user query: "cost is strategy bid price", implying First Price Auction logic.
                cost = bid_price 
                
                if remaining_budget >= cost:
                    remaining_budget -= cost
                    win_count += 1
                    total_spend += cost
                    if row['is_conversion'] == 1:
                        conversion_count += 1
                else:
                    # Not enough budget to pay
                    is_win = False
                    cost = 0.0

            # Record history periodically (every 100 or so) to save memory
            if index % 50 == 0:
                # New: Calculate average bid for this interval
                avg_bid = interval_bid_sum / interval_bid_count if interval_bid_count > 0 else 0.0
                
                history.append({
                    "timestamp": current_time,
                    "remaining_budget": remaining_budget,
                    "win_count": win_count,
                    "conversion_count": conversion_count,
                    "total_spend": total_spend,
                    "avg_bid_price": avg_bid # Add to history
                })
                
                # Reset counters
                interval_bid_sum = 0.0
                interval_bid_count = 0

        # Final Metrics
        win_rate = win_count / bids_placed if bids_placed > 0 else 0
        avg_cpm = (total_spend * 1000 / win_count) if win_count > 0 else 0
        avg_cpa = (total_spend / conversion_count) if conversion_count > 0 else 0
        
        return {
            "win_count": win_count,
            "bids_placed": bids_placed,
            "win_rate": win_rate,
            "conversion_count": conversion_count,
            "total_spend": total_spend,
            "remaining_budget": remaining_budget,
            "avg_cpm": avg_cpm,
            "avg_cpa": avg_cpa,
            "history": history
        }

