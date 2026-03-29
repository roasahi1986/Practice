import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Generator
from ..core.interfaces import IDataGenerator

class MockDataGenerator(IDataGenerator):
    """Generates synthetic bid request data."""

    def __init__(self, start_time: int = 1700000000):
        self.start_time = start_time
        
    def generate_data(self, num_records: int = 1000, total_budget: float = 1000.0) -> pd.DataFrame:
        """Generates a DataFrame of synthetic bid logs.
        
        Adjusts the number of records or data such that the sum of winner prices (cost to win all)
        is approximately 80% of the total budget.
        """
        
        platforms = ['iOS', 'Android']
        geos = ['US', 'EU', 'APAC']
        placements = ['Banner', 'Video', 'Interstitial']
        
        # Target total cost logic:
        # We assume 'winner_price' is the actual cost to win one impression (First Price Auction).
        # We want the sum of all winner prices to be approximately 80% of the total budget.
        # This ensures scarcity: the budget can cover most but not all opportunities.
        target_total_cost = total_budget * 0.8
        
        data = []
        current_time = self.start_time
        current_total_cost = 0.0
        
        # Safety break: Allow enough records to likely fill the budget
        # We need roughly budget / min_avg_price records.
        # Let's be safe and allow max(500,000, budget * 10)
        max_records = max(500000, int(total_budget * 10)) 
        
        count = 0
        while current_total_cost < target_total_cost and count < max_records:
            # Time advances slightly
            current_time += random.randint(1, 5)
            
            geo = random.choices(geos, weights=[0.4, 0.3, 0.3])[0]
            platform = random.choice(platforms)
            placement = random.choice(placements)
            
            # Base price logic
            base_price = 1.0
            if geo == 'US': base_price *= 2.0
            if placement == 'Video': base_price *= 3.0
            if platform == 'iOS': base_price *= 1.2
            
            # Winner price: Sample from log-normal distribution
            # This implicitly forms a distribution that has percentiles.
            # The 'winner_price_percentiles' passed to strategy are calculated FROM this generated data.
            # So if we generate using lognormal, the percentiles will reflect lognormal.
            # To ensure the winner price is "sampled from percentiles", we just need to ensure consistency.
            # The current approach (Generate -> Calculate Percentiles -> Pass to Strategy) is correct
            # because the strategy sees the percentiles of the *actual* historical data (which is what we are generating here).
            
            # However, if the user implies we should sample from a *pre-defined* percentile distribution,
            # that's a different requirement. Assuming the current flow:
            # 1. Generate History (this function)
            # 2. Calculate Stats from History (get_percentiles)
            # 3. Pass Stats to Strategy
            # This is consistent.
            
            winner_price = np.random.lognormal(mean=np.log(base_price), sigma=0.5)
            
            # Ensure winner_price is strictly positive
            winner_price = max(0.01, winner_price)
            
            # Cost for this single impression (Directly use price, no CPM division, per user request)
            cost = winner_price
            
            # Conversion probability
            cv_prob = 0.01
            if geo == 'US': cv_prob *= 1.5
            if placement == 'Interstitial': cv_prob *= 2.0
            
            is_conversion = 1 if random.random() < cv_prob else 0
            
            record = {
                'timestamp': current_time,
                'platform': platform,
                'geo': geo,
                'placement_type': placement,
                'winner_price': round(winner_price, 2),
                'is_conversion': is_conversion,
                'segment_id': f"{platform}_{geo}_{placement}"
            }
            data.append(record)
            current_total_cost += cost
            count += 1
            
        return pd.DataFrame(data)

    def load_data(self, source: str) -> pd.DataFrame:
        # In a real system, this would load from a CSV/DB
        if source == "mock":
            return self.generate_data()
        try:
            return pd.read_csv(source)
        except Exception:
            return self.generate_data()

    @staticmethod
    def get_percentiles(df: pd.DataFrame) -> Dict[int, float]:
        """Calculates price percentiles from a dataframe."""
        if df.empty:
            return {p: 0.0 for p in range(10, 100, 10)}
        
        percentiles = {}
        for p in range(10, 100, 10):
            percentiles[p] = float(np.percentile(df['winner_price'], p))
        return percentiles

    @staticmethod
    def get_conversion_rate(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        return float(df['is_conversion'].mean())

