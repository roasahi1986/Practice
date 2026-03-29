import pytest
from bidding_arena.core.constants import StrategyType
from bidding_arena.generation.generator import StrategyGenerator
from bidding_arena.generation.llm_client import MockLLMClient
from bidding_arena.core.strategy import DynamicStrategy
from bidding_arena.core.engine import ReplayEngine
from bidding_arena.data.generator import MockDataGenerator
from bidding_arena.core.interfaces import BidRequest

def test_mock_generation_and_execution():
    # 1. Setup
    client = MockLLMClient()
    generator = StrategyGenerator(client)
    
    # 2. Generate
    metadata = generator.generate(StrategyType.AGGRESSIVE)
    assert metadata is not None
    assert "def bidding_strategy" in metadata.code
    
    # 3. Compile
    strategy = DynamicStrategy(metadata)
    
    # 4. Bid
    request = BidRequest(
        initial_budget=100.0,
        total_duration=60,
        remaining_budget=100.0,
        remaining_time=60,
        winner_price_percentiles={50: 2.0, 90: 5.0},
        conversion_rate=0.01
    )
    bid = strategy.bid(request)
    assert isinstance(bid, float)
    assert bid >= 0.0

def test_replay_engine():
    # 1. Data
    data_gen = MockDataGenerator()
    data = data_gen.generate_data(num_records=50)
    
    # 2. Strategy
    client = MockLLMClient()
    generator = StrategyGenerator(client)
    metadata = generator.generate(StrategyType.CONSERVATIVE)
    strategy = DynamicStrategy(metadata)
    
    # 3. Engine
    engine = ReplayEngine(initial_budget=50.0)
    results = engine.run(strategy, data)
    
    # 4. Assertions
    assert "win_count" in results
    assert "total_spend" in results
    assert results["bids_placed"] == 50
    assert results["total_spend"] <= 50.0 # Should not overspend

