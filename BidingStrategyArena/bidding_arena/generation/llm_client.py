import time
from ..core.interfaces import ILLMClient
from typing import List, Dict, Any

class MockLLMClient(ILLMClient):
    """Mock LLM client for testing without Ollama."""
    
    def generate_strategy_code(self, prompt: str) -> str:
        """Returns a dummy strategy based on keywords in the prompt."""
        
        # Simple template for a valid strategy
        base_code = """
import math

def bidding_strategy(
    initial_budget: float,
    total_duration: int,
    remaining_budget: float,
    remaining_time: int,
    winner_price_percentiles: dict,
    conversion_rate: float
) -> float:
"""
        
        if "Aggressive" in prompt:
            # Aggressive: Use high percentile
            body = """
    # Aggressive strategy
    percentile_90 = winner_price_percentiles.get(90, 10.0)
    bid = percentile_90 * 1.1
    if remaining_time < total_duration * 0.2:
        bid *= 1.5
    return max(0.0, bid)
"""
        elif "Conservative" in prompt:
            # Conservative: Use median, ROI check
            body = """
    # Conservative strategy
    median_price = winner_price_percentiles.get(50, 5.0)
    bid = median_price * 0.8
    if conversion_rate < 0.01:
        bid = 0.0
    return max(0.0, bid)
"""
        else:
            # Default/Random
            body = """
    # Default strategy
    avg_price = winner_price_percentiles.get(50, 5.0)
    return max(0.1, avg_price)
"""
        return base_code + body

    def generate_text(self, prompt: str) -> str:
        """Mock text generation."""
        return "Mock analysis: The strategy is aggressive but runs out of budget too fast. Suggest lowering bids in low conversion periods."

    def analyze_strategies(self, strategies_data: List[Dict[str, Any]]) -> str:
        return "Mock Analysis: Aggressive strategies performed best in high conversion segments."

class OllamaLLMClient(ILLMClient):
    """Real implementation connecting to Ollama."""
    
    def __init__(self, model: str = "gemma3:12b"):
        try:
            import ollama
            self.client = ollama
        except ImportError:
            self.client = None
        self.model = model

    def generate_text(self, prompt: str) -> str:
        if not self.client:
            raise RuntimeError("Ollama library not installed.")
        
        response = self.client.generate(model=self.model, prompt=prompt)
        return response['response']

    def generate_strategy_code(self, prompt: str) -> str:
        if not self.client:
            raise RuntimeError("Ollama library not installed.")
        
        response = self.client.generate(model=self.model, prompt=prompt)
        content = response['response']
        
        # Simple extraction of code block
        if "```python" in content:
            code = content.split("```python")[1].split("```")[0]
        elif "```" in content:
            code = content.split("```")[1].split("```")[0]
        else:
            code = content
            
        return code.strip()

    def analyze_strategies(self, strategies_data: List[Dict[str, Any]]) -> str:
        if not self.client:
            raise RuntimeError("Ollama library not installed.")
        
        prompt = f"Analyze these bidding strategies results: {strategies_data}"
        response = self.client.generate(model=self.model, prompt=prompt)
        return response['response']

