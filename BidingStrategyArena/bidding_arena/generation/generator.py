import time
import uuid
from typing import Optional
from ..core.interfaces import StrategyMetadata, ILLMClient
from ..core.constants import StrategyType
from .prompts import PromptBuilder
from .validator import CodeValidator

class StrategyGenerator:
    def __init__(self, llm_client: ILLMClient):
        self.llm_client = llm_client
    
    def generate(self, strategy_type: StrategyType, retries: int = 3) -> StrategyMetadata:
        """Generates a valid bidding strategy code."""
        prompt = PromptBuilder.build(strategy_type)
        
        last_error = None
        
        for i in range(retries):
            try:
                code = self.llm_client.generate_strategy_code(prompt)
                
                # Basic cleanup if LLM returns markdown blocks
                code = self._clean_code(code)

                if CodeValidator.validate(code):
                    return StrategyMetadata(
                        id=str(uuid.uuid4()),
                        name=f"{strategy_type.value}_{int(time.time())}",
                        strategy_type=strategy_type.value,
                        code=code,
                        created_at=time.time()
                    )
            except Exception as e:
                last_error = e
                print(f"Attempt {i+1} failed: {e}")
                continue
        
        raise ValueError(f"Failed to generate valid strategy for {strategy_type.value} after {retries} attempts. Last error: {last_error}")

    def analyze_and_optimize(self, strategy: StrategyMetadata, metrics: dict, history_context: str = "", retries: int = 3) -> tuple[str, StrategyMetadata]:
        # 1. Analyze
        analysis_prompt = PromptBuilder.build_analysis_prompt(strategy.code, metrics)
        analysis = self.llm_client.generate_text(analysis_prompt)
        
        # 2. Optimize
        optimize_prompt = PromptBuilder.build_optimization_prompt(strategy.code, analysis, history_context)
        
        last_error = None
        for i in range(retries):
            try:
                code = self.llm_client.generate_strategy_code(optimize_prompt)
                code = self._clean_code(code)
                
                if CodeValidator.validate(code):
                     optimized_meta = StrategyMetadata(
                        id=str(uuid.uuid4()),
                        name=f"{strategy.name}_optimized",
                        strategy_type="Optimized",
                        code=code,
                        created_at=time.time()
                    )
                     return analysis, optimized_meta
            except Exception as e:
                last_error = e
                continue
        
        raise ValueError(f"Failed to optimize strategy after {retries} attempts. Last error: {last_error}")

    def _clean_code(self, text: str) -> str:
        """Removes markdown code fences if present."""
        if "```python" in text:
            text = text.split("```python")[1]
        elif "```" in text:
            text = text.split("```")[1]
            
        if "```" in text: # closing fence
            text = text.split("```")[0]
            
        return text.strip()

