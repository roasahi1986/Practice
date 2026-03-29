from ..core.constants import StrategyType

class PromptBuilder:
    @staticmethod
    def build(strategy_type: StrategyType) -> str:
        base_prompt = """
You are an expert in RTB (Real-Time Bidding) algorithms.
Your goal is to write a Python function `bidding_strategy` that determines the bid price for an ad request.

### Function Signature
```python
import math

def bidding_strategy(
    initial_budget: float,           # Initial budget (USD)
    total_duration: int,             # Total duration (seconds)
    remaining_budget: float,         # Remaining budget
    remaining_time: int,             # Remaining time (seconds)
    winner_price_percentiles: dict,  # {10: p10, ..., 50: median, ..., 90: p90} (Market Price)
    conversion_rate: float           # Historical conversion rate (0.0 to 1.0)
) -> float:                          # Return bid price (USD)
```

### Constraints
1.  **Output**: ONLY the Python code. No markdown formatting, no explanations.
2.  **Libraries**: Use ONLY `math`. No `numpy`, `pandas`, etc.
3.  **Performance**: Must run in < 1ms. Avoid loops if possible.
4.  **Safety**: Bid must be >= 0.

### Strategy Requirements
"""
        
        specific_instructions = {
            StrategyType.IMPRESSION_FOCUSED: """
**Type: Impression-Focused**
- **Goal**: Maximize the number of won auctions (impressions).
- **Tactic**: Bid around the 40th-60th percentile of historical winner prices.
- **Budget**: Spend budget evenly over the remaining time.
""",
            StrategyType.CONVERSION_FOCUSED: """
**Type: Conversion-Focused**
- **Goal**: Maximize total conversions.
- **Tactic**: Bid aggressively (e.g., 70-80th percentile) ONLY when `conversion_rate` is high relative to average.
- **Budget**: Conserve budget for high-value opportunities.
""",
            StrategyType.AGGRESSIVE: """
**Type: Aggressive**
- **Goal**: Spend budget quickly and win high-value inventory.
- **Tactic**: Bid high (e.g., 80-90th percentile).
- **Budget**: Front-load spending. If time is running out, increase bids further.
""",
            StrategyType.CONSERVATIVE: """
**Type: Conservative**
- **Goal**: Minimize CPA (Cost Per Action) and strictly control ROI.
- **Tactic**: Bid low (e.g., < 50th percentile). Only bid if `conversion_rate` is very promising.
- **Budget**: Keep a safety buffer. It's okay to under-spend.
""",
            StrategyType.ADAPTIVE: """
**Type: Adaptive**
- **Goal**: Dynamically adjust based on remaining resources.
- **Tactic**: Start conservative. If `remaining_budget` is high relative to `remaining_time`, become aggressive.
""",
            StrategyType.HYBRID: """
**Type: Hybrid**
- **Goal**: Balance volume and efficiency.
- **Tactic**: Use a weighted score of `conversion_rate` and price percentiles.
"""
        }

        return base_prompt + specific_instructions.get(strategy_type, "") + "\n\nRETURN ONLY THE CODE."

    @staticmethod
    def build_analysis_prompt(strategy_code: str, metrics: dict) -> str:
        return f"""
You are a senior data scientist specializing in AdTech and Bidding Strategies.
Analyze the following bidding strategy code and its performance metrics.

### Strategy Code
```python
{strategy_code}
```

### Performance Metrics
{metrics}

### Task
Identify 2-3 key strengths and 2-3 weaknesses of this strategy based on the metrics.
Explain WHY it performed this way (e.g., did it bid too high and run out of budget? did it bid too low and miss conversions?).
Provide concrete suggestions for improvement.

### Guidelines
- **Be concise and professional.**
- **STRICTLY limit your response to under 50 words.**
- **Use short bullet points only.**
- **NO background information** (e.g., "RTB is...").
- **NO filler text.**
- Focus strictly on interpreting the data and code logic.
"""

    @staticmethod
    def build_optimization_prompt(original_code: str, analysis: str, history_context: str = "") -> str:
         return f"""
You are an expert Python developer for Real-Time Bidding systems.
Your task is to REWRITE the following bidding strategy to improve its performance, based on the provided analysis.

### History of Strategy Evolution
{history_context}

### Original Strategy (Latest Version)
```python
{original_code}
```

### Analysis & Recommendations (Latest Version)
{analysis}

### Requirements
1.  **Goal**: Maximize Conversions (Total number of conversions) while managing budget.
2.  **Adaptation Strategy**: If previous performance was poor (e.g., extremely low budget utilization or negligible conversions), perform a RADICAL architectural change rather than minor tweaks.
3.  **Function Signature**: MUST match the original signature exactly:
```python
def bidding_strategy(
    initial_budget: float,
    total_duration: int,
    remaining_budget: float,
    remaining_time: int,
    winner_price_percentiles: dict,
    conversion_rate: float
) -> float:
```
4.  **Constraints**: 
    - Use ONLY `math` library.
    - Run in < 1ms.
    - Bid >= 0.
5.  **Output**: ONLY the Python code for the function `bidding_strategy`. No markdown, no comments outside the code.
"""

