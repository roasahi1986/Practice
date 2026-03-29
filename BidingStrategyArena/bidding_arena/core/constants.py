from enum import Enum

class StrategyType(Enum):
    IMPRESSION_FOCUSED = "Impression-Focused"
    CONVERSION_FOCUSED = "Conversion-Focused"
    AGGRESSIVE = "Aggressive"
    CONSERVATIVE = "Conservative"
    ADAPTIVE = "Adaptive"
    HYBRID = "Hybrid"

