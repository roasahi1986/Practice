from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

@dataclass
class BidRequest:
    """Encapsulates the context for a single bid request."""
    initial_budget: float
    total_duration: int
    remaining_budget: float
    remaining_time: int
    winner_price_percentiles: Dict[int, float]
    conversion_rate: float

@dataclass
class BidResult:
    """Result of a bidding action."""
    bid_price: float
    is_win: bool
    cost: float
    is_conversion: bool

@dataclass
class StrategyMetadata:
    """Metadata for a generated strategy."""
    id: str
    name: str
    strategy_type: str  # e.g., "Impression-Focused", "Aggressive"
    code: str
    created_at: float
    metrics: Optional[Dict[str, float]] = None

class IBiddingStrategy(ABC):
    """Interface for a bidding strategy wrapper."""
    
    @abstractmethod
    def bid(self, request: BidRequest) -> float:
        """Calculates the bid price based on the request context."""
        pass

class ILLMClient(ABC):
    """Interface for interacting with the LLM."""
    
    @abstractmethod
    def generate_strategy_code(self, prompt: str) -> str:
        """Generates strategy code based on a prompt."""
        pass

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generates raw text based on a prompt."""
        pass

    @abstractmethod
    def analyze_strategies(self, strategies_data: List[Dict[str, Any]]) -> str:
        """Analyzes a list of strategies and returns insights."""
        pass

class IReplayEngine(ABC):
    """Interface for the replay engine."""
    
    @abstractmethod
    def run(self, strategy: IBiddingStrategy, data_iterator: Any) -> Dict[str, Any]:
        """Runs the simulation for a given strategy over the data."""
        pass

class IDataGenerator(ABC):
    """Interface for data generation/loading."""
    
    @abstractmethod
    def generate_data(self, num_records: int) -> Any:
        pass
    
    @abstractmethod
    def load_data(self, source: str) -> Any:
        pass

