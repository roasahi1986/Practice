# LLM Bidding Strategy Arena 🤖💸

An AI-powered system that generates, validates, and evaluates Real-Time Bidding (RTB) strategies using Large Language Models (Gemma3 via Ollama).

## 🌟 Features

- **Automated Strategy Generation**: Uses LLMs to write Python bidding code based on high-level goals (e.g., "Maximize Impressions", "Aggressive").
- **Sandboxed Execution**: Validates generated code for safety and syntax.
- **Historical Replay Engine**: Simulates auctions using mock or real historical data.
- **Interactive Arena**: Compare multiple strategies side-by-side with real-time visualization.
- **Design Patterns**: Built with Clean Architecture principles (Strategy, Builder, Adapter, Factory).

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- [Optional] [Ollama](https://ollama.com/) installed and running with `gemma3:12b` model for real AI generation.
  - Run `ollama run gemma3:12b` to pull the model.

### 2. Installation

```bash
pip install -r requirements.txt
```

### 3. Run the Arena

You can launch the web interface using the provided helper script:

```bash
chmod +x run_arena.sh
./run_arena.sh
```

Or manually:

```bash
streamlit run bidding_arena/visualization/app.py
```

## 🏗 System Architecture

- **`core/`**: Interfaces and base logic (Clean Architecture).
- **`generation/`**: LLM interaction, Prompt Builder, and Code Validator.
- **`data/`**: Data generation and loading.
- **`visualization/`**: Streamlit dashboard.

## 🛠 Design Patterns Used

- **Strategy Pattern**: `IBiddingStrategy` allows swapping different bidding logic (generated code) at runtime.
- **Builder Pattern**: `PromptBuilder` constructs complex prompts based on strategy configuration.
- **Adapter Pattern**: `ILLMClient` abstracts the AI provider (Mock vs Ollama).
- **Factory Method**: `StrategyGenerator` encapsulates the creation and validation of strategy objects.

## 🧪 Testing

Run unit tests (coming soon):

```bash
pytest tests/
```

