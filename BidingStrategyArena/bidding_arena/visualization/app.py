import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import uuid
from typing import List, Dict, Any

from bidding_arena.core.constants import StrategyType
from bidding_arena.core.interfaces import StrategyMetadata
from bidding_arena.core.strategy import DynamicStrategy
from bidding_arena.core.engine import ReplayEngine
from bidding_arena.generation.generator import StrategyGenerator
from bidding_arena.generation.llm_client import MockLLMClient, OllamaLLMClient
from bidding_arena.generation.validator import CodeValidator
from bidding_arena.data.generator import MockDataGenerator

DEFAULT_CUSTOM_STRATEGY = """def bidding_strategy(
    initial_budget: float,
    total_duration: int,
    remaining_budget: float,
    remaining_time: int,
    winner_price_percentiles: dict,
    conversion_rate: float
) -> float:
    import math
    # Example: Bid fixed price
    return 1.0
"""

def main():
    st.set_page_config(page_title="Bidding Strategy Arena", layout="wide")
    st.title("🤖 LLM Bidding Strategy Arena")

    # Sidebar: Configuration
    st.sidebar.header("Configuration")
    
    use_ollama = st.sidebar.checkbox("Use Real Ollama (Gemma3)", value=True)
    model_name = st.sidebar.text_input("Model Name", value="gemma3:12b") if use_ollama else "mock"
    
    selected_types = st.sidebar.multiselect(
        "Select Strategy Types to Generate",
        [t.value for t in StrategyType],
        default=[t.value for t in StrategyType]
    )

    initial_budget = st.sidebar.number_input("Initial Budget ($)", value=100000.0)
    # MODIFIED: Increase slider range to support larger simulations
    num_records = st.sidebar.slider("Base Simulation Records", 1000, 100000, 10000)

    # Custom Strategy Input (Sidebar)
    with st.sidebar.expander("📝 Custom Strategy (Optional)", expanded=False):
        custom_strategy_code = st.text_area(
            "Paste Python Code", 
            value=DEFAULT_CUSTOM_STRATEGY, 
            height=300,
            help="Define a 'bidding_strategy' function. Edit this to enable."
        )

    # Initialize Session State
    if 'market_data' not in st.session_state:
        st.session_state['market_data'] = None
    if 'results' not in st.session_state:
        st.session_state['results'] = []
    if 'optimization_round' not in st.session_state:
        st.session_state['optimization_round'] = 0
    if 'analyses' not in st.session_state:
        st.session_state['analyses'] = {} # round -> analysis_text

    # Main Action Button
    if st.sidebar.button("🚀 Generate & Compete"):
        # Reset state
        st.session_state['results'] = []
        st.session_state['optimization_round'] = 0
        st.session_state['analyses'] = {}
        
        # Run Initial Arena
        # Pass custom_strategy_code even if it's default, process_custom_strategy handles validation if we want strictness,
        # but here we only run if it's NOT default to avoid noise.
        # Actually, let's allow running the default one if the user wants to test the "fixed price" baseline.
        # But previous logic was: custom_strategy_code if custom_strategy_code != DEFAULT_CUSTOM_STRATEGY else None
        
        # Let's check if the code inside is functionally different or if the user intends to run it.
        # For simplicity, we stick to the previous logic: only run if it's different from the placeholder.
        # OR we can just run it if the expander is open? No way to detect that easily.
        # Let's rely on the text content.
        
        code_to_run = custom_strategy_code if custom_strategy_code != DEFAULT_CUSTOM_STRATEGY else None
        
        run_initial_arena(
            use_ollama, 
            model_name, 
            selected_types, 
            initial_budget, 
            num_records,
            code_to_run
        )

    # Render Persistent Dashboard if results exist
    if st.session_state['results']:
        render_dashboard(use_ollama, model_name, initial_budget)

def get_generator(use_ollama, model_name):
    if use_ollama:
        try:
            llm_client = OllamaLLMClient(model=model_name)
        except Exception as e:
            st.error(f"Failed to initialize Ollama: {e}")
            return None, None
    else:
        llm_client = MockLLMClient()
    return StrategyGenerator(llm_client), llm_client

def generate_round_analysis(results, round_num, llm_client):
    """Helper to generate and store analysis for a round."""
    analysis_data = [{
        "name": res['strategy_name'],
        "win_rate": f"{res['win_rate']:.2%}",
        "cpa": f"${res['avg_cpa']:.2f}"
    } for res in results]
    
    try:
        analysis_text = llm_client.analyze_strategies(analysis_data)
        st.session_state['analyses'][round_num] = analysis_text
        return True
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return False

def process_custom_strategy(code: str, budget: float, market_data: Any, round_num: int) -> Dict[str, Any]:
    """Helper to process and run a custom strategy."""
    if not CodeValidator.validate(code):
        st.error("Custom strategy validation failed. Ensure it has 'bidding_strategy' function and uses only allowed imports (math).")
        return None

    try:
        metadata = StrategyMetadata(
            id=str(uuid.uuid4()),
            name=f"User_Custom_Strategy_R{round_num}",
            strategy_type="Custom",
            code=code,
            created_at=time.time()
        )
        strategy = DynamicStrategy(metadata)
        engine = ReplayEngine(initial_budget=budget)
        sim_result = engine.run(strategy, market_data)
        
        sim_result['strategy_name'] = f"User Custom (Round {round_num})"
        sim_result['metadata'] = metadata
        sim_result['round'] = round_num
        return sim_result
    except Exception as e:
        st.error(f"Error running custom strategy: {e}")
        return None

def run_initial_arena(use_ollama, model_name, selected_types, initial_budget, num_records, custom_code=None):
    generator, llm_client = get_generator(use_ollama, model_name)
    if not generator: return

    # 1. Data Generation
    with st.spinner("Generating Market Data..."):
        data_gen = MockDataGenerator()
        # MODIFIED: Pass 'initial_budget' to ensure data generation scales with budget
        market_data = data_gen.generate_data(num_records, total_budget=initial_budget)
        st.session_state['market_data'] = market_data
        st.success(f"Generated {len(market_data)} bid requests.")
        
        # Show data sample
        with st.expander("View Market Data Sample"):
            st.dataframe(market_data.head())
        
    # 2. Strategy Generation & Execution (Initial Round)
    results = []
    
    # Run Custom Strategy if provided
    if custom_code:
        st.info("Running Custom Strategy...")
        custom_result = process_custom_strategy(custom_code, initial_budget, market_data, 0)
        if custom_result:
            results.append(custom_result)

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, s_type_str in enumerate(selected_types):
        s_type = StrategyType(s_type_str)
        status_text.text(f"Generating strategy for {s_type.value}...")
        
        try:
            # Generate Code
            metadata = generator.generate(s_type)
            strategy = DynamicStrategy(metadata)
            
            # Run Simulation
            engine = ReplayEngine(initial_budget=initial_budget)
            sim_result = engine.run(strategy, market_data)
            
            sim_result['strategy_name'] = s_type.value
            sim_result['metadata'] = metadata
            sim_result['round'] = 0
            results.append(sim_result)
            
        except Exception as e:
            st.error(f"Error processing {s_type.value}: {e}")
        
        progress_bar.progress((i + 1) / len(selected_types))

    st.session_state['results'] = results
    status_text.empty()

    # NOTE: We do NOT generate analysis automatically here anymore.
    # We just let it fall through to render_dashboard, where the user can manually trigger it.

def render_dashboard(use_ollama, model_name, initial_budget):
    all_results = st.session_state['results']
    max_round = st.session_state['optimization_round']
    
    # Pre-calculate client to avoid multiple inits
    generator, llm_client = get_generator(use_ollama, model_name)
    
    # Iterate through all rounds to display history sequentially
    for r in range(max_round + 1):
        round_results = [res for res in all_results if res.get('round', 0) == r]
        
        if not round_results:
            continue
            
        st.divider()
        st.markdown(f"## 🏁 Round {r} Results")
        
        # 1. Display Strategy Code for this round
        with st.expander(f"View Round {r} Strategy Codes", expanded=False):
            cols = st.columns(len(round_results))
            for idx, res in enumerate(round_results):
                with cols[idx % len(cols)]: # Wrap around if too many
                    st.markdown(f"**{res['strategy_name']}**")
                    st.code(res['metadata'].code, language="python")

        # 2. Display Round Metrics
        display_round_metrics(round_results)
        
        # 3. AI Analysis Report (Manual Trigger)
        if r in st.session_state['analyses']:
            with st.expander(f"🧠 AI Analysis Report (Round {r})", expanded=True):
                st.markdown(st.session_state['analyses'][r])
        else:
            # Manual Trigger Button
            if st.button(f"📝 Generate Analysis Report (Round {r})", key=f"analyze_btn_{r}"):
                 with st.spinner("Analyzing..."):
                     if llm_client:
                         if generate_round_analysis(round_results, r, llm_client):
                             st.rerun()

    # --- Global Summary Section ---
    st.divider()
    st.header("🏆 Global Leaderboard & Trends")
    
    # 4. Global Table
    display_round_metrics(all_results)
    
    # 5. Global Charts (Cumulative)
    display_global_charts(all_results)
    
    # 6. Optimization Control (Always at bottom)
    st.divider()
    st.subheader("🧬 AI Optimization Lab")
    
    next_round = max_round + 1
    # Find results from the latest round to optimize
    latest_results = [res for res in all_results if res.get('round', 0) == max_round]
    
    # Input for Custom Strategy in Next Round
    with st.expander(f"📝 Inject Custom Strategy for Round {next_round} (Optional)", expanded=False):
        next_round_custom_code = st.text_area(
            "Paste Python Code", 
            value=DEFAULT_CUSTOM_STRATEGY, 
            height=300,
            key=f"custom_code_round_{next_round}",
            help="Define a 'bidding_strategy' function."
        )
    
    if st.button(f"✨ Auto-Optimize All Round {max_round} Strategies (Start Round {next_round})"):
        # Check if user modified the default template
        custom_code_to_pass = next_round_custom_code if next_round_custom_code != DEFAULT_CUSTOM_STRATEGY else None
        
        perform_optimization(
            use_ollama, 
            model_name, 
            initial_budget, 
            latest_results, 
            next_round,
            custom_code_to_pass
        )
        st.rerun()

def perform_optimization(use_ollama, model_name, initial_budget, source_results, round_num, custom_code=None):
    generator, llm_client = get_generator(use_ollama, model_name)
    market_data = st.session_state['market_data']
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    new_results = []
    
    # 1. Run Custom Strategy if provided
    if custom_code:
        st.info("Running Injected Custom Strategy...")
        custom_result = process_custom_strategy(custom_code, initial_budget, market_data, round_num)
        if custom_result:
            new_results.append(custom_result)
    
    # 2. Optimize Existing Strategies
    for i, res in enumerate(source_results):
        strategy_name = res['strategy_name']
        status_text.text(f"Optimizing {strategy_name}...")
        
        # Prepare metrics
        metrics_summary = {
            "win_rate": f"{res['win_rate']:.2%}",
            "avg_cpm": f"${res['avg_cpm']:.2f}",
            "total_spend": f"${res['total_spend']:.2f}"
        }
        
        # Build history context AND find best previous version
        import re
        base_name = re.sub(r" \(Round \d+\)$", "", strategy_name)
        
        # Find all previous results for this strategy lineage
        all_res = st.session_state['results']
        # Filter for strategies that start with the base name AND have a round number < current round
        lineage = [r for r in all_res if r['strategy_name'].startswith(base_name) and r.get('round', 0) < round_num]
        lineage.sort(key=lambda x: x.get('round', 0))
        
        history_context = f"History of '{base_name}' evolution:\n"
        
        best_candidate = res # Default to current
        best_metric = res['conversion_count']
        
        for l in lineage:
            r_num = l.get('round', 0)
            # Use Conversions as the primary metric for 'Best' selection
            metric_val = l['conversion_count']
            if metric_val > best_metric:
                best_metric = metric_val
                best_candidate = l
                
            # Take a generous snippet or even full code if token limit allows. 
            code_content = l['metadata'].code
            if len(code_content) > 500:
                code_content = code_content[:500] + "...(truncated)"
            
            history_context += f"--- Round {r_num} ---\n"
            history_context += f"Metrics: Win Rate={l['win_rate']:.2%}, CPA=${l['avg_cpa']:.2f}, Conversions={l['conversion_count']}\n"
            history_context += f"Code:\n```python\n{code_content}\n```\n\n"

        # Prepare metrics of the BEST candidate to analyze
        metrics_summary = {
            "win_rate": f"{best_candidate['win_rate']:.2%}",
            "avg_cpm": f"${best_candidate['avg_cpm']:.2f}",
            "total_spend": f"${best_candidate['total_spend']:.2f}",
            "conversions": best_candidate['conversion_count']
        }
        
        # Add a note to history context telling LLM we are reverting if needed
        # Note: 'res' is the strategy from the PREVIOUS round (the one we selected to optimize).
        # If best_candidate is NOT res, it means an earlier round was better.
        if best_candidate['strategy_name'] != res['strategy_name']:
            history_context += f"\nNOTE: The latest version (Round {res.get('round',0)}) performed worse than Round {best_candidate.get('round',0)}. We are automatically REVERTING to Round {best_candidate.get('round',0)} as the base for this new optimization to recover performance.\n"

        # --- NEW: Re-roll Mechanism ---
        # Threshold: If conversions are extremely low (e.g. < 10), consider it a failed lineage.
        if best_candidate['conversion_count'] < 10:
            history_context += "\nCRITICAL ALERT: This strategy lineage has consistently failed to generate meaningful conversions. DISCARD the previous logic. Generate a COMPLETELY NEW and AGGRESSIVE strategy from scratch to break this deadlock.\n"
        # ------------------------------

        try:
            # Analyze & Optimize using the BEST candidate
            _, new_meta = generator.analyze_and_optimize(best_candidate['metadata'], metrics_summary, history_context)
            
            # Run Simulation
            opt_strategy = DynamicStrategy(new_meta)
            engine = ReplayEngine(initial_budget=initial_budget)
            opt_res = engine.run(opt_strategy, market_data)
            
            # Update Name
            # Clean up previous round info from name if exists
            base_name = strategy_name.split(" (Round")[0]
            opt_res['strategy_name'] = f"{base_name} (Round {round_num})"
            opt_res['metadata'] = new_meta
            opt_res['round'] = round_num
            
            new_results.append(opt_res)
            
        except Exception as e:
            st.error(f"Optimization failed for {strategy_name}: {e}")
            
        progress_bar.progress((i + 1) / len(source_results))
    
    # Append new results to session state
    st.session_state['results'].extend(new_results)
    st.session_state['optimization_round'] = round_num
    
    # Don't generate analysis automatically. Let render_dashboard handle it with manual trigger.
    st.rerun()

def display_round_metrics(results):
    metrics_df = pd.DataFrame([{
        "Strategy": r['strategy_name'],
        "Round": r.get('round', 0),
        "Wins": r['win_count'],
        "Win Rate": f"{r['win_rate']:.2%}",
        "Spend": f"${r['total_spend']:.2f}",
        "Conversions": r['conversion_count'],
        "CPA": f"${r['avg_cpa']:.2f}"
    } for r in results])
    
    st.table(metrics_df.sort_values(by="Conversions", ascending=False))

def display_global_charts(results):
    col1, col2 = st.columns(2)
    
    all_history = []
    for r in results:
        hist_df = pd.DataFrame(r['history'])
        hist_df['Strategy'] = r['strategy_name']
        all_history.append(hist_df)
    
    if all_history:
        full_hist = pd.concat(all_history)
        
        with col1:
            st.markdown("### 📉 Remaining Budget vs Time")
            fig_budget = px.line(full_hist, x="timestamp", y="remaining_budget", color="Strategy", markers=True)
            st.plotly_chart(fig_budget, use_container_width=True)
            
        with col2:
            st.markdown("### 🏷️ Moving Average Bid Price ($)")
            fig_bid = px.line(full_hist, x="timestamp", y="avg_bid_price", color="Strategy", markers=True)
            fig_bid.update_layout(yaxis_title="Avg Bid Price ($)")
            st.plotly_chart(fig_bid, use_container_width=True)
            
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 🎯 Cumulative Conversions")
            fig_conv = px.line(full_hist, x="timestamp", y="conversion_count", color="Strategy", markers=True)
            st.plotly_chart(fig_conv, use_container_width=True)

        with col4:
            st.markdown("### 📊 Win Rate & Conversion Efficiency")
            # Calculate metrics for chart
            chart_data = []
            for r in results:
                chart_data.append({
                    "Strategy": r['strategy_name'],
                    "Win Rate": r['win_rate'],
                    "Conversions": r['conversion_count']
                })
            df_chart = pd.DataFrame(chart_data)
            
            # Create figure with secondary y-axis
            from plotly.subplots import make_subplots
            fig_bar = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add Win Rate bar (Left Axis)
            fig_bar.add_trace(
                go.Bar(name='Win Rate', x=df_chart['Strategy'], y=df_chart['Win Rate'], offsetgroup=1),
                secondary_y=False,
            )
            
            # Add Conversions bar (Right Axis)
            fig_bar.add_trace(
                go.Bar(name='Conversions', x=df_chart['Strategy'], y=df_chart['Conversions'], offsetgroup=2),
                secondary_y=True,
            )
            
            # Layout adjustments
            fig_bar.update_layout(
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_bar.update_yaxes(title_text="Win Rate", secondary_y=False, tickformat=".0%")
            fig_bar.update_yaxes(title_text="Conversions", secondary_y=True)
            
            st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
