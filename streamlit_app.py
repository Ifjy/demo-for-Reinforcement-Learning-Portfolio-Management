# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt  # Ensure imported for plt.close()

# import asyncio # Not used in the provided snippet, can be removed if not needed elsewhere
from backend_logic import (
    load_all_assets,
    run_rl_simulation,
    run_ews_simulation,
    run_buy_and_hold_simulation,
    buyingWinner_agent_periodic,
    MeanVariance_agent_periodic,
    EWS_agent_periodic,
    run_benchmark_agent_simulation,
    calculate_financial_metrics,
    generate_portfolio_value_plot,
    generate_weights_pie_chart,  # Kept import in case of future use, but not directly used for RL agent now
    generate_weights_heatmap,
    DEFAULT_CONFIG_PATH,
)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="强化学习投资组合管理演示")

    st.title("🚀 风险敏感型强化学习投资组合管理演示")
    st.markdown(
        """
        这个演示展示了一个使用风险敏感型强化学习开发的投资组合管理策略。
        它将与等权重（EWS）、买入并持有（B&H）等基准策略在动态选择的股票子集上进行性能比较。
        **重要假设：** 配置文件中 `close_pos` 指定的输入数据特征被假定为股票的 **阶段收益率**。
        """
    )

    # --- Sidebar: Configuration Inputs ---
    st.sidebar.header("⚙️ 数据与模型配置")
    config_file = st.sidebar.text_input("配置文件路径 (.jsonc)", DEFAULT_CONFIG_PATH)
    model_file_override = st.sidebar.text_input(
        "已训练模型路径 (可选，覆盖配置)",
        help="如果为空，则使用配置文件中 'folder_path_for_model' 和 'experiment_name_for_model' 构建的路径",
    )
    num_stocks_ui = st.sidebar.number_input(
        "选择股票数量 (0 则使用配置)",
        min_value=0,
        value=0,  # Default to using config
        step=1,
        help="设为 0 则使用配置文件中的 'N_stock'。否则，此值将覆盖配置。",
    )

    # --- Cache and Session State Management ---
    if "assets_loaded" not in st.session_state:
        st.session_state.assets_loaded = False
    if "simulation_results" not in st.session_state:
        st.session_state.simulation_results = None
    if "portfolio_values_to_plot" not in st.session_state:
        st.session_state.portfolio_values_to_plot = None
    if "all_metrics_df" not in st.session_state:
        st.session_state.all_metrics_df = pd.DataFrame()

    cache_key_params = f"{config_file}-{model_file_override}"
    if num_stocks_ui > 0:
        cache_key_params += f"-N_UI_{num_stocks_ui}"

    if (
        "current_cache_key" not in st.session_state
        or st.session_state.current_cache_key != cache_key_params
    ):
        st.session_state.assets_loaded = False
        st.session_state.simulation_results = None
        st.session_state.portfolio_values_to_plot = None
        st.session_state.all_metrics_df = pd.DataFrame()
        st.session_state.current_cache_key = cache_key_params
        # Clear previous selections if cache is invalidated
        for key in [
            "last_selected_indices",
            "last_selected_names",
            "current_applied_seed",
        ]:
            if key in st.session_state:
                del st.session_state[key]

    # --- Load Assets ---
    if not st.session_state.assets_loaded:
        with st.spinner("⏳ 正在加载和处理资产..."):
            assets = load_all_assets(
                config_path=config_file,
                model_path_override=(
                    model_file_override if model_file_override else None
                ),
                num_stocks_to_select_override=(
                    num_stocks_ui if num_stocks_ui > 0 else None
                ),
            )
            if assets and all(a is not None for a in assets):
                (
                    st.session_state.agent,
                    st.session_state.test_env_instance,
                    st.session_state.test_data_numpy,
                    st.session_state.stock_names_list,
                    st.session_state.config_loaded_dict,
                ) = assets
                st.session_state.assets_loaded = True
                st.sidebar.success("✅ 资产加载成功!")
            else:
                st.sidebar.error("❌ 资产加载失败。请检查侧边栏输入或控制台错误。")
                st.stop()  # Stop execution if assets can't be loaded

    # Make variables available from session state
    agent = st.session_state.agent
    test_env_instance = st.session_state.test_env_instance
    stock_names_list = st.session_state.stock_names_list
    config_loaded_dict = st.session_state.config_loaded_dict
    test_data_numpy = st.session_state.test_data_numpy  # For display

    # --- Sidebar: Display Loaded Asset Information ---
    st.sidebar.subheader("ℹ️ 已加载资产信息")
    if "current_applied_seed" in st.session_state:
        st.sidebar.write(f"**当前应用种子:** `{st.session_state.current_applied_seed}`")
    st.sidebar.write(
        f"**选定股票 ({len(stock_names_list)}支):** `{', '.join(stock_names_list) if stock_names_list else '无'}`"
    )
    if "last_selected_names" in st.session_state:
        with st.sidebar.expander("显示选定的股票索引和名称 (调试用)"):
            st.write(
                "**选定索引:**", st.session_state.get("last_selected_indices", "N/A")
            )
            st.write(
                "**选定名称:**", st.session_state.get("last_selected_names", "N/A")
            )

    st.sidebar.write(f"**测试数据形状:** `{test_data_numpy.shape}`")
    st.sidebar.write(f"**初始财富:** `{config_loaded_dict.get('init_wealth', 'N/A')}`")
    st.sidebar.write(
        f"**收盘价特征索引:** `{config_loaded_dict.get('close_pos', 'N/A')}`"
    )

    risk_free_rate_for_metrics = config_loaded_dict.get(
        "risk_free_rate_annual_metrics", 0.0
    )
    st.sidebar.write(
        f"**指标计算用无风险利率 (来自配置):** `{risk_free_rate_for_metrics*100:.2f}%`"
    )

    # --- Sidebar: Benchmark Strategy Options ---
    st.sidebar.header("📊 基准策略选项")
    run_rl_agent = st.sidebar.checkbox("运行 RL Agent", value=True)
    run_ews_daily = st.sidebar.checkbox("运行 EWS (每日等权)", value=True)
    run_bnh = st.sidebar.checkbox("运行 买入并持有 (B&H)", value=True)

    st.sidebar.subheader("周期性调仓基准:")
    holding_period_periodic = st.sidebar.slider(
        "调仓周期 (天)",
        5,
        120,
        20,
        5,
        help="适用于买入赢家、周期性EWS和均值方差策略。",
    )

    max_top_n = len(stock_names_list) if stock_names_list else 1
    default_top_n = min(10, max_top_n) if max_top_n > 0 else 1
    top_n_winner_ui = st.sidebar.number_input(
        "Top N 股票 (买入赢家)",
        min_value=1,
        max_value=max_top_n if max_top_n > 0 else 1,
        value=default_top_n,
        step=1,
        help="为“买入赢家”策略选择表现最好的股票数量。",
        disabled=(max_top_n == 0 or not stock_names_list),
    )

    # Allow user to override risk-free rate for MV periodic and metrics calculation
    annual_rf_mv_periodic_input = st.sidebar.number_input(
        "年化无风险利率 (用于均值方差 & 指标)",
        min_value=0.0,
        max_value=0.2,
        value=risk_free_rate_for_metrics,
        step=0.005,
        format="%.4f",
        help="用于均值方差优化，也将更新用于下方所有财务指标计算的无风险利率。",
    )
    # Update the risk_free_rate_for_metrics based on user input for broader use
    risk_free_rate_for_metrics = annual_rf_mv_periodic_input

    run_buying_winner = st.sidebar.checkbox(
        f"运行 买入赢家 (每 {holding_period_periodic} 天)",
        value=False,
        disabled=(not stock_names_list),
    )
    run_mean_variance = st.sidebar.checkbox(
        f"运行 均值方差 (每 {holding_period_periodic} 天)",
        value=False,
        disabled=(not stock_names_list),
    )
    run_ews_periodic = st.sidebar.checkbox(
        f"运行 EWS (周期性, 每 {holding_period_periodic} 天)",
        value=False,
        disabled=(not stock_names_list),
    )

    # --- Main Content Area: Run Simulation and View Results ---
    st.header("🏁 运行模拟并查看结果")
    if st.button("▶️ 运行所有选定模拟", key="run_sim_button"):
        if not st.session_state.assets_loaded:
            st.error("资产未加载。请检查配置和日志。")
            st.stop()

        with st.spinner("🌪️ 模拟进行中... 请稍候。"):
            st.session_state.simulation_results = {}
            portfolio_values_to_plot = {}
            metrics_data_frames = []

            # Define which simulations to run based on checkboxes
            sim_runners_config = {
                "RL Agent": {
                    "runner": run_rl_simulation,
                    "params": {"agent": agent},
                    "run_flag": run_rl_agent,
                },
                "EWS (Daily)": {
                    "runner": run_ews_simulation,
                    "params": {},
                    "run_flag": run_ews_daily,
                },
                "Buy & Hold (B&H)": {  # Name made consistent for display
                    "runner": run_buy_and_hold_simulation,
                    "params": {},
                    "run_flag": run_bnh,
                },
            }

            # Add periodic benchmarks if selected
            if run_buying_winner:
                sim_runners_config[f"买入赢家 (周期 {holding_period_periodic}天)"] = {
                    "runner": run_benchmark_agent_simulation,
                    "params": {
                        "agent_function": buyingWinner_agent_periodic,
                        "agent_name": f"买入赢家 (周期 {holding_period_periodic}天)",
                        "holding_period": holding_period_periodic,
                        "N_stock_to_pick": top_n_winner_ui,
                    },
                    "run_flag": True,  # Already checked by run_buying_winner
                }
            if run_mean_variance:
                sim_runners_config[f"均值方差 (周期 {holding_period_periodic}天)"] = {
                    "runner": run_benchmark_agent_simulation,
                    "params": {
                        "agent_function": MeanVariance_agent_periodic,
                        "agent_name": f"均值方差 (周期 {holding_period_periodic}天)",
                        "holding_period": holding_period_periodic,
                        "risk_free_rate_annual": annual_rf_mv_periodic_input,  # Use the input value
                    },
                    "run_flag": True,
                }
            if run_ews_periodic:
                sim_runners_config[f"EWS (周期 {holding_period_periodic}天)"] = {
                    "runner": run_benchmark_agent_simulation,
                    "params": {
                        "agent_function": EWS_agent_periodic,
                        "agent_name": f"EWS (周期 {holding_period_periodic}天)",
                        "holding_period": holding_period_periodic,
                    },
                    "run_flag": True,
                }

            with st.expander("显示传递给模拟的核心配置参数 (调试用)"):
                st.json(
                    {
                        "N_stock": config_loaded_dict.get("N_stock"),
                        "stock_names": config_loaded_dict.get("stock_names"),
                        "init_wealth": config_loaded_dict.get("init_wealth"),
                        "window_size": config_loaded_dict.get("window_size"),
                        "close_pos": config_loaded_dict.get("close_pos"),
                        "seed_in_config": config_loaded_dict.get("seed"),
                        "applied_seed_for_run": st.session_state.get(
                            "current_applied_seed", "N/A"
                        ),
                        "risk_free_rate_for_metrics": risk_free_rate_for_metrics,  # Show current RF rate
                    }
                )

            for name, config in sim_runners_config.items():
                if config["run_flag"]:
                    st.markdown(f"--- \n**🚀 开始模拟: {name}**")
                    env_copy = copy.deepcopy(test_env_instance)
                    try:
                        # For run_benchmark_agent_simulation, agent_name is part of params
                        # For others, it's the key 'name'
                        sim_params = {**config["params"]}
                        if (
                            "agent_name" not in sim_params
                            and "agent_function" in sim_params
                        ):
                            sim_params["agent_name"] = (
                                name  # Pass name if it's a benchmark agent
                            )

                        values, weights_df, log_returns = config["runner"](
                            env=env_copy, config=config_loaded_dict, **sim_params
                        )
                        st.session_state.simulation_results[name] = {
                            "values": values,
                            "weights": weights_df,
                            "log_returns": log_returns,
                        }
                        portfolio_values_to_plot[name] = values
                        if log_returns is not None and not log_returns.empty:
                            metrics_data_frames.append(
                                calculate_financial_metrics(
                                    log_returns,
                                    name,
                                    risk_free_rate_annual=risk_free_rate_for_metrics,
                                )
                            )
                        st.success(f"✅ {name} 模拟完成。")
                        with st.expander(f"查看 {name} 的结果摘要 (调试用)"):
                            st.write(
                                "投资组合最终价值:",
                                values.iloc[-1] if not values.empty else "N/A",
                            )
                            st.write(
                                "权重 DataFrame (前5行):",
                                (
                                    weights_df.head()
                                    if not weights_df.empty
                                    else "无权重数据"
                                ),
                            )
                            st.write(
                                "对数收益率 (前5行):",
                                (
                                    log_returns.head().to_frame()
                                    if log_returns is not None and not log_returns.empty
                                    else "无收益率数据"
                                ),
                            )

                    except Exception as e:
                        st.error(f"❌ {name} 模拟过程中发生错误: {e}")
                        st.exception(e)  # Provides full traceback in the app

            st.session_state.portfolio_values_to_plot = portfolio_values_to_plot
            if metrics_data_frames:
                st.session_state.all_metrics_df = pd.concat(metrics_data_frames)
            else:
                st.session_state.all_metrics_df = pd.DataFrame()
            st.success("🎉 所有选定模拟均已完成！")

            # Debug for Mean Variance if it ran and stored debug info (from backend_logic potentially)
            # This assumes backend_logic.py might store such keys if MV has issues.
            # If not, this section won't show anything.
            mv_debug_keys = sorted(
                [key for key in st.session_state.keys() if key.startswith("mv_debug_t")]
            )
            if mv_debug_keys:
                with st.expander("⚙️ 均值方差策略调试信息 (若有)", collapsed=True):
                    key_to_display = None
                    for key in reversed(mv_debug_keys):
                        if isinstance(st.session_state[key], dict) and (
                            st.session_state[key].get("rebalancing_triggered")
                            or st.session_state[key].get("error")
                        ):
                            key_to_display = key
                            break
                    if not key_to_display and mv_debug_keys:
                        key_to_display = mv_debug_keys[-1]

                    if key_to_display and key_to_display in st.session_state:
                        st.write(
                            f"**时间步 {key_to_display.replace('mv_debug_t', '')} 的调试信息：**"
                        )
                        st.json(st.session_state[key_to_display])
                    else:
                        st.info("均值方差策略可能未进行调仓，或未记录特定调试信息。")

    # --- Results Display Area ---
    if st.session_state.portfolio_values_to_plot:
        st.subheader("📈 投资组合价值比较")
        fig_values = generate_portfolio_value_plot(
            st.session_state.portfolio_values_to_plot
        )
        st.plotly_chart(fig_values, use_container_width=True)

    if not st.session_state.all_metrics_df.empty:
        st.subheader("📊 关键性能指标")

        def style_metrics_df(df_to_style):  # Styling function remains the same
            styled = df_to_style.copy()
            percent_cols = [
                "Cumulative Return",
                "Annualized Return",
                "Annualized Volatility (Std)",
                "Annualized Downside Std",
                "Max Drawdown",
            ]
            float_cols = [
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Skewness",
                "Kurtosis",
            ]
            for col in percent_cols:
                if col in styled.columns:
                    styled[col] = styled[col].apply(
                        lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
                    )
            for col in float_cols:
                if col in styled.columns:
                    styled[col] = styled[col].apply(
                        lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
                    )
            return styled

        st.dataframe(
            style_metrics_df(st.session_state.all_metrics_df), use_container_width=True
        )
    elif st.session_state.get("run_sim_button"):  # if button was pressed but no results
        st.info("模拟已运行，但未生成指标数据 (可能所有策略均未选中或运行失败)。")
    else:
        st.info("运行模拟后，此处将显示投资组合价值图表和性能指标。")

    st.markdown("---")

    # --- Portfolio Weights Heatmap ---
    if st.session_state.simulation_results:
        st.subheader("🔥 投资组合权重时序热力图")
        strategies_with_weights = [
            name
            for name, results in st.session_state.simulation_results.items()
            if "weights" in results
            and isinstance(results["weights"], pd.DataFrame)
            and not results["weights"].empty
        ]
        if not strategies_with_weights:
            st.info("没有包含权重数据的模拟结果可用于显示热力图。")
        else:
            selected_strategy_for_heatmap = st.selectbox(
                "选择一个策略以查看其权重分配热力图:",
                options=sorted(strategies_with_weights),
                index=0,
            )
            if selected_strategy_for_heatmap:
                weights_df_to_plot = st.session_state.simulation_results[
                    selected_strategy_for_heatmap
                ]["weights"]
                if weights_df_to_plot.empty or weights_df_to_plot.shape[1] == 0:
                    st.warning(
                        f"选定策略 '{selected_strategy_for_heatmap}' 没有权重数据或没有资产可在热力图中显示。"
                    )
                else:
                    with st.spinner(
                        f"正在为 {selected_strategy_for_heatmap} 生成权重热力图..."
                    ):
                        try:
                            fig_heatmap = generate_weights_heatmap(
                                weights_df_to_plot, selected_strategy_for_heatmap
                            )
                            st.pyplot(fig_heatmap)
                            plt.close(
                                fig_heatmap
                            )  # Crucial: close figure to free memory
                        except Exception as e:
                            st.error(
                                f"为 {selected_strategy_for_heatmap} 生成热力图时发生错误: {e}"
                            )
                            st.exception(e)
    st.markdown("---")
    st.caption("强化学习投资组合管理演示 | 数据假设适用。")
