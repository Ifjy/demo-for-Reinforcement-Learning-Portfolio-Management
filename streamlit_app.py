# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt  # Ensure imported for plt.close()
from PIL import Image
import plotly.express as px

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
    run_user_strategy_simulation,
    DEFAULT_CONFIG_PATH,
)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="强化学习投资组合管理演示")
    model_arc = Image.open("model_arc.png")
    col1, col2, col3 = st.columns([1, 2, 1])  # 比例 1:2:1（中间占 50%）
    with col2:
        st.image(model_arc, use_container_width=True, caption="模型架构示意图")
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
                    st.session_state.train_data_numpy,
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
    train_data_numpy = st.session_state.train_data_numpy  # For potential future use

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
        "risk_free_rate_annual_metrics", 0.02
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
    # --- Interactive User Strategy Section ---
    with st.expander("👨‍💼 挑战者模式：创建你自己的投资组合", expanded=True):
        st.markdown(
            """
        下面是本次回测期间，所有备选股票的独立走势和关键指标。
        请分析它们，并选择你认为组合起来能表现最好的股票。
        """
        )

        available_stocks = st.session_state.get("stock_names_list", [])
        test_data_numpy = st.session_state.get("test_data_numpy")
        config_loaded_dict = st.session_state.get("config_loaded_dict", {})

        if not available_stocks or test_data_numpy is None:
            st.warning("股票资产尚未加载，无法显示选股信息。")
        else:
            # --- 新增图表和指标的代码 ---
            with st.spinner("正在生成股票走势图和指标..."):
                try:
                    # 从配置中获取收益率所在的特征索引
                    close_pos_index = config_loaded_dict.get("close_pos")
                    if close_pos_index is None:
                        st.error("配置中未找到 'close_pos'，无法计算收益率。")
                    else:
                        # 提取所有股票在回测期内的阶段收益率
                        returns_df = pd.DataFrame(
                            train_data_numpy[:, :, close_pos_index],
                            columns=available_stocks,
                        )

                        # 1. 计算并展示累计收益图
                        st.subheader("备选股票累计收益走势")
                        cumulative_returns_df = (1 + returns_df).cumprod()

                        # 使用 Plotly 绘制交互式图表
                        fig_trends = px.line(
                            cumulative_returns_df,
                            title="股票累计收益（回测期内）",
                            labels={
                                "index": "时间步",
                                "value": "累计乘积收益",
                                "variable": "股票",
                            },
                        )
                        st.plotly_chart(fig_trends, use_container_width=True)

                        # 2. 计算并展示关键指标表格
                        st.subheader("关键性能指标")
                        metrics = []
                        # 假设一年有252个交易日
                        annualization_factor = 252

                        for stock in available_stocks:
                            stock_returns = returns_df[stock]
                            total_return = cumulative_returns_df[stock].iloc[-1] - 1
                            annualized_return = (1 + total_return) ** (
                                annualization_factor / len(stock_returns)
                            ) - 1
                            annualized_volatility = stock_returns.std() * np.sqrt(
                                annualization_factor
                            )
                            sharpe_ratio = (
                                annualized_return / annualized_volatility
                                if annualized_volatility != 0
                                else 0
                            )

                            metrics.append(
                                {
                                    "股票": stock,
                                    "总回报率": f"{total_return:.2%}",
                                    "年化回报率": f"{annualized_return:.2%}",
                                    "年化波动率": f"{annualized_volatility:.2%}",
                                    "夏普比率": f"{sharpe_ratio:.2f}",
                                }
                            )

                        metrics_df = pd.DataFrame(metrics)
                        st.dataframe(metrics_df, use_container_width=True)

                except Exception as e:
                    st.error(f"生成选股信息时出错: {e}")
        # 使用 st.multiselect 让用户选择
        user_selection = st.multiselect(
            label="请选择你的股票 (建议选择5-10支):",
            options=available_stocks,
            key="user_selected_stocks",  # 将选择结果存储在 session_state 中
        )

        if user_selection:
            st.success(
                f"你已经选择了 {len(user_selection)} 支股票。点击下方的“运行模拟”按钮开始挑战！"
            )
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
            # --- 新增代码：检查并添加用户策略 ---
            if (
                "user_selected_stocks" in st.session_state
                and st.session_state.user_selected_stocks
            ):
                user_stocks = st.session_state.user_selected_stocks
                sim_runners_config[f"用户精选 ({len(user_stocks)}支)"] = {
                    "runner": run_user_strategy_simulation,
                    # 在这里添加一个新的参数 "all_in_env_stock_names"
                    "params": {
                        "selected_stocks": user_stocks,
                        "all_in_env_stock_names": stock_names_list,  # stock_names_list 在主程序中是可用的
                    },
                    "run_flag": True,
                }
            # --- 新增代码结束 ---
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

                        values, weights_df, log_returns, turnover_rates = config[
                            "runner"
                        ](env=env_copy, config=config_loaded_dict, **sim_params)
                        st.session_state.simulation_results[name] = {
                            "values": values,
                            "weights": weights_df,
                            "log_returns": log_returns,
                            "turnover_rates": turnover_rates,
                        }
                        portfolio_values_to_plot[name] = values
                        if log_returns is not None and not log_returns.empty:
                            metrics_data_frames.append(
                                calculate_financial_metrics(
                                    log_returns,
                                    turnover_rate_list=turnover_rates,
                                    policy_name=name,
                                    risk_free_rate_annual=risk_free_rate_for_metrics,
                                )
                            )
                        st.success(f"✅ {name} 模拟完成。")
                        # with st.expander(f"查看 {name} 的结果摘要 (调试用)"):
                        #     st.write(
                        #         "投资组合最终价值:",
                        #         values.iloc[-1] if not values.empty else "N/A",
                        #     )
                        #     st.write(
                        #         "权重 DataFrame (前5行):",
                        #         (
                        #             weights_df.head()
                        #             if not weights_df.empty
                        #             else "无权重数据"
                        #         ),
                        #     )
                        #     st.write(
                        #         "对数收益率 (前5行):",
                        #         (
                        #             log_returns.head().to_frame()
                        #             if log_returns is not None and not log_returns.empty
                        #             else "无收益率数据"
                        #         ),
                        #     )

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
            # if mv_debug_keys:
            #     with st.expander("⚙️ 均值方差策略调试信息 (若有)", collapsed=True):
            #         key_to_display = None
            #         for key in reversed(mv_debug_keys):
            #             if isinstance(st.session_state[key], dict) and (
            #                 st.session_state[key].get("rebalancing_triggered")
            #                 or st.session_state[key].get("error")
            #             ):
            #                 key_to_display = key
            #                 break
            #         if not key_to_display and mv_debug_keys:
            #             key_to_display = mv_debug_keys[-1]

            #         if key_to_display and key_to_display in st.session_state:
            #             st.write(
            #                 f"**时间步 {key_to_display.replace('mv_debug_t', '')} 的调试信息：**"
            #             )
            #             st.json(st.session_state[key_to_display])
            #         else:
            #             st.info("均值方差策略可能未进行调仓，或未记录特定调试信息。")

    # --- Results Display Area ---
    if st.session_state.portfolio_values_to_plot:
        st.subheader("📈 投资组合价值比较")
        fig_values = generate_portfolio_value_plot(
            st.session_state.portfolio_values_to_plot
        )
        st.plotly_chart(fig_values, use_container_width=True)

    if not st.session_state.all_metrics_df.empty:
        st.subheader("📊 关键性能指标")

        def style_metrics_df(df_to_style: pd.DataFrame):
            """
            应用样式到指标 DataFrame：居中内容，格式化数字以正确显示，同时保留数值排序。
            确保 'Annualized Turnover Rate' 和 'Max Drawdown Period' 也被正确格式化。
            """
            try:
                # 创建一个 Styler 对象。直接对原始 DataFrame 进行操作，不先转换为字符串。
                styler = df_to_style.copy().style

                # 1. 内容居中
                styler.set_properties(**{"text-align": "center"})

                # 2. 定义各列的显示格式
                #    这样处理后，Streamlit 在排序时仍会使用原始的数值数据。
                format_dict = {}

                percent_cols = [
                    "Cumulative Return",
                    "Annualized Return",
                    "Annualized Volatility (Std)",
                    "Annualized Downside Std",
                    "Max Drawdown",
                    "Annualized Turnover Rate",  # 新增的换手率也应为百分比
                ]
                float_cols = [
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Calmar Ratio",
                    "Skewness",
                    "Kurtosis",
                ]
                # Max Drawdown Period 通常是整数（天数）
                int_cols = ["Max Drawdown Period"]

                for col in percent_cols:
                    if col in df_to_style.columns:
                        # Pandas Styler 的 format 支持 Python 格式规范字符串
                        format_dict[col] = "{:.2%}"

                for col in float_cols:
                    if col in df_to_style.columns:
                        format_dict[col] = "{:.2f}"

                for col in int_cols:
                    if col in df_to_style.columns:
                        # 确保整数列正确显示，并处理可能的 NaN 或 Inf
                        format_dict[col] = lambda x: (
                            f"{int(x)}"
                            if pd.notnull(x)
                            and isinstance(x, (int, float))
                            and not np.isinf(x)
                            and not np.isnan(x)
                            else "N/A"
                        )

                # 应用格式化，并为 NaN 值指定显示内容
                styler.format(format_dict, na_rep="N/A")

                return styler  # 返回 Styler 对象

            except Exception as e:
                st.error(f"应用样式时发生错误: {e}")
                # 如果出错，返回原始 DataFrame 的 Styler 对象，不做任何修改
                return df_to_style.copy().style

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
