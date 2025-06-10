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
    with st.expander("🔬 核心方法简介 (点击展开)", expanded=True):
        st.markdown(
            """
            我们的投资组合管理策略基于一种经过改进的深度确定性策略梯度（DDPG）算法，其核心是**风险敏感性**。
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            st.info("🎯 **核心思想三句话**")
            st.markdown(
                """
                1.  **风险敏感RL**: 传统RL只追求最高收益，而我们的方法通过在奖励中加入**方差惩罚项**，使Agent在追求收益的同时，更倾向于选择**收益稳定、风险较低**的投资策略。
                2.  **ARE模块作用**: 该模块用于提取高维资产表达，为actor-critic网络提供更丰富的输入信息，帮助Agent更好地理解市场状态和资产表现。
                3.  **重要假设**: 本次回测**考虑交易手续费** (但假定完全以收盘价成交无滑点)，因此所有策略表现均为理想情况下的结果，这亦是未来工作的改进方向。
                """
            )

        with col2:
            st.warning("🧠 **算法关键更新规则**")
            st.markdown("我们的算法主要通过修改DDPG的目标价值函数来实现风险规避。")

            # 使用 st.latex 来展示公式，效果最好
            st.markdown("**1. 风险调整后的目标 (Target)**:")
            st.latex(
                r"""
            y_i = \underbrace{r_i - \beta(r_i - \eta)^2}_{\text{风险调整后奖励}} - \underbrace{J}_{\text{长期目标}} + \gamma Q_{\omega'}(s_{i+1}, \mu_{\theta'}(s_{i+1}))
            """
            )

            st.markdown(
                r"""
                - $r_i$ 是单步收益。
                - $\beta(r_i - \eta)^2$ 是核心的**风险惩罚项**，当单步收益 $r_i$ 偏离长期平均收益 $\eta$ 时，Agent会受到惩罚。
                - $J$ 是策略的长期平均表现，即 $\eta - \rho \eta_{\sigma}$ (均值-方差目标)。
                - $Q_{\omega'}(...)$ 是目标Q网络给出的未来价值估计，与标准DDPG一致。
                """
            )

        st.markdown("---")
        st.markdown("**2. 网络更新**:")
        st.markdown(
            """
            - **价值网络 (Critic) 更新**: 通过最小化损失函数 $L$ 来更新价值网络，使其能够准确估计风险调整后的长期回报。
            """
        )
        st.latex(
            r"""
        L(\omega) = \frac{1}{N} \sum_{i=1}^{N} (y_i - Q_{\omega}(s_i, a_i))^2
        """
        )
        st.markdown(
            """
            - **策略网络 (Actor) 更新**: 沿用策略梯度来更新策略网络，目标是生成能够最大化上述风险调整后Q值的动作（即投资权重）。
            """
        )
        st.latex(
            r"""
        \nabla_{\theta} J \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} \mu_{\theta}(s_i) \nabla_{a} Q_{\omega}(s_i, a) \big|_{a=\mu_{\theta}(s_i)}
        """
        )
    st.markdown(
        """
        这个演示展示了一个使用风险敏感型强化学习开发的投资组合管理策略。
        它将与等权重（EWS）、买入并持有（B&H）等基准策略在动态选择的股票子集上进行性能比较。
        **重要假设：** 配置文件中 `close_pos` 指定的输入数据特征被假定为股票的 **阶段收益率**。
        """
    )
    st.header("模型架构解析")  # 给这个板块一个总标题

    # 创建三个选项卡
    tab1, tab2, tab3, tab4 = st.tabs(
        ["▶️ 整体架构", "🧩 ARE模块", "🤖 Actor网络", "🧐 Critic网络"]
    )

    with tab1:
        st.image(
            "main_arc.png",
            use_container_width=True,
            caption="整体模型架构：数据经过ARE处理后，输入到DDPG的Actor-Critic网络中。",
        )
        st.info("点击上方不同的选项卡查看各模块的详细结构。")

    with tab2:
        st.image(
            "are.png",
            use_container_width=True,
            caption="ARE (Asset Representation Extractor) 模块：用于从历史数据中估计资产收益的动态范围，为风险敏感性提供依据。",
        )

    with tab3:
        # 假设你的Actor图名叫 actor_arc.png
        st.image(
            "actor_arc.png",
            use_container_width=True,
            caption="Actor 网络：负责根据当前市场状态，生成具体的投资组合权重。",
        )

    with tab4:
        # 假设你的Critic图名叫 critic_arc.png
        st.image(
            "critic_arc.png",
            use_container_width=True,
            caption="Critic 网络：负责评估 Actor 所做决策的好坏（Q值），并指导 Actor 的学习。",
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
                    st.session_state.all_original_stock_names,
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
    all_original_stock_names = st.session_state.all_original_stock_names

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
        st.header("🏆 挑战开始：构建你的专属投资组合！")  # <-- 新增的大标题
        st.markdown(
            """
            下面是本次回测期间，所有备选股票的独立走势和关键指标。
            请分析它们，并选择你认为组合起来能表现最好的股票。
            你的选择将构成一个“买入并持有”策略，与AI策略一决高下。
            """
        )

        available_stocks = st.session_state.get("stock_names_list", [])
        # 使用 train 部分 也就是历史数据 而不是 test 用来查看
        train_data_numpy = st.session_state.get("train_data_numpy")
        config_loaded_dict = st.session_state.get("config_loaded_dict", {})

        if not available_stocks or train_data_numpy is None:
            st.warning("股票资产尚未加载，无法显示选股信息。")
        else:
            with st.spinner("正在生成股票走势图和指标..."):
                try:
                    close_pos_index = config_loaded_dict.get("close_pos")
                    if close_pos_index is None:
                        st.error("配置中未找到 'close_pos'，无法计算收益率。")
                    else:
                        returns_df = pd.DataFrame(
                            train_data_numpy[:, :, close_pos_index],
                            columns=available_stocks,
                        )

                        # --- 1. 计算并展示累计收益图 (已优化排序) ---
                        st.subheader("备选股票累计收益走势")
                        cumulative_returns_df = (1 + returns_df).cumprod()

                        # **【图例排序优化】**
                        # 1. 获取每个股票的最终累计收益
                        final_values = cumulative_returns_df.iloc[-1]
                        # 2. 根据最终收益降序排列，得到新的股票顺序
                        sorted_stocks_by_value = final_values.sort_values(
                            ascending=False
                        ).index
                        # 3. 按照新的顺序重新排列DataFrame的列
                        df_for_plot = cumulative_returns_df[sorted_stocks_by_value]

                        fig_trends = px.line(
                            # 使用排序后的DataFrame进行绘图
                            df_for_plot,
                            title="股票累计收益（回测期内）",
                            labels={
                                "index": "时间步",
                                "value": "累计乘积收益",
                                "variable": "股票",
                            },
                        )
                        st.plotly_chart(fig_trends, use_container_width=True)

                        # --- 2. 计算并展示关键指标表格 (已优化排序) ---
                        st.subheader("关键性能指标")
                        metrics = []
                        annualization_factor = 252

                        # **【表格排序优化】**
                        # 在循环中，将指标存为原始的 float 数字，而不是格式化的字符串
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
                                    "总回报率": total_return,  # 存储为原始数字
                                    "年化回报率": annualized_return,  # 存储为原始数字
                                    "年化波动率": annualized_volatility,  # 存储为原始数字
                                    "夏普比率": sharpe_ratio,  # 存储为原始数字
                                }
                            )

                        metrics_df = pd.DataFrame(metrics)

                        # 使用 Styler 对象对数字进行格式化，以供显示，同时保留其数值用于排序
                        st.dataframe(
                            metrics_df.style.format(
                                {
                                    "总回报率": "{:.2%}",
                                    "年化回报率": "{:.2%}",
                                    "年化波动率": "{:.2%}",
                                    "夏普比率": "{:.2f}",
                                }
                            ),
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(f"生成选股信息时出错: {e}")

            st.markdown("---")

            user_selection = st.multiselect(
                label="分析完毕后，请在这里选择你的股票 (建议选择5-10支):",
                options=available_stocks,
                key="user_selected_stocks",
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

                    except Exception as e:
                        st.error(f"❌ {name} 模拟过程中发生错误: {e}")
                        st.exception(e)  # Provides full traceback in the app

            st.session_state.portfolio_values_to_plot = portfolio_values_to_plot
            if metrics_data_frames:
                st.session_state.all_metrics_df = pd.concat(metrics_data_frames)
            else:
                st.session_state.all_metrics_df = pd.DataFrame()
            st.success("🎉 所有选定模拟均已完成！")

            mv_debug_keys = sorted(
                [key for key in st.session_state.keys() if key.startswith("mv_debug_t")]
            )

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
            应用样式到指标 DataFrame：
            1. 统一居中对齐。
            2. 根据优劣高亮单元格（绿色）。
            3. 格式化数字显示，同时保留其原始值用于排序。
            """
            try:
                # 复制DataFrame以避免修改原始数据
                df_copy = df_to_style.copy()

                # 定义哪些列是越大越好，哪些是越小越好
                # 注意：这里我们把'Max Drawdown'也归为越小越好，因为它的值是负数，绝对值越小越好。
                # 'Annualized Turnover Rate' 通常也是越小越好。
                max_is_better_cols = [
                    "Cumulative Return",
                    "Annualized Return",
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Calmar Ratio",
                ]
                min_is_better_cols = [
                    "Annualized Volatility (Std)",
                    "Annualized Downside Std",
                    "Max Drawdown",
                    "Annualized Turnover Rate",
                    "Max Drawdown Period",
                ]

                # 创建 Styler 对象
                styler = df_copy.style

                # 1. 先应用所有列的数字格式化
                # --------------------------------
                format_dict = {}
                percent_cols = [
                    "Cumulative Return",
                    "Annualized Return",
                    "Annualized Volatility (Std)",
                    "Annualized Downside Std",
                    "Max Drawdown",
                    "Annualized Turnover Rate",
                ]
                float_cols = [
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Calmar Ratio",
                    "Skewness",
                    "Kurtosis",
                ]
                int_cols = ["Max Drawdown Period"]

                for col in percent_cols:
                    if col in df_copy.columns:
                        format_dict[col] = "{:.2%}"
                for col in float_cols:
                    if col in df_copy.columns:
                        format_dict[col] = "{:.2f}"
                for col in int_cols:
                    if col in df_copy.columns:
                        format_dict[col] = lambda x: (
                            f"{int(x)}" if pd.notnull(x) and np.isfinite(x) else "N/A"
                        )

                styler.format(format_dict, na_rep="N/A")

                # 2. 定义一个通用的高亮和居中函数
                # --------------------------------
                def highlight_and_center(series, mode):
                    """
                    根据 series 的最大值或最小值高亮，并对所有单元格居中。
                    mode: 'max' 或 'min'
                    """
                    is_target = (
                        (series == series.max())
                        if mode == "max"
                        else (series == series.min())
                    )

                    # 返回一个包含CSS样式的列表，长度与Series相同
                    # 无论是否高亮，都应用 'text-align: center'
                    return [
                        (
                            "background-color: #d4edda; color: #155724; font-weight: bold; text-align: center"
                            if v
                            else "text-align: center"
                        )
                        for v in is_target
                    ]

                # 3. 将高亮函数应用到指定的列上
                # --------------------------------
                style_dict = {}
                for col in max_is_better_cols:
                    if col in df_copy.columns:
                        style_dict[col] = lambda s, mode="max": highlight_and_center(
                            s, mode
                        )

                for col in min_is_better_cols:
                    if col in df_copy.columns:
                        style_dict[col] = lambda s, mode="min": highlight_and_center(
                            s, mode
                        )

                # Styler.apply() 接受一个返回样式列表的函数
                # 我们用一个字典来为不同列指定不同的函数
                styler.apply(
                    lambda s: style_dict.get(
                        s.name, lambda x: ["text-align: center"] * len(s)
                    )(s),
                    axis=0,
                )

                return styler

            except Exception as e:
                st.error(f"应用样式时发生错误: {e}")
                # 如果出错，返回一个基础的、只居中的Styler对象
                return df_to_style.copy().style.set_properties(
                    **{"text-align": "center"}
                )

        st.dataframe(
            style_metrics_df(st.session_state.all_metrics_df), use_container_width=False
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
