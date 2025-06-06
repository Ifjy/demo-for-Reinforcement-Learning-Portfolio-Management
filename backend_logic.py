# backend_logic.py
import streamlit as st
import torch
import json5
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

# from matplotlib import font_manager # Not explicitly used
from pypfopt import expected_returns, risk_models, efficient_frontier
from env import Porfolio_Env  # Assumed to exist
from agent import DDPG_multitask  # Assumed to exist
import net  # Assumed to exist
from data_preprocess import data_process2  # Assumed to exist
import random as py_random

# import logging # Removed
# import sys # Removed as it was only for logging

# --- Default Paths and Configuration ---
DEFAULT_CONFIG_PATH = "config.jsonc"
DEFAULT_MODEL_PATH_TEMPLATE = "ddpg_multitask_experiment_model.pth"
DEFAULT_FTR_PATH = "000300r.ftr"


def load_config_file(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json5.load(f)
        # logger.info(f"Config file {config_path} loaded successfully.") # Removed
        return config
    except FileNotFoundError:
        st.error(f"错误：配置文件 {config_path} 未找到。")
        # logger.error(f"Config file {config_path} not found.") # Removed
        return None
    except Exception as e:
        st.error(f"错误：加载配置文件 {config_path} 失败: {e}")
        # logger.error(f"Error loading config file {config_path}: {e}", exc_info=True) # Removed
        return None


@st.cache_resource
def load_all_assets(
    config_path=DEFAULT_CONFIG_PATH,
    model_path_override=None,
    num_stocks_to_select_override=None,
):
    config_main = load_config_file(config_path)
    if config_main is None:
        return None, None, None, None, None

    device = torch.device(config_main.get("device", "cpu"))
    config_main["device"] = str(device)

    seed_value = config_main.get("seed")
    if seed_value is None:
        st.error("错误：配置中未找到 'seed'！将使用默认种子 42。")
        seed_value = 42
    elif not isinstance(seed_value, int):
        st.warning(f"警告：配置中的 'seed' (值: {seed_value}) 不是整数。尝试转换...")
        try:
            seed_value = int(seed_value)
            st.info(f"种子成功转换为整数: {seed_value}")
        except ValueError:
            st.error(f"错误：无法将种子 '{seed_value}' 转换为整数。将使用默认种子 42。")
            seed_value = 42

    # logger.info(f"Using seed value: {seed_value}") # Removed
    py_random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    st.session_state["current_applied_seed"] = seed_value

    raw_data_ftr_path = config_main.get("data_path_ftr", DEFAULT_FTR_PATH)
    try:
        # logger.info(f"Loading data from: {raw_data_ftr_path}") # Removed
        raw_dataframe = pd.read_feather(raw_data_ftr_path)
        adata_full, all_original_stock_names = data_process2(raw_dataframe)
        # logger.info(
        #     f"Data processed. Full data shape: {adata_full.shape}, Total original stocks: {len(all_original_stock_names)}"
        # ) # Removed
    except Exception as e:
        st.error(f"错误：加载或处理数据时出错: {e}")
        # logger.error(f"Error loading or processing data: {e}", exc_info=True) # Removed
        return None, None, None, None, None

    if (
        not isinstance(adata_full, np.ndarray)
        or not isinstance(all_original_stock_names, list)
        or (
            adata_full.ndim == 3
            and adata_full.shape[1] != len(all_original_stock_names)
        )
    ):
        st.error("错误：数据处理 (data_process2) 返回的数据不一致或格式不正确。")
        # logger.error(
        #     f"Data processing returned inconsistent data. Shape: {getattr(adata_full, 'shape', 'N/A')}, Names len: {len(all_original_stock_names) if isinstance(all_original_stock_names, list) else 'N/A'}"
        # ) # Removed
        return None, None, None, None, None

    N_stock = (
        num_stocks_to_select_override
        if num_stocks_to_select_override is not None
        and num_stocks_to_select_override > 0
        else config_main.get("N_stock")
    )
    if N_stock is None or not isinstance(N_stock, int) or N_stock <= 0:
        st.error(
            f"错误：'N_stock' ({N_stock}) 无效或未在配置中正确设置。请在UI或配置文件中指定一个正整数。"
        )
        # logger.error(f"Invalid N_stock value: {N_stock}") # Removed
        return None, None, None, None, None

    if N_stock > len(all_original_stock_names):
        st.warning(
            f"警告：请求的 N_stock ({N_stock}) 大于可用股票数 ({len(all_original_stock_names)})。将使用所有可用股票。"
        )
        N_stock = len(all_original_stock_names)
    config_main["N_stock"] = N_stock

    if not all_original_stock_names:
        st.error("错误：没有可供选择的股票 (all_original_stock_names 为空)。")
        # logger.error("No stocks available for selection.") # Removed
        return None, None, None, None, None

    try:
        if N_stock > len(all_original_stock_names):
            N_stock = len(all_original_stock_names)

        selected_indices = py_random.sample(
            range(len(all_original_stock_names)), N_stock
        )
        selected_stock_names = [all_original_stock_names[i] for i in selected_indices]

        # logger.info(
        #     f"Selected {N_stock} stocks. Names: {', '.join(selected_stock_names) if selected_stock_names else 'None'}"
        # ) # Removed
        config_main["stock_names"] = selected_stock_names
        st.session_state["last_selected_indices"] = selected_indices
        st.session_state["last_selected_names"] = selected_stock_names
    except ValueError as e:
        st.error(
            f"错误：从股票列表中采样时出错: {e}. "
            f"可能是 N_stock ({N_stock}) 与列表长度 ({len(all_original_stock_names)})不匹配。"
        )
        # logger.error(
        #     f"Error sampling from all_original_stock_names: {e}", exc_info=True
        # ) # Removed
        return None, None, None, None, None

    total_time_steps = adata_full.shape[0]
    train_start_prop = config_main.get("train_start_prop", 0.0)
    train_end_prop = config_main.get("train_end_prop", 0.7)
    test_end_prop = config_main.get("test_end_prop", 1.0)

    train_start_abs_idx = int(total_time_steps * train_start_prop)
    train_end_abs_idx = int(total_time_steps * train_end_prop)
    test_end_abs_idx = int(total_time_steps * test_end_prop)

    if not (
        0
        <= train_start_abs_idx
        < train_end_abs_idx
        <= test_end_abs_idx
        <= total_time_steps
    ):
        st.error(
            f"错误：无效的数据分割索引。请检查 train/test_end_prop。 T={total_time_steps}, TrainStart={train_start_abs_idx}, TrainEnd={train_end_abs_idx}, TestEnd={test_end_abs_idx}"
        )
        # logger.error(
        #     f"Invalid data split indices. T={total_time_steps}, TrainStart={train_start_abs_idx}, TrainEnd={train_end_abs_idx}, TestEnd={test_end_abs_idx}"
        # ) # Removed
        return None, None, None, None, None

    test_data_np = adata_full[train_end_abs_idx:test_end_abs_idx, selected_indices, :]
    if test_data_np.shape[0] == 0:
        st.error("错误：测试数据集为空。请检查数据分割比例和总数据长度。")
        # logger.error("Test data slice is empty.") # Removed
        return None, None, None, None, None
    # logger.info(f"Test data shape: {test_data_np.shape}") # Removed

    in_channels = N_stock
    if N_stock == 0:
        st.error("错误: N_stock (选定股票数量) 为0，无法初始化模型。")
        # logger.error("N_stock is 0, cannot initialize model.") # Removed
        return None, None, None, None, None

    in_features = test_data_np.shape[2]
    num_actions = in_channels + 1

    try:
        actor = net.PolicyNet2(
            in_channels,
            in_features,
            config_main["embed_dim"],
            num_actions,
            config_main["hidden_size"],
            config_main.get("portfolio_size", 1),
        ).to(device)
        critic = net.Critic2(
            in_channels,
            in_features,
            config_main["embed_dim"],
            num_actions,
            config_main["hidden_size"],
        ).to(device)
        lsre = None
        if (
            not config_main.get("use_simple_agent", 0)
            and "use_batch_lsre" in config_main
        ):
            lsre_class = (
                net.BatchLSRE if config_main["use_batch_lsre"] == 1 else net.LSRE
            )
            lsre = lsre_class(
                config_main["window_size"],
                in_features,
                config_main["embed_dim"],
                num_actions,
            ).to(device)
        agent = DDPG_multitask(actor, critic, lsre, None, config_main)
    except KeyError as e:
        st.error(f"错误：模型初始化时缺少配置参数: {e}。请检查config.jsonc文件。")
        # logger.error(
        #     f"Missing config parameter for model initialization: {e}", exc_info=True
        # ) # Removed
        return None, None, None, None, None
    except Exception as e:
        st.error(f"错误：模型初始化失败: {e}")
        # logger.error(f"Model initialization failed: {e}", exc_info=True) # Removed
        return None, None, None, None, None

    model_load_path = model_path_override
    if not model_load_path:
        folder_path = config_main.get("folder_path_for_model", "")
        exp_name = config_main.get("experiment_name_for_model", "")
        if not folder_path or not exp_name:
            st.warning(
                "警告：配置文件中缺少 'folder_path_for_model' 或 'experiment_name_for_model'。将尝试加载默认模型名称。"
            )
            model_load_path = DEFAULT_MODEL_PATH_TEMPLATE
        else:
            model_load_path = DEFAULT_MODEL_PATH_TEMPLATE

    # logger.info(f"Attempting to load model weights from: {model_load_path}") # Removed
    try:
        checkpoint = torch.load(model_load_path, map_location=device)
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        if (
            lsre
            and "lsre_state_dict" in checkpoint
            and hasattr(agent, "lsre")
            and agent.lsre is not None
        ):
            agent.lsre.load_state_dict(checkpoint["lsre_state_dict"])
        agent.actor.eval()
        agent.critic.eval()
        if lsre and hasattr(agent, "lsre") and agent.lsre is not None:
            agent.lsre.eval()
        # logger.info("Model weights loaded and models set to eval mode.") # Removed
    except FileNotFoundError:
        st.error(
            f"错误：模型文件 {model_load_path} 未找到。请确保路径正确或模型已训练。"
        )
        # logger.error(f"Model file {model_load_path} not found.") # Removed
        return None, None, None, None, None
    except Exception as e:
        st.error(f"错误：加载模型权重失败 (路径: {model_load_path}): {e}")
        # logger.error(
        #     f"Error loading model weights from {model_load_path}: {e}", exc_info=True
        # ) # Removed
        return None, None, None, None, None

    try:
        test_env = Porfolio_Env(data=test_data_np, config=config_main)
        # logger.info("Portfolio_Env initialized for testing.") # Removed
    except Exception as e:
        st.error(f"错误：初始化 Portfolio_Env 失败: {e}")
        # logger.error(f"Error initializing Portfolio_Env: {e}", exc_info=True) # Removed
        return None, None, None, None, None

    return agent, test_env, test_data_np, selected_stock_names, config_main


# --- Helper Function: Extract Results from Environment ---
def _extract_simulation_results(env: Porfolio_Env, config: dict, strategy_name: str):
    portfolio_values = pd.Series(env.asset_memory, name=strategy_name)
    # if portfolio_values.empty: # Removed logger.warning
    # pass

    N_stock = config.get("N_stock", 0)
    num_expected_cols_actions = N_stock + 1
    stock_names_from_config = config.get("stock_names", [])

    if len(stock_names_from_config) != N_stock:
        # logger.warning( # Removed
        #     f"[{strategy_name}] Mismatch: len(stock_names)={len(stock_names_from_config)} vs N_stock={N_stock}. "
        #     "Using generic stock names for weights DataFrame columns."
        # )
        columns_for_df_weights = ["Cash"] + [f"Stock_{i+1}" for i in range(N_stock)]
    else:
        columns_for_df_weights = ["Cash"] + stock_names_from_config

    if len(columns_for_df_weights) != num_expected_cols_actions:
        # logger.error( # Removed
        #     f"[{strategy_name}] Internal error: Constructed column names length ({len(columns_for_df_weights)}) "
        #     f"does not match expected action columns ({num_expected_cols_actions}). Falling back to generic names."
        # )
        columns_for_df_weights = [
            f"WeightCol_{i}" for i in range(num_expected_cols_actions)
        ]

    actions_for_df = np.empty((0, num_expected_cols_actions))
    if not env.action_memory:
        # logger.warning(f"[{strategy_name}]: Action memory is empty.") # Removed
        pass
    else:
        try:
            actions_for_df = np.vstack(env.action_memory)
            if actions_for_df.ndim != 2:
                raise ValueError(f"vstack result not 2D, shape {actions_for_df.shape}")
            if actions_for_df.shape[1] != num_expected_cols_actions:
                # logger.warning( # Removed
                #     f"[{strategy_name}]: Action memory columns ({actions_for_df.shape[1]}) "
                #     f"don't match expected ({num_expected_cols_actions}). This might occur if actions are malformed."
                # )
                pass
        except ValueError:
            # logger.warning(f"[{strategy_name}]: np.vstack failed for action_memory. Trying np.array.") # Removed
            try:
                temp_actions = np.array(env.action_memory, dtype=object)
                if all(
                    isinstance(arr, np.ndarray) and arr.ndim == 1
                    for arr in temp_actions
                ) and all(
                    len(arr) == len(temp_actions[0])
                    for arr in temp_actions
                    if len(temp_actions) > 0
                ):
                    actions_for_df = np.array([list(arr) for arr in temp_actions])
                elif (
                    temp_actions.ndim == 1
                    and len(env.action_memory) == 1
                    and isinstance(temp_actions[0], (list, np.ndarray))
                ):
                    actions_for_df = np.array(temp_actions[0]).reshape(1, -1)
                elif temp_actions.ndim == 2:
                    actions_for_df = temp_actions
                else:
                    # logger.error(f"[{strategy_name}]: Cannot form 2D action array from action_memory. Shape: {temp_actions.shape if hasattr(temp_actions, 'shape') else 'N/A'}") # Removed
                    actions_for_df = np.empty((0, num_expected_cols_actions))
            except Exception:  # Simplified error logging to avoid logger
                # logger.error(f"[{strategy_name}]: Error converting action_memory to array: {e_arr}", exc_info=False) # Removed
                actions_for_df = np.empty((0, num_expected_cols_actions))

    final_columns_weights = columns_for_df_weights
    if actions_for_df.shape[0] > 0 and actions_for_df.shape[1] != len(
        columns_for_df_weights
    ):
        # logger.warning( # Removed
        #     f"[{strategy_name}]: Actual action columns in data ({actions_for_df.shape[1]}) "
        #     f"!= expected names ({len(columns_for_df_weights)}). Using generic names."
        # )
        final_columns_weights = [f"Weight_{i}" for i in range(actions_for_df.shape[1])]
    elif actions_for_df.shape[0] == 0:
        if actions_for_df.shape[1] != len(columns_for_df_weights):
            final_columns_weights = (
                [f"Weight_{i}" for i in range(actions_for_df.shape[1])]
                if actions_for_df.shape[1] > 0
                else ["NoActionData"]
            )

    portfolio_weights = pd.DataFrame(actions_for_df, columns=final_columns_weights)
    # if portfolio_weights.empty and actions_for_df.shape[0] > 0 : # Removed logger.error
    # pass

    daily_log_returns_list = []
    if env.reward_memory:
        if len(env.reward_memory) > 1 and env.reward_memory[0] == 0:
            daily_log_returns_list = env.reward_memory[1:]
        elif len(env.reward_memory) >= 1:
            daily_log_returns_list = env.reward_memory

    daily_log_returns = pd.Series(
        daily_log_returns_list, name=f"{strategy_name} Log Returns"
    )
    if daily_log_returns.empty and len(env.asset_memory) > 1:
        # logger.warning(f"[{strategy_name}]: Log returns from reward_memory is empty. Recalculating from asset values.") # Removed
        daily_log_returns = get_log_returns_from_portfolio_values(portfolio_values)
        daily_log_returns.name = f"{strategy_name} Log Returns (Recalculated)"

    if hasattr(env, "turnover_record"):
        temp_turnover_series = pd.Series(
            env.turnover_record, name=f"{strategy_name} Turnover Rates"
        )
        turnover_rates = pd.to_numeric(temp_turnover_series, errors="coerce")
        # if turnover_rates.isnull().all() and not temp_turnover_series.empty: # Removed logger.warning
        # pass
    else:
        # logger.warning(f"[{strategy_name}] env.turnover_record attribute not found. Turnover rates will be empty.") # Removed
        turnover_rates = pd.Series(dtype=float, name=f"{strategy_name} Turnover Rates")

    # if turnover_rates.empty and hasattr(env, 'action_memory') and len(env.action_memory) > 1 : # Removed logger.warning
    # pass

    return portfolio_values, portfolio_weights, daily_log_returns, turnover_rates


def get_log_returns_from_portfolio_values(portfolio_values_series):
    if not isinstance(portfolio_values_series, pd.Series):
        portfolio_values_series = pd.Series(portfolio_values_series)
    if portfolio_values_series.empty or len(portfolio_values_series) < 2:
        return pd.Series(dtype=float)
    portfolio_values_series = portfolio_values_series.clip(lower=1e-9)
    log_returns = np.log(
        portfolio_values_series / portfolio_values_series.shift(1)
    ).dropna()
    return log_returns


# --- RL Agent Simulation ---
def run_rl_simulation(agent, env: Porfolio_Env, config: dict):
    # logger.info("Starting RL Agent simulation.") # Removed
    state, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        with torch.no_grad():
            action_tensor = agent.take_action(state, "eva")
        state, _, done, truncated, _ = env.step(action_tensor.detach().cpu().numpy())
    # logger.info("RL Agent simulation finished.") # Removed
    return _extract_simulation_results(env, config, "RL Agent")


# --- EWS (Daily Equal Weight) Simulation ---
def run_ews_simulation(env: Porfolio_Env, config: dict):
    # logger.info("Starting EWS (Daily) simulation.") # Removed
    env.reset()
    done = False
    truncated = False
    stock_num = config.get("N_stock", 0)

    equal_action = np.zeros(stock_num + 1)
    if stock_num > 0:
        equal_action[1:] = 1.0 / stock_num
    else:
        equal_action[0] = 1.0

    while not done and not truncated:
        _, _, done, truncated, _ = env.step(equal_action)
    # logger.info("EWS (Daily) simulation finished.") # Removed
    return _extract_simulation_results(env, config, "EWS (Daily)")


# --- Buy and Hold (B&H) Simulation ---
def run_buy_and_hold_simulation(env: Porfolio_Env, config: dict):
    # logger.info("Starting Buy & Hold (B&H) simulation.") # Removed
    state, _ = env.reset()
    done = False
    truncated = False

    N_stock = config.get("N_stock", 0)
    portfolio_size_ratio = config.get("portfolio_size", 1.0)
    initial_bnh_action = np.zeros(N_stock + 1)

    if N_stock > 0:
        num_stocks_to_buy_initially = int(N_stock * portfolio_size_ratio)
        if portfolio_size_ratio > 0 and num_stocks_to_buy_initially == 0:
            num_stocks_to_buy_initially = 1
        num_stocks_to_buy_initially = min(num_stocks_to_buy_initially, N_stock)

        if num_stocks_to_buy_initially > 0:
            available_stock_indices_in_env = list(range(N_stock))
            selected_indices_for_bnh_portfolio = py_random.sample(
                available_stock_indices_in_env, num_stocks_to_buy_initially
            )
            weight_per_selected_stock = 1.0 / num_stocks_to_buy_initially
            for stock_idx_in_env in selected_indices_for_bnh_portfolio:
                initial_bnh_action[stock_idx_in_env + 1] = weight_per_selected_stock
            initial_bnh_action[0] = 0.0
        else:
            initial_bnh_action[0] = 1.0
    else:
        initial_bnh_action[0] = 1.0

    first_step = True
    while not done and not truncated:
        action_to_take = initial_bnh_action if first_step else state["weight"].flatten()
        state, _, done, truncated, _ = env.step(action_to_take)
        first_step = False
    # logger.info("Buy & Hold (B&H) simulation finished.") # Removed
    return _extract_simulation_results(env, config, "Buy & Hold (B&H)")


# --- Periodic Rebalancing Strategy Agents ---
def buyingWinner_agent_periodic(
    state, current_env_time, holding_period=20, N_stock_to_pick=10
):
    window_size = state["history"].shape[0]
    num_available_stocks = state["history"].shape[1]

    if (current_env_time - window_size) % holding_period == 0:
        if num_available_stocks == 0:
            return np.array([1.0])

        mean_performance = np.mean(state["history"][:, :, 0], axis=0)
        num_select = min(N_stock_to_pick, num_available_stocks)
        if num_select == 0:
            action = np.zeros(num_available_stocks + 1)
            action[0] = 1.0
            return action

        top_indices = np.argsort(mean_performance)[-num_select:]
        action = np.zeros(num_available_stocks + 1)
        action[top_indices + 1] = 1.0 / num_select
    else:
        action = state["weight"].flatten()
    return action


def MeanVariance_agent_periodic(
    state, current_env_time, holding_period=20, risk_free_rate_annual=0.02
):
    window_size = state["history"].shape[0]
    num_available_stocks = state["history"].shape[1]

    if (current_env_time - window_size) % holding_period == 0:
        if num_available_stocks == 0:
            return np.array([1.0])

        returns_df = pd.DataFrame(state["history"][:, :, 0])

        if returns_df.shape[0] < 2:
            return state["weight"].flatten()

        try:
            mu = expected_returns.mean_historical_return(returns_df, True)
            S = risk_models.sample_cov(returns_df, True)
            ef = efficient_frontier.EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.max_sharpe(risk_free_rate=risk_free_rate_annual)
            cleaned_weights = ef.clean_weights(rounding=4)

            action = np.zeros(num_available_stocks + 1)
            current_sum_weights = 0

            for stock_idx, weight in cleaned_weights.items():
                if 0 <= stock_idx < num_available_stocks:
                    action[stock_idx + 1] = weight
                    current_sum_weights += weight

            action[0] = max(0, 1.0 - current_sum_weights)

            if current_sum_weights > 1.0 + 1e-6:
                action[1:] = action[1:] / current_sum_weights
                action[0] = 0.0
        except Exception as e:
            # Instead of logger.warning, we might consider if st.warning is appropriate here,
            # but typically backend logic shouldn't directly call st UI elements if it can be avoided.
            # For now, just letting the error propagate or returning a default action.
            # print(f"MeanVariance Opt Error at t={current_env_time}: {e}. Holding weights.") # Optional: print to console
            action = state["weight"].flatten()
    else:
        action = state["weight"].flatten()
    return action


def EWS_agent_periodic(state, current_env_time, holding_period=20):
    window_size = state["history"].shape[0]
    num_available_stocks = state["history"].shape[1]

    if (current_env_time - window_size) % holding_period == 0:
        action = np.zeros(num_available_stocks + 1)
        if num_available_stocks > 0:
            action[1:] = 1.0 / num_available_stocks
        else:
            action[0] = 1.0
    else:
        action = state["weight"].flatten()
    return action


# --- Generic Benchmark Runner ---
def run_benchmark_agent_simulation(
    agent_function,
    env: Porfolio_Env,
    config: dict,
    agent_name: str,
    **agent_specific_params,
):
    # logger.info(f"Starting Benchmark Agent simulation: {agent_name}") # Removed
    state, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        current_env_time = env.t
        action_np = agent_function(state, current_env_time, **agent_specific_params)
        state, _, done, truncated, _ = env.step(action_np)
    # logger.info(f"Benchmark Agent simulation finished: {agent_name}") # Removed
    return _extract_simulation_results(env, config, agent_name)


def run_user_strategy_simulation(env, config, selected_stocks):
    """
    为用户选择的股票子集运行一个每日等权重策略的模拟。
    这本质上是一个经过筛选的 EWS 模拟。

    Args:
        env: The portfolio management environment instance.
        config: The configuration dictionary.
        selected_stocks (list): A list of stock names selected by the user.

    Returns:
        A tuple containing:
        - pd.Series: Portfolio values over time.
        - pd.DataFrame: Portfolio weights over time.
        - pd.Series: Portfolio log returns over time.
        - list: Portfolio turnover rates over time.
    """
    # 获取环境中所有股票的名称
    all_stock_names = env.stock_names

    # 创建一个权重向量，只为选定的股票分配权重
    num_selected = len(selected_stocks)
    if num_selected == 0:
        # 如果没有选择股票，则返回空结果
        empty_series = pd.Series(dtype=np.float64)
        empty_df = pd.DataFrame()
        return empty_series, empty_df, empty_series, []

    # 为每个选定的股票计算等权重
    equal_weight = 1.0 / num_selected

    # 构建一个完整的权重向量，未被选中的股票权重为0
    # 注意：这里的权重顺序必须和环境中的股票顺序一致
    action = np.zeros(len(all_stock_names))
    for i, stock_name in enumerate(all_stock_names):
        if stock_name in selected_stocks:
            action[i] = equal_weight

    # 因为这个策略的权重是固定的，所以我们可以在循环外定义好 action
    # 注意，action 的第一个元素是现金权重，但在这里我们的简单策略是不保留现金
    # 假设 action[0] 是现金，但我们的环境设计可能不同，这里我们直接使用股票权重
    # 根据你的环境设计，可能需要调整。假设环境期望一个只包含股票权重的 action。
    # 如果环境期望第一个是现金，那么应该是 action_with_cash = np.insert(action, 0, 0.0)

    obs, info = env.reset()
    done = False

    # 存储结果
    portfolio_values = [env.init_wealth]
    portfolio_log_returns = []
    weights_history = []
    turnover_rates = []

    while not done:
        # 在每个时间步都使用相同的、预先计算好的权重向量
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 记录数据
        portfolio_values.append(info["portfolio_value"])
        portfolio_log_returns.append(info.get("portfolio_log_return", np.nan))
        weights_history.append(info["portfolio_weights"])
        turnover_rates.append(info.get("turnover_rate", 0.0))

    # 转换结果为 Pandas DataFrame/Series
    dates = env.get_date_index()
    portfolio_values_s = pd.Series(portfolio_values, index=dates, name="User's Pick")
    log_returns_s = pd.Series(
        portfolio_log_returns, index=dates[1:], name="User's Pick"
    )

    # 创建权重 DataFrame
    # 权重历史记录的是每个时间步结束后的权重，所以长度应该和日期对应
    weights_df = pd.DataFrame(weights_history, index=dates[1:], columns=all_stock_names)

    return portfolio_values_s, weights_df, log_returns_s, turnover_rates


# --- Plotting (Matplotlib for Heatmap) ---
def generate_weights_heatmap(weights_df: pd.DataFrame, strategy_name: str):
    plt.rcParams["axes.unicode_minus"] = False

    if weights_df.empty or weights_df.shape[1] == 0:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(
            0.5,
            0.5,
            "No weight data to display.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"Weight Heatmap: {strategy_name} (No Data)")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    data_to_plot = weights_df.T
    min_assets_for_large_yticks = 20
    min_fig_height = 4
    max_fig_height = 15
    asset_label_fontsize = 8

    if data_to_plot.shape[0] > min_assets_for_large_yticks:
        fig_height = max(
            min_fig_height, min(max_fig_height, data_to_plot.shape[0] * 0.25)
        )
        if data_to_plot.shape[0] > 50:
            asset_label_fontsize = 6
    else:
        fig_height = max(
            min_fig_height, min(max_fig_height, data_to_plot.shape[0] * 0.4)
        )

    fig, ax = plt.subplots(figsize=(12, fig_height))
    heatmap = ax.imshow(
        data_to_plot, aspect="auto", cmap="YlGnBu", origin="lower", vmin=0.0, vmax=1.0
    )

    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Portfolio Weight", rotation=270, labelpad=15, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Time Step Index", fontsize=12)
    ax.set_ylabel("Assets", fontsize=12)
    ax.set_title(
        f"Portfolio Weight Allocation Heatmap: {strategy_name}", fontsize=14, pad=20
    )

    ax.set_yticks(np.arange(data_to_plot.shape[0]))
    ax.set_yticklabels(data_to_plot.index, fontsize=asset_label_fontsize)

    num_timesteps = data_to_plot.shape[1]
    step_size = max(1, num_timesteps // 10)
    xtick_positions = np.arange(0, num_timesteps, step_size)
    ax.set_xticks(xtick_positions)

    if isinstance(weights_df.index, pd.RangeIndex):
        xtick_labels = [str(i) for i in weights_df.index[::step_size]]
    else:
        xtick_labels = [str(label) for label in weights_df.index[::step_size]]
    ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=8)

    try:
        fig.tight_layout(pad=1.0)
    except Exception:
        pass
    return fig


# --- Financial Metrics Calculations ---


def annual_Std(return_list: list):
    return_list_np = np.array(return_list)
    if len(return_list_np) == 0:
        return 0.0
    return np.std(return_list_np) * np.sqrt(250)


def annual_Return(return_list: list):
    return_list_np = np.array(return_list)
    if len(return_list_np) == 0:
        return 0.0
    return np.mean(return_list_np) * 250


def cum_Return(return_list: list):
    return_list_np = np.array(return_list)
    if len(return_list_np) == 0:
        return 0.0
    return np.exp(np.sum(return_list_np)) - 1


def sharpe_Ratio(return_list: list, risk_free_rate_annual=0.0):
    if len(return_list) < 2:
        return np.nan
    ann_ret = annual_Return(return_list)
    ann_std = annual_Std(return_list)
    if ann_std == 0 or np.isclose(ann_std, 0):
        return (
            np.nan
            if np.isclose(ann_ret - risk_free_rate_annual, 0)
            else np.inf * np.sign(ann_ret - risk_free_rate_annual)
        )
    return (ann_ret - risk_free_rate_annual) / (ann_std + 1e-9)


def max_Drawdown(return_list: list):
    if not isinstance(return_list, np.ndarray):
        return_list_np = np.array(return_list)
    else:
        return_list_np = return_list.copy()

    if len(return_list_np) == 0:
        return 0.0, 0

    cumulative_log_returns = np.cumsum(return_list_np)
    portfolio_values = np.exp(np.insert(cumulative_log_returns, 0, 0.0))
    df_values = pd.Series(portfolio_values)
    roll_max = df_values.expanding().max()
    drawdown_series = df_values / roll_max
    drawdown_percent = drawdown_series - 1.0
    min_drawdown_value_idx = drawdown_percent.idxmin()
    max_drawdown_value = drawdown_percent[min_drawdown_value_idx]

    if min_drawdown_value_idx == 0:
        peak_start_index = 0
    else:
        peaks_before_trough = drawdown_series.iloc[:min_drawdown_value_idx][
            np.isclose(drawdown_series.iloc[:min_drawdown_value_idx], 1.0)
        ]
        if not peaks_before_trough.empty:
            peak_start_index = peaks_before_trough.last_valid_index()
        else:
            peak_start_index = 0

    if peak_start_index is None:
        max_drawdown_period = min_drawdown_value_idx
    else:
        max_drawdown_period = min_drawdown_value_idx - peak_start_index

    return max_drawdown_value, int(max_drawdown_period)


def calmar_Ratio(return_list: list, risk_free_rate_annual=0.0):
    if len(return_list) < 2:
        return np.nan
    ann_ret = annual_Return(return_list)
    max_dd_val, _ = max_Drawdown(return_list)
    if abs(max_dd_val) < 1e-9:
        return (
            np.nan
            if np.isclose(ann_ret - risk_free_rate_annual, 0)
            else np.inf * np.sign(ann_ret - risk_free_rate_annual)
        )
    return (ann_ret - risk_free_rate_annual) / (abs(max_dd_val) + 1e-9)


def annual_DownsideStd(return_list: list, required_return_annual=0.0):
    if len(return_list) == 0:
        return 0.0
    daily_mar = required_return_annual / 250.0
    return_list_np = np.array(return_list)
    downside_diffs = daily_mar - return_list_np
    downside_returns_squares = np.square(downside_diffs[downside_diffs > 0])
    if len(downside_returns_squares) == 0:
        return 0.0
    mean_sq_downside = np.mean(downside_returns_squares)
    daily_downside_std = np.sqrt(mean_sq_downside)
    return daily_downside_std * np.sqrt(250)


def sortino_Ratio(
    return_list: list, risk_free_rate_annual=0.0, required_return_annual=0.0
):
    if len(return_list) < 2:
        return np.nan
    ann_ret = annual_Return(return_list)
    ann_down_std = annual_DownsideStd(
        return_list, required_return_annual=required_return_annual
    )
    if ann_down_std == 0 or np.isclose(ann_down_std, 0):
        return (
            np.nan
            if np.isclose(ann_ret - risk_free_rate_annual, 0)
            else np.inf * np.sign(ann_ret - risk_free_rate_annual)
        )
    return (ann_ret - risk_free_rate_annual) / (ann_down_std + 1e-9)


def skewness(return_list: list):
    return_list_np = np.array(return_list)
    if len(return_list_np) < 3:
        return np.nan
    return scipy.stats.skew(return_list_np)


def kurtosis_val(return_list: list):
    return_list_np = np.array(return_list)
    if len(return_list_np) < 4:
        return np.nan
    return scipy.stats.kurtosis(return_list_np, fisher=True)


def annual_Turnover_Rate(turnover_data):
    if isinstance(turnover_data, pd.Series):
        numeric_array = pd.to_numeric(turnover_data, errors="coerce").to_numpy()
    elif isinstance(turnover_data, np.ndarray):
        if not np.issubdtype(turnover_data.dtype, np.number):
            numeric_array = pd.to_numeric(
                pd.Series(turnover_data), errors="coerce"
            ).to_numpy()
        else:
            numeric_array = turnover_data
    else:
        try:
            numeric_array = pd.to_numeric(
                pd.Series(turnover_data), errors="coerce"
            ).to_numpy()
        except Exception:
            numeric_array = np.array([])

    numeric_array = numeric_array.flatten()

    if len(numeric_array) == 0:
        return np.nan

    valid_turnovers = numeric_array[~np.isnan(numeric_array)]

    if len(valid_turnovers) == 0:
        return np.nan

    return np.mean(valid_turnovers) * 250


def calculate_financial_metrics(
    log_return_list, turnover_rate_list, policy_name=None, risk_free_rate_annual=0.0
):
    log_return_list_np = np.array(log_return_list).flatten()

    if len(log_return_list_np) == 0:
        metrics_dict = {
            "Policy": policy_name,
            "Cumulative Return": np.nan,
            "Annualized Return": np.nan,
            "Annualized Volatility (Std)": np.nan,
            "Annualized Downside Std": np.nan,
            "Max Drawdown": np.nan,
            "Max Drawdown Period": np.nan,
            "Sharpe Ratio": np.nan,
            "Sortino Ratio": np.nan,
            "Calmar Ratio": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
            "Annualized Turnover Rate": annual_Turnover_Rate(turnover_rate_list),
        }
        return pd.DataFrame([metrics_dict]).set_index("Policy")

    metrics = {"Policy": policy_name}
    metrics["Cumulative Return"] = cum_Return(log_return_list_np)
    metrics["Annualized Return"] = annual_Return(log_return_list_np)
    metrics["Annualized Volatility (Std)"] = annual_Std(log_return_list_np)

    mar_for_downside = risk_free_rate_annual
    metrics["Annualized Downside Std"] = annual_DownsideStd(
        log_return_list_np, required_return_annual=mar_for_downside
    )

    md_val, md_period = max_Drawdown(log_return_list_np)
    metrics["Max Drawdown"] = md_val
    metrics["Max Drawdown Period"] = md_period

    metrics["Sharpe Ratio"] = sharpe_Ratio(
        log_return_list_np, risk_free_rate_annual=risk_free_rate_annual
    )
    metrics["Sortino Ratio"] = sortino_Ratio(
        log_return_list_np,
        risk_free_rate_annual=risk_free_rate_annual,
        required_return_annual=mar_for_downside,
    )
    metrics["Calmar Ratio"] = calmar_Ratio(
        log_return_list_np, risk_free_rate_annual=risk_free_rate_annual
    )
    metrics["Skewness"] = skewness(log_return_list_np)
    metrics["Kurtosis"] = kurtosis_val(log_return_list_np)
    metrics["Annualized Turnover Rate"] = annual_Turnover_Rate(turnover_rate_list)

    return pd.DataFrame([metrics]).set_index("Policy")


# Plotting functions for Plotly (used by streamlit_app.py)
import plotly.graph_objects as go


def generate_portfolio_value_plot(portfolio_values_dict):
    fig = go.Figure()
    for name, series in portfolio_values_dict.items():
        plot_x = (
            series.index
            if isinstance(series.index, pd.DatetimeIndex)
            else np.arange(len(series))
        )
        fig.add_trace(go.Scatter(x=plot_x, y=series.values, mode="lines", name=name))
    fig.update_layout(
        title="Investment Portfolio Value Over Time",
        xaxis_title="Time Steps",
        yaxis_title="Portfolio Value",
        legend_title_text="Strategies",
    )
    return fig


def generate_weights_pie_chart(
    final_weights_series, title="Final Portfolio Allocation"
):
    valid_weights = final_weights_series[final_weights_series > 1e-5]

    if valid_weights.empty:
        fig = go.Figure()
        fig.update_layout(
            title_text=title,
            annotations=[
                dict(text="No significant weights to display", showarrow=False)
            ],
        )
        return fig

    fig = go.Figure(
        data=[
            go.Pie(
                labels=valid_weights.index,
                values=valid_weights.values,
                hole=0.3,
                textinfo="label+percent",
                insidetextorientation="radial",
            )
        ]
    )
    fig.update_layout(title_text=title)
    return fig
