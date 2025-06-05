# backend_logic.py
import streamlit as st
import torch
import json5
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import (
    font_manager,
)  # Keep for now, though not explicitly setting font paths
from pypfopt import expected_returns, risk_models, efficient_frontier
from env import Porfolio_Env  # Assumed to exist
from agent import DDPG_multitask  # Assumed to exist
import net  # Assumed to exist
from data_preprocess import data_process2  # Assumed to exist
import random as py_random
import logging
import sys
import traceback  # For detailed error logging if needed

# Configure basic logging
logging.basicConfig(
    stream=sys.stdout,  # Changed to stdout for better compatibility with Streamlit's console
    level=logging.INFO,  # Changed to INFO for less verbose default logging, DEBUG is fine too
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Default Paths and Configuration ---
DEFAULT_CONFIG_PATH = "config.jsonc"
DEFAULT_MODEL_PATH_TEMPLATE = "ddpg_multitask_experiment_model.pth"
DEFAULT_FTR_PATH = "000300.ftr"


def load_config_file(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json5.load(f)
        logger.info(f"Config file {config_path} loaded successfully.")
        return config
    except FileNotFoundError:
        st.error(f"错误：配置文件 {config_path} 未找到。")
        logger.error(f"Config file {config_path} not found.")
        return None
    except Exception as e:
        st.error(f"错误：加载配置文件 {config_path} 失败: {e}")
        logger.error(f"Error loading config file {config_path}: {e}", exc_info=True)
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

    # --- Seed Management ---
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

    logger.info(f"Using seed value: {seed_value}")
    py_random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    st.session_state["current_applied_seed"] = seed_value

    # 1. Load and process data
    raw_data_ftr_path = config_main.get("data_path_ftr", DEFAULT_FTR_PATH)
    try:
        logger.info(f"Loading data from: {raw_data_ftr_path}")
        raw_dataframe = pd.read_feather(raw_data_ftr_path)
        adata_full, all_original_stock_names = data_process2(
            raw_dataframe
        )  # Ensure data_process2 is robust
        logger.info(
            f"Data processed. Full data shape: {adata_full.shape}, Total original stocks: {len(all_original_stock_names)}"
        )
    except Exception as e:
        st.error(f"错误：加载或处理数据时出错: {e}")
        logger.error(f"Error loading or processing data: {e}", exc_info=True)
        return None, None, None, None, None

    if (
        not isinstance(adata_full, np.ndarray)
        or not isinstance(all_original_stock_names, list)
        or (
            adata_full.ndim == 3
            and adata_full.shape[1] != len(all_original_stock_names)
        )
    ):  # check for 3D array
        st.error("错误：数据处理 (data_process2) 返回的数据不一致或格式不正确。")
        logger.error(
            f"Data processing returned inconsistent data. adata_full type: {type(adata_full)}, names type: {type(all_original_stock_names)}"
        )
        if isinstance(adata_full, np.ndarray):
            logger.error(f"adata_full shape: {adata_full.shape}")
        if isinstance(all_original_stock_names, list):
            logger.error(
                f"all_original_stock_names length: {len(all_original_stock_names)}"
            )
        return None, None, None, None, None

    # 2. Select stocks
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
        logger.error(f"Invalid N_stock value: {N_stock}")
        return None, None, None, None, None

    if N_stock > len(all_original_stock_names):
        st.warning(
            f"警告：请求的 N_stock ({N_stock}) 大于可用股票数 ({len(all_original_stock_names)})。将使用所有可用股票。"
        )
        N_stock = len(all_original_stock_names)
    config_main["N_stock"] = N_stock

    if not all_original_stock_names:  # No stocks to select from
        st.error("错误：没有可供选择的股票 (all_original_stock_names 为空)。")
        logger.error("No stocks available for selection.")
        return None, None, None, None, None

    try:
        # Ensure sampling is possible
        if N_stock > len(
            all_original_stock_names
        ):  # Should have been caught, but double check
            N_stock = len(all_original_stock_names)

        selected_indices = py_random.sample(
            range(len(all_original_stock_names)), N_stock
        )
        selected_stock_names = [all_original_stock_names[i] for i in selected_indices]

        logger.info(
            f"Selected {N_stock} stocks. Indices: {selected_indices}, Names: {selected_stock_names}"
        )
        config_main["stock_names"] = selected_stock_names
        st.session_state["last_selected_indices"] = selected_indices
        st.session_state["last_selected_names"] = selected_stock_names
    except ValueError as e:
        st.error(
            f"错误：从股票列表中采样时出错: {e}. "
            f"可能是 N_stock ({N_stock}) 与列表长度 ({len(all_original_stock_names)})不匹配。"
        )
        logger.error(
            f"Error sampling from all_original_stock_names: {e}", exc_info=True
        )
        return None, None, None, None, None

    # 3. Split data to get test set
    total_time_steps = adata_full.shape[0]
    # Provide default split ratios if not in config to prevent errors
    train_start_prop = config_main.get("train_start_prop", 0.0)
    train_end_prop = config_main.get("train_end_prop", 0.7)
    test_end_prop = config_main.get("test_end_prop", 1.0)

    train_start_abs_idx = int(total_time_steps * train_start_prop)
    train_end_abs_idx = int(
        total_time_steps * train_end_prop
    )  # This is effectively the start of the test set
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
        logger.error(
            f"Invalid data split indices. T={total_time_steps}, TrainStart={train_start_abs_idx}, TrainEnd={train_end_abs_idx}, TestEnd={test_end_abs_idx}"
        )
        return None, None, None, None, None

    # adata_full is (time, stocks, features)
    # selected_indices are for the 'stocks' dimension
    test_data_np = adata_full[train_end_abs_idx:test_end_abs_idx, selected_indices, :]
    if test_data_np.shape[0] == 0:
        st.error("错误：测试数据集为空。请检查数据分割比例和总数据长度。")
        logger.error("Test data slice is empty.")
        return None, None, None, None, None
    logger.info(f"Test data shape: {test_data_np.shape}")

    # 4. Initialize model and environment
    in_channels = N_stock  # This should be the number of selected stocks
    if N_stock == 0:  # Should be caught earlier, but as safety
        st.error("错误: N_stock (选定股票数量) 为0，无法初始化模型。")
        logger.error("N_stock is 0, cannot initialize model.")
        return None, None, None, None, None

    in_features = test_data_np.shape[2]  # Number of features per stock
    num_actions = in_channels + 1  # Stocks + Cash

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
        logger.error(
            f"Missing config parameter for model initialization: {e}", exc_info=True
        )
        return None, None, None, None, None
    except Exception as e:
        st.error(f"错误：模型初始化失败: {e}")
        logger.error(f"Model initialization failed: {e}", exc_info=True)
        return None, None, None, None, None

    model_load_path = model_path_override
    if not model_load_path:  # Construct from config if override not provided
        folder_path = config_main.get("folder_path_for_model", "")
        exp_name = config_main.get("experiment_name_for_model", "")
        if not folder_path or not exp_name:
            st.warning(
                "警告：配置文件中缺少 'folder_path_for_model' 或 'experiment_name_for_model'。将尝试加载默认模型名称。"
            )
            model_load_path = DEFAULT_MODEL_PATH_TEMPLATE  # This might not exist
        else:
            # Example of constructing a path, adjust if your template is different
            # model_load_path = f"{folder_path}/{exp_name}/{DEFAULT_MODEL_PATH_TEMPLATE}"
            model_load_path = (
                DEFAULT_MODEL_PATH_TEMPLATE  # Assuming it's relative or full path
            )

    logger.info(f"Attempting to load model weights from: {model_load_path}")
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
        logger.info("Model weights loaded and models set to eval mode.")
    except FileNotFoundError:
        st.error(
            f"错误：模型文件 {model_load_path} 未找到。请确保路径正确或模型已训练。"
        )
        logger.error(f"Model file {model_load_path} not found.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"错误：加载模型权重失败 (路径: {model_load_path}): {e}")
        logger.error(
            f"Error loading model weights from {model_load_path}: {e}", exc_info=True
        )
        return None, None, None, None, None

    try:
        test_env = Porfolio_Env(data=test_data_np, config=config_main)
        logger.info("Portfolio_Env initialized for testing.")
    except Exception as e:
        st.error(f"错误：初始化 Portfolio_Env 失败: {e}")
        logger.error(f"Error initializing Portfolio_Env: {e}", exc_info=True)
        return None, None, None, None, None

    # Success message moved to streamlit_app.py after this function returns
    return agent, test_env, test_data_np, selected_stock_names, config_main


# --- Helper Function: Extract Results from Environment ---
def _extract_simulation_results(env: Porfolio_Env, config: dict, strategy_name: str):
    portfolio_values = pd.Series(env.asset_memory, name=strategy_name)
    if portfolio_values.empty:
        logger.warning(f"[{strategy_name}] Asset memory is empty.")
    else:
        logger.debug(
            f"[{strategy_name}] Asset memory length: {len(env.asset_memory)}, last value: {env.asset_memory[-1]:.2f}"
        )

    N_stock = config.get("N_stock", 0)
    num_expected_cols_actions = N_stock + 1

    stock_names_from_config = config.get("stock_names", [])
    # Ensure stock_names_from_config matches N_stock for column naming
    if len(stock_names_from_config) != N_stock:
        logger.warning(
            f"[{strategy_name}] Mismatch: len(stock_names)={len(stock_names_from_config)} vs N_stock={N_stock}. "
            "Using generic stock names for weights DataFrame columns."
        )
        # This might happen if N_stock was overridden but stock_names in config wasn't dynamically updated
        # However, load_all_assets *does* update config_main["stock_names"]
        columns_for_df_weights = ["Cash"] + [f"Stock_{i+1}" for i in range(N_stock)]
    else:
        columns_for_df_weights = ["Cash"] + stock_names_from_config

    if len(columns_for_df_weights) != num_expected_cols_actions:
        # This case should ideally not be hit if N_stock and stock_names_from_config are consistent
        logger.error(
            f"[{strategy_name}] Internal error: Constructed column names length ({len(columns_for_df_weights)}) "
            f"does not match expected action columns ({num_expected_cols_actions}). "
            f"N_stock: {N_stock}, stock_names: {stock_names_from_config}. Falling back to generic names."
        )
        columns_for_df_weights = [
            f"WeightCol_{i}" for i in range(num_expected_cols_actions)
        ]

    actions_for_df = np.empty((0, num_expected_cols_actions))
    if not env.action_memory:
        logger.warning(f"[{strategy_name}]: Action memory is empty.")
    else:
        try:
            # Attempt to vstack, assuming list of 1D arrays or compatible shapes
            actions_for_df = np.vstack(env.action_memory)
            if actions_for_df.ndim != 2:
                raise ValueError(f"vstack result not 2D, shape {actions_for_df.shape}")
            if actions_for_df.shape[1] != num_expected_cols_actions:
                logger.warning(
                    f"[{strategy_name}]: Action memory columns ({actions_for_df.shape[1]}) "
                    f"don't match expected ({num_expected_cols_actions}). This might occur if actions are malformed."
                )
                # Attempt to pad or truncate if this is a common issue, or adjust columns_for_df_weights
                # For now, we'll let final_columns_weights handle it.
        except (
            ValueError
        ) as ve_vstack:  # Handle cases where vstack fails (e.g. inconsistent shapes)
            logger.warning(
                f"[{strategy_name}]: np.vstack failed for action_memory: {ve_vstack}. Trying np.array."
            )
            try:
                temp_actions = np.array(env.action_memory)
                if (
                    temp_actions.ndim == 1 and len(env.action_memory) == 1
                ):  # Single action recorded
                    actions_for_df = temp_actions.reshape(1, -1)
                elif temp_actions.ndim == 2:
                    actions_for_df = temp_actions
                else:  # Fallback for other complex/jagged structures
                    logger.error(
                        f"[{strategy_name}]: Cannot form 2D action array. Shape: {temp_actions.shape if hasattr(temp_actions, 'shape') else 'N/A'}. Action memory content: {env.action_memory[:2]}"
                    )
                    # Create an empty DataFrame with correct columns if conversion fails badly
                    actions_for_df = np.empty((0, num_expected_cols_actions))
            except Exception as e_arr:
                logger.error(
                    f"[{strategy_name}]: Error converting action_memory to array: {e_arr}",
                    exc_info=True,
                )
                actions_for_df = np.empty((0, num_expected_cols_actions))
        logger.debug(
            f"[{strategy_name}] Actions for DataFrame shape: {actions_for_df.shape}"
        )

    final_columns_weights = columns_for_df_weights
    if actions_for_df.shape[0] > 0 and actions_for_df.shape[1] != len(
        columns_for_df_weights
    ):
        logger.warning(
            f"[{strategy_name}]: Actual action columns in data ({actions_for_df.shape[1]}) "
            f"!= expected number of columns based on names ({len(columns_for_df_weights)}). Using generic names."
        )
        final_columns_weights = [f"Weight_{i}" for i in range(actions_for_df.shape[1])]
    elif actions_for_df.shape[0] == 0:  # Empty data
        if actions_for_df.shape[1] != len(columns_for_df_weights):
            final_columns_weights = (
                [f"Weight_{i}" for i in range(actions_for_df.shape[1])]
                if actions_for_df.shape[1] > 0
                else ["NoActionData"]
            )

    portfolio_weights = pd.DataFrame(actions_for_df, columns=final_columns_weights)
    if (
        portfolio_weights.empty and actions_for_df.shape[0] > 0
    ):  # Data existed but DF creation failed
        logger.error(
            f"[{strategy_name}] portfolio_weights DataFrame is empty despite actions_for_df having data. Shape: {actions_for_df.shape}"
        )

    daily_log_returns_list = []
    if env.reward_memory:
        # Assuming reward_memory[0] might be an initial state or 0, actual returns start from index 1
        # This depends on Porfolio_Env's reward structure.
        if (
            len(env.reward_memory) > 1 and env.reward_memory[0] == 0
        ):  # Common pattern for initial 0 reward
            daily_log_returns_list = env.reward_memory[1:]
        elif (
            len(env.reward_memory) >= 1
        ):  # Take all if first is not 0, or only one reward
            daily_log_returns_list = env.reward_memory
        logger.debug(
            f"[{strategy_name}] Raw reward memory length: {len(env.reward_memory)}. Extracted log returns length: {len(daily_log_returns_list)}"
        )

    daily_log_returns = pd.Series(
        daily_log_returns_list, name=f"{strategy_name} Log Returns"
    )
    if daily_log_returns.empty and len(env.asset_memory) > 1:
        logger.warning(
            f"[{strategy_name}]: Log returns from reward_memory is empty. Recalculating from asset values."
        )
        daily_log_returns = get_log_returns_from_portfolio_values(portfolio_values)
        daily_log_returns.name = f"{strategy_name} Log Returns (Recalculated)"

    return portfolio_values, portfolio_weights, daily_log_returns


def get_log_returns_from_portfolio_values(portfolio_values_series):
    if not isinstance(portfolio_values_series, pd.Series):
        portfolio_values_series = pd.Series(portfolio_values_series)
    if portfolio_values_series.empty or len(portfolio_values_series) < 2:
        return pd.Series(dtype=float)  # Return empty series if not enough data
    # Clip to prevent log(0) or log(negative) if data is noisy
    portfolio_values_series = portfolio_values_series.clip(lower=1e-9)
    log_returns = np.log(
        portfolio_values_series / portfolio_values_series.shift(1)
    ).dropna()
    return log_returns


# --- RL Agent Simulation ---
def run_rl_simulation(agent, env: Porfolio_Env, config: dict):
    logger.info("Starting RL Agent simulation.")
    state, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        with torch.no_grad():
            action_tensor = agent.take_action(
                state, "eva"
            )  # Assuming "eva" is evaluation mode
        state, _, done, truncated, _ = env.step(action_tensor.detach().cpu().numpy())
    logger.info("RL Agent simulation finished.")
    return _extract_simulation_results(env, config, "RL Agent")


# --- EWS (Daily Equal Weight) Simulation ---
def run_ews_simulation(env: Porfolio_Env, config: dict):
    logger.info("Starting EWS (Daily) simulation.")
    env.reset()
    done = False
    truncated = False
    stock_num = config.get("N_stock", 0)  # Use .get for safety

    equal_action = np.zeros(stock_num + 1)  # [Cash, Stock1, ...]
    if stock_num > 0:
        equal_action[1:] = 1.0 / stock_num  # Equal weight to stocks, 0 to cash
    else:  # No stocks selected, all in cash
        equal_action[0] = 1.0

    while not done and not truncated:
        _, _, done, truncated, _ = env.step(equal_action)
    logger.info("EWS (Daily) simulation finished.")
    return _extract_simulation_results(env, config, "EWS (Daily)")


# --- Buy and Hold (B&H) Simulation ---
def run_buy_and_hold_simulation(env: Porfolio_Env, config: dict):
    logger.info("Starting Buy & Hold (B&H) simulation.")
    state, _ = env.reset()  # state includes 'weight' and 'history'
    done = False
    truncated = False

    N_stock = config.get("N_stock", 0)
    # portfolio_size_ratio is how many of N_stock to include, default 1.0 (all selected stocks)
    portfolio_size_ratio = config.get("portfolio_size", 1.0)

    initial_bnh_action = np.zeros(N_stock + 1)  # [Cash, Stock1, ..., StockN]

    if N_stock > 0:
        num_stocks_to_buy_initially = int(N_stock * portfolio_size_ratio)
        if portfolio_size_ratio > 0 and num_stocks_to_buy_initially == 0:
            num_stocks_to_buy_initially = 1  # Buy at least one if ratio > 0
        num_stocks_to_buy_initially = min(num_stocks_to_buy_initially, N_stock)

        if num_stocks_to_buy_initially > 0:
            # Select which stocks to include in B&H from the N_stock available to the env
            # This selection is based on the already selected N_stock for the environment
            available_stock_indices_in_env = list(
                range(N_stock)
            )  # These are indices from 0 to N_stock-1

            # py_random is seeded globally. Sample from the 0..N_stock-1 indices.
            selected_indices_for_bnh_portfolio = py_random.sample(
                available_stock_indices_in_env, num_stocks_to_buy_initially
            )

            weight_per_selected_stock = 1.0 / num_stocks_to_buy_initially
            for stock_idx_in_env in selected_indices_for_bnh_portfolio:
                initial_bnh_action[stock_idx_in_env + 1] = (
                    weight_per_selected_stock  # +1 for action vector (cash is 0)
                )
            initial_bnh_action[0] = 0.0  # Fully invested in selected stocks
            logger.debug(
                f"B&H: Initial buy of {num_stocks_to_buy_initially} stocks. Action: {initial_bnh_action}"
            )
        else:  # portfolio_size_ratio was 0 or num_stocks_to_buy became 0
            initial_bnh_action[0] = 1.0  # All cash
            logger.debug("B&H: No stocks to buy, all cash.")
    else:  # N_stock is 0
        initial_bnh_action[0] = 1.0  # All cash
        logger.debug("B&H: N_stock is 0, all cash.")

    first_step = True
    while not done and not truncated:
        # On first step, apply initial_bnh_action to establish the portfolio.
        # On subsequent steps, the action is current weights, meaning "hold current portfolio"
        # (weights will drift naturally due to price changes).
        action_to_take = initial_bnh_action if first_step else state["weight"].flatten()
        state, _, done, truncated, _ = env.step(action_to_take)
        first_step = False
    logger.info("Buy & Hold (B&H) simulation finished.")
    return _extract_simulation_results(
        env, config, "Buy & Hold (B&H)"
    )  # Consistent name


# --- Periodic Rebalancing Strategy Agents ---
def buyingWinner_agent_periodic(
    state, current_env_time, holding_period=20, N_stock_to_pick=10
):
    window_size = state["history"].shape[0]  # History: (window, num_stocks, features)
    num_available_stocks = state["history"].shape[1]

    if (current_env_time - window_size) % holding_period == 0:  # Rebalance condition
        if num_available_stocks == 0:
            return np.array([1.0])  # All cash if no stocks

        # Assuming feature 0 is the return/price used for performance
        # Mean return over the lookback window for each stock
        mean_performance = np.mean(state["history"][:, :, 0], axis=0)

        num_select = min(N_stock_to_pick, num_available_stocks)
        if num_select == 0:  # Should not happen if num_available_stocks > 0
            action = np.zeros(num_available_stocks + 1)
            action[0] = 1.0  # All cash
            return action

        # Indices of top performing stocks
        top_indices = np.argsort(mean_performance)[
            -num_select:
        ]  # Smallest to largest, take last num_select

        action = np.zeros(num_available_stocks + 1)  # [Cash, Stock1, ...]
        action[top_indices + 1] = 1.0 / num_select  # +1 to shift for cash at index 0
        # logger.debug(f"BuyingWinner t={current_env_time}: Rebalanced. Action: {action}")
    else:  # Hold current weights
        action = state["weight"].flatten()
    return action


def MeanVariance_agent_periodic(
    state, current_env_time, holding_period=20, risk_free_rate_annual=0.02
):
    window_size = state["history"].shape[0]
    num_available_stocks = state["history"].shape[1]

    debug_info = {"rebalancing_triggered": False}

    if (current_env_time - window_size) % holding_period == 0:
        debug_info["rebalancing_triggered"] = True
        if num_available_stocks == 0:
            debug_info["action"] = "All cash (no stocks)"
            # st.session_state[f"mv_debug_t{current_env_time}"] = debug_info # Optional: for app debugging
            return np.array([1.0])  # All cash

        # History: (window, num_stocks, features). Feature 0 is price/return.
        # PyPortfolioOpt expects daily returns: (days, assets)
        # If history stores prices, we need to calculate returns. If it stores returns, it's direct.
        # Assuming state["history"][:, :, 0] gives a (window_size, num_available_stocks) array of returns.
        returns_df = pd.DataFrame(state["history"][:, :, 0])
        debug_info["returns_df_shape"] = returns_df.shape

        if returns_df.shape[0] < 2:  # Not enough data for covariance
            debug_info["action"] = "Hold (not enough return data)"
            # st.session_state[f"mv_debug_t{current_env_time}"] = debug_info
            return state["weight"].flatten()

        try:
            # annualized_historical_return can estimate mu, or use returns_df directly if they are period returns
            mu = expected_returns.mean_historical_return(
                returns_df, compounding=False, frequency=252
            )  # Daily returns
            S = risk_models.sample_cov(returns_df, frequency=252)  # Daily returns

            ef = efficient_frontier.EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.max_sharpe(risk_free_rate=risk_free_rate_annual)
            cleaned_weights = ef.clean_weights(rounding=4)  # Rounds to 4 decimal places

            debug_info["ef_weights_raw"] = dict(cleaned_weights)

            action = np.zeros(num_available_stocks + 1)  # [Cash, Stock1, ...]
            current_sum_weights = 0

            # cleaned_weights keys are integer column indices of returns_df (0 to num_available_stocks-1)
            for stock_idx, weight in cleaned_weights.items():
                if 0 <= stock_idx < num_available_stocks:  # Ensure index is valid
                    action[stock_idx + 1] = weight  # +1 for action vector
                    current_sum_weights += weight

            action[0] = max(
                0, 1.0 - current_sum_weights
            )  # Cash is remainder, ensure non-negative

            # Normalize stock weights if cash was negative and clipped to 0
            if current_sum_weights > 1.0 + 1e-6:  # If sum of stock weights > 1
                action[1:] = action[1:] / current_sum_weights
                action[0] = 0.0  # Recalculate cash after normalization
                debug_info["normalized"] = True

            # logger.debug(f"MeanVariance t={current_env_time}: Rebalanced. Action: {action}")
            debug_info["action"] = action.tolist()

        except Exception as e:
            # Using print for backend, st.warning/error for Streamlit app directly
            print(
                f"MV Opt Warning/Error at t={current_env_time} for {num_available_stocks} stocks: {e}"
            )  # Log to console
            # traceback.print_exc() # For more detailed server-side logs
            debug_info["error"] = str(e)
            action = state[
                "weight"
            ].flatten()  # Hold current weights if optimization fails
            debug_info["action"] = "Hold (optimization error)"
        # st.session_state[f"mv_debug_t{current_env_time}"] = debug_info # Optional
    else:  # Hold current weights
        action = state["weight"].flatten()
    return action


def EWS_agent_periodic(state, current_env_time, holding_period=20):
    window_size = state["history"].shape[0]
    num_available_stocks = state["history"].shape[1]

    if (current_env_time - window_size) % holding_period == 0:
        action = np.zeros(num_available_stocks + 1)  # [Cash, Stock1, ...]
        if num_available_stocks > 0:
            action[1:] = 1.0 / num_available_stocks
        else:
            action[0] = 1.0  # All cash
        # logger.debug(f"EWS_periodic t={current_env_time}: Rebalanced. Action: {action}")
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
    logger.info(f"Starting Benchmark Agent simulation: {agent_name}")
    state, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        current_env_time = env.t  # Assuming env has a time attribute 't'
        action_np = agent_function(state, current_env_time, **agent_specific_params)
        state, _, done, truncated, _ = env.step(action_np)
    logger.info(f"Benchmark Agent simulation finished: {agent_name}")
    return _extract_simulation_results(env, config, agent_name)


# --- Plotting (Matplotlib for Heatmap) ---
def generate_weights_heatmap(weights_df: pd.DataFrame, strategy_name: str):
    """Generates a heatmap of portfolio weights over time with English labels and titles."""
    # Ensure Matplotlib uses its default font or system available fallback English font
    # No specific Chinese font settings needed here.
    plt.rcParams["axes.unicode_minus"] = False  # For correct minus sign display

    if (
        weights_df.empty or weights_df.shape[1] == 0
    ):  # No columns (assets) or no rows (time steps)
        fig, ax = plt.subplots(figsize=(10, 2))  # Smaller figure for no data
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

    data_to_plot = weights_df.T  # Transpose: Assets as Y-axis, Time as X-axis

    # Adjust height based on number of assets, ensuring a minimum height
    min_assets_for_large_yticks = 20
    min_fig_height = 4
    max_fig_height = 15  # Cap height
    asset_label_fontsize = 8

    if data_to_plot.shape[0] > min_assets_for_large_yticks:  # many assets
        fig_height = max(
            min_fig_height, min(max_fig_height, data_to_plot.shape[0] * 0.25)
        )
        if data_to_plot.shape[0] > 50:
            asset_label_fontsize = 6  # smaller font for very many assets
    else:  # few assets
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
    ax.set_yticklabels(
        data_to_plot.index, fontsize=asset_label_fontsize
    )  # Asset names from DataFrame index

    num_timesteps = data_to_plot.shape[1]
    step_size = max(1, num_timesteps // 10)  # Aim for about 10 ticks
    xtick_positions = np.arange(0, num_timesteps, step_size)
    ax.set_xticks(xtick_positions)

    # X-axis labels from original weights_df index (time steps)
    if isinstance(weights_df.index, pd.RangeIndex):
        xtick_labels = [str(i) for i in weights_df.index[::step_size]]
    else:  # DatetimeIndex or other
        xtick_labels = [str(label) for label in weights_df.index[::step_size]]
    ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=8)

    try:
        fig.tight_layout(pad=1.0)
    except Exception:
        # logger.warning("fig.tight_layout() failed in generate_weights_heatmap.", exc_info=False)
        pass  # Sometimes tight_layout can fail with certain backends or complex plots
    return fig


# --- Financial Metrics Calculations ---
# (annual_Std, annual_Return, cum_Return, sharpe_Ratio, max_Drawdown,
# calmar_Ratio, annual_DownsideStd, sortino_Ratio, skewness, kurtosis,
# calculate_financial_metrics functions remain as you provided, they are standard
# and seem correct. Ensure they handle empty lists gracefully, which they do.)


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
        return np.nan  # Need at least 2 points for std dev
    ann_ret = annual_Return(return_list)
    ann_std = annual_Std(return_list)
    if ann_std == 0 or np.isclose(ann_std, 0):
        return (
            np.nan
            if np.isclose(ann_ret - risk_free_rate_annual, 0)
            else np.inf * np.sign(ann_ret - risk_free_rate_annual)
        )
    return (ann_ret - risk_free_rate_annual) / (
        ann_std + 1e-9
    )  # Added epsilon for stability


def max_Drawdown(return_list: list):
    if len(return_list) == 0:
        return 0.0, (0, 0)
    cumulative_log_returns = np.cumsum(np.array(return_list))
    # Wealth curve, assuming starting wealth of 1
    portfolio_values = np.exp(np.insert(cumulative_log_returns, 0, 0.0))
    peak_values = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - peak_values) / (
        peak_values + 1e-9
    )  # Epsilon for stability
    max_dd = np.min(drawdowns)
    # Finding indices is optional, returning 0 as placeholder matches your original
    return max_dd, 0


def calmar_Ratio(return_list: list, risk_free_rate_annual=0.0):
    if len(return_list) < 2:
        return np.nan
    ann_ret = annual_Return(return_list)
    max_dd_val, _ = max_Drawdown(return_list)
    if abs(max_dd_val) < 1e-9:  # If drawdown is virtually zero
        return (
            np.nan
            if np.isclose(ann_ret - risk_free_rate_annual, 0)
            else np.inf * np.sign(ann_ret - risk_free_rate_annual)
        )
    return (ann_ret - risk_free_rate_annual) / (abs(max_dd_val) + 1e-9)


def annual_DownsideStd(return_list: list, required_return_annual=0.0):
    if len(return_list) == 0:
        return 0.0
    daily_mar = required_return_annual / 250.0  # Daily MAR
    return_list_np = np.array(return_list)
    # Differences from MAR; positive if return < MAR (i.e., shortfall)
    downside_diffs = daily_mar - return_list_np
    # Consider only actual shortfalls (where return < MAR, so diff > 0)
    downside_returns_squares = np.square(downside_diffs[downside_diffs > 0])
    if len(downside_returns_squares) == 0:
        return 0.0  # No returns below MAR
    mean_sq_downside = np.mean(downside_returns_squares)
    daily_downside_std = np.sqrt(mean_sq_downside)
    return daily_downside_std * np.sqrt(250)


def sortino_Ratio(
    return_list: list, risk_free_rate_annual=0.0, required_return_annual=0.0
):
    if len(return_list) < 2:
        return np.nan
    ann_ret = annual_Return(return_list)
    # For Sortino, MAR for downside deviation is often the risk-free rate or target return
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
        return np.nan  # Scipy needs at least 3 points for skew
    return scipy.stats.skew(return_list_np)


def kurtosis_val(
    return_list: list,
):  # Renamed to avoid conflict with scipy.stats.kurtosis
    return_list_np = np.array(return_list)
    if len(return_list_np) < 4:
        return np.nan  # Scipy needs at least 4 for kurtosis
    return scipy.stats.kurtosis(
        return_list_np, fisher=True
    )  # Fisher=True gives excess kurtosis


def calculate_financial_metrics(
    log_return_list, policy_name=None, risk_free_rate_annual=0.0
):
    log_return_list_np = np.array(log_return_list).flatten()  # Ensure 1D array

    if len(log_return_list_np) == 0:
        metrics_dict = {
            "Policy": policy_name,
            "Cumulative Return": np.nan,
            "Annualized Return": np.nan,
            "Annualized Volatility (Std)": np.nan,
            "Annualized Downside Std": np.nan,
            "Max Drawdown": np.nan,
            "Sharpe Ratio": np.nan,
            "Sortino Ratio": np.nan,
            "Calmar Ratio": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
        }
        return pd.DataFrame([metrics_dict]).set_index("Policy")

    metrics = {"Policy": policy_name}
    metrics["Cumulative Return"] = cum_Return(log_return_list_np)
    metrics["Annualized Return"] = annual_Return(log_return_list_np)
    metrics["Annualized Volatility (Std)"] = annual_Std(log_return_list_np)

    # For Sortino and DownsideStd, MAR is often the risk-free rate or 0.
    # Using risk_free_rate_annual as MAR for downside deviation as a common choice.
    mar_for_downside = risk_free_rate_annual
    metrics["Annualized Downside Std"] = annual_DownsideStd(
        log_return_list_np, required_return_annual=mar_for_downside
    )

    md_val, _ = max_Drawdown(log_return_list_np)
    metrics["Max Drawdown"] = md_val

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
    metrics["Kurtosis"] = kurtosis_val(log_return_list_np)  # Use the renamed function

    return pd.DataFrame([metrics]).set_index("Policy")


# Plotting functions for Plotly (used by streamlit_app.py)
import plotly.graph_objects as go


def generate_portfolio_value_plot(portfolio_values_dict):
    fig = go.Figure()
    for name, series in portfolio_values_dict.items():
        # Ensure series index is plottable (e.g. not multi-index or complex objects)
        # Using a simple range if index is not datetime-like for simplicity.
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
    # Filter out very small weights to make pie chart readable
    valid_weights = final_weights_series[
        final_weights_series > 1e-5
    ]  # Threshold for display

    if valid_weights.empty:
        # logger.warning(f"No significant weights for pie chart: '{title}'. All weights might be < 1e-5 or series empty.")
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
