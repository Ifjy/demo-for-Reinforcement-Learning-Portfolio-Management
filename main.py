import os
import torch
import json5
import random
import datetime
import numpy as np
from env import Porfolio_Env
from data_preprocess import data_process2
from agent import DDPG_multitask, DDPG
import shutil
from utils import (
    seed_everything,
    ReplayBuffer,
    stock_preview,
    EvaALL,
    result_plot,
    metric,
    copy_current_script_to_folder,
    mvddpg_alg,
    CustomSummaryWriter,
    plot_agent_statistics,
    ews_reward_df,
)
import net
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


# todo Stock preview 有误 需要修复
# todo PPO代码完善
# todo 确实后续action不愿意 变动 尝试 pm
# todo 现在观察到 action后续集中在同一个点附近 查看原因


def load_config(config_path="config.jsonc"):
    with open(config_path, "r") as f:
        config = json5.load(f)
    return config


def setup_environment(config):
    # Load data
    data = pd.read_feather(config["data_path"])
    adata, stock_names = data_process2(data)
    train_start = int(len(adata) * config["train_start"])
    train_length = int(len(adata) * config["train_length"])
    test_length = int(len(adata) * config["test_length"])
    # Select stocks
    pre_index = [
        "601390.XSHG",
        "600588.XSHG",
        "601618.XSHG",
        "600029.XSHG",
        "601186.XSHG",
        "000002.XSHE",
        "000768.XSHE",
        "002202.XSHE",
        "601766.XSHG",
        "601111.XSHG",
    ]
    stock_indices = [i for i in range(len(stock_names)) if stock_names[i] in pre_index]
    stock_names = [stock_names[i] for i in stock_indices]
    config["stock_names"] = stock_names
    # Define training and test datasets
    train_data = adata[
        train_start:train_length,
        stock_indices,
        :,
    ]
    test_data = adata[train_length:test_length, stock_indices, :]

    # Plot data preview
    fig, ax = stock_preview(
        adata[
            train_start:test_length,
            stock_indices,
            config["close_pos"],
        ],
        stock_names,
        train_pos=train_length - train_start,
        test_pos=test_length - train_start,
    )
    fig.savefig(config["folder_path"] + "stock_preview.png")
    fig.clf()
    return train_data, test_data, stock_names


def create_multi_task_agent(config, train_data):
    in_channels = config["N_stock"]
    in_features = train_data.shape[-1]
    num_actions = config["N_stock"] + 1

    actor = net.PolicyNet2(
        in_channels=in_channels,
        in_features=in_features,
        embed_dim=config["embed_dim"],
        num_actions=num_actions,
        hidden_size=config["hidden_size"],
        portfolio_size=config["portfolio_size"],
    ).to(config["device"])

    critic = net.Critic2(
        in_channels=in_channels,
        in_features=in_features,
        embed_dim=config["embed_dim"],
        num_actions=num_actions,
        hidden_size=config["hidden_size"],
    ).to(config["device"])
    if config["use_batch_lsre"] == 1:
        lsre = net.BatchLSRE(
            window_size=config["window_size"],
            in_features=in_features,
            embed_dim=config["embed_dim"],
            num_actions=num_actions,
        ).to(config["device"])
    else:
        lsre = net.LSRE(
            window_size=config["window_size"],
            in_features=in_features,
            embed_dim=config["embed_dim"],
            num_actions=num_actions,
        ).to(config["device"])
    agent = DDPG_multitask(
        actor=actor,
        critic=critic,
        lsre=lsre,
        writer=None,
        config=config,
    )
    return (agent,)


def main(config_path="config.jsonc"):
    # Load configuration and initialize seeds
    # test = f"{config["folder_path"]}agent_statistics/"

    config = load_config(config_path)
    seed_everything(config["seed"])
    # Set up directories and paths with timestamp and description
    wtime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    config["folder_path"] = f"{config['result_dir']}/{wtime}/"
    os.makedirs(config["folder_path"], exist_ok=True)
    # copy_current_script_to_folder(__file__, config["folder_path"])
    # Copy the configuration file to the results folder
    net_path = os.path.abspath(net.__file__)
    main_path = os.path.abspath(__file__)
    shutil.copyfile(config_path, f"{config['folder_path']}config.jsonc")
    shutil.copyfile(net_path, f"{config['folder_path']}net.py")
    shutil.copyfile(main_path, f"{config['folder_path']}main.py")
    # Log experiment description if provided
    if "experiment_description" in config:
        with open(f"{config['folder_path']}description.txt", "w") as f:
            f.write(config["experiment_description"])

    # Load environment and dataset
    train_data, test_data, stock_names = setup_environment(config)

    # Initialize agent and replay buffer
    if config["use_simple_agent"] == 1:
        agent, writer = create_simple_agent(config, train_data)
    else:
        agent, writer = create_multi_task_agent(config, train_data)
    # agent, writer = create_simple_agent(config, train_data)
    replay_buffer = ReplayBuffer(capacity=config["buffer_size"])

    # Define environment instances
    train_env = Porfolio_Env(
        data=train_data,
        config=config,
    )

    test_env = Porfolio_Env(
        data=test_data,
        config=config,
    )

    # Evaluate before training
    (
        train_set_ews_reward_list,
        test_set_ews_reward_list,
        eva_reward_list,
        save_return_list_train_set_before,
        save_return_list_test_set_before,
        before_train_metric_df,
        before_test_metric_df,
    ) = EvaALL(
        stock_num=config["N_stock"],
        train_env=train_env,
        test_env=test_env,
        agent=agent,
        file_head=config["experiment_name"],
        wtime=wtime,
        dpi=config["dpi"],
        path=config["folder_path"],
        subfix="before",
    )

    # Train agent using mvddpg_alg
    return_list, test_return_list, last_return_series = mvddpg_alg(
        env=train_env,
        agent=agent,
        replay_buffer=replay_buffer,
        test_env=test_env,
        config=config,
    )

    # Save trained model
    torch.save(
        {
            "critic_state_dict": agent.critic.state_dict(),
            "target_critic_state_dict": agent.target_critic.state_dict(),
            "actor_state_dict": agent.actor.state_dict(),
            "target_actor_state_dict": agent.target_actor.state_dict(),
            "lsre_state_dict": agent.lsre.state_dict(),
        },
        config["folder_path"] + config["experiment_name"] + "_model.pth",
    )

    # Evaluate after training and plot results
    (
        train_set_ews_reward_list,
        test_set_ews_reward_list,
        eva_reward_list,
        save_return_list_train_set_after,
        save_return_list_test_set_after,
        after_train_metric_df,
        after_test_metric_df,
    ) = EvaALL(
        stock_num=config["N_stock"],
        train_env=train_env,
        test_env=test_env,
        agent=agent,
        file_head=config["experiment_name"],
        wtime=wtime,
        dpi=config["dpi"],
        path=config["folder_path"],
        subfix="after",
    )
    result_plot(
        return_list=return_list,
        test_return_list=test_return_list,
        critic_loss_list=agent.critic_loss_list,
        actor_loss_list=agent.actor_loss_list,
        save_return_list_train_set_before=save_return_list_train_set_before,
        save_return_list_train_set_after=save_return_list_train_set_after,
        train_set_ews_reward_list=train_set_ews_reward_list,
        test_set_ews_reward_list=test_set_ews_reward_list,
        save_return_list_test_set_before=save_return_list_test_set_before,
        save_return_list_test_set_after=save_return_list_test_set_after,
        file_head=config["experiment_name"],
        wtime=wtime,
        dpi=config["dpi"],
        env_name=config["env_name"],
        path=config["folder_path"],
    )
    plot_agent_statistics(
        writer=writer, save_path=f'{config["folder_path"]}agent_statistics/'
    )
    # todo 增加对 每次评估计算return 使用agent.writer
    ews_metric_df_train, ews_metric_df_test = ews_reward_df(
        train_set_ews_reward_list,
        test_set_ews_reward_list,
    )
    metrics_all = pd.concat(
        [
            ews_metric_df_train,
            ews_metric_df_test,
            before_train_metric_df,
            after_train_metric_df,
            before_test_metric_df,
            after_test_metric_df,
        ],
        # todo add more policy like: BuyAndHold, BestSingle,WorstSingle..etc
    )
    print(metrics_all)
    metrics_all.to_csv(f"{config['folder_path']}{wtime}metrics.csv")


if __name__ == "__main__":
    main("/home/psdz/Lin PJ/rlpm/ddpg_cnn/config.jsonc")
