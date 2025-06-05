import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import scipy.stats
from scipy.stats.mstats import gmean
import scipy
import torch
import pickle
import random
from tqdm import tqdm
import os
import shutil
import torch.nn as nn
from io import BytesIO
from PIL import Image
import threading
from matplotlib import cm
import copy
from torch.utils.tensorboard import SummaryWriter


class CustomSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, **kwargs):
        super().__init__(log_dir, **kwargs)
        self.data = {}

    def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
        super().add_scalar(tag, scalar_value, global_step, **kwargs)
        if tag not in self.data:
            self.data[tag] = []
        self.data[tag].append((global_step, scalar_value))

    def get_data(self, tag):
        return self.data.get(tag, [])


def annual_Std(return_list: list):
    """
    :return: The annual standard deviation.
    """
    return_list = return_list.copy()
    dailyStd = np.std(return_list)  # 得到日度波动率
    annualStd = dailyStd * np.sqrt(250)  # 得到年化波动率
    return annualStd


def annual_Return(return_list: list):
    """
    :return: Annual return.
    """
    return_list = return_list.copy()
    annualReturn = np.mean(return_list) * 250  # 得到年化收益率
    return annualReturn


def cum_Return(return_list: list):
    """
    :return: The cumulative return.
    """
    return_list = return_list.copy()
    cumReturn = np.exp(np.sum(return_list))
    return cumReturn - 1


def sharpe_Ratio(return_list: list):
    """
    :return:  Sharpe ratio.
    """
    return_list = return_list.copy()
    annualReturn = annual_Return(return_list)
    annualStd = annual_Std(return_list)
    sharpeRatio = annualReturn / (annualStd + 1e-8)  # 得到夏普比
    return sharpeRatio


def max_Drawdown(return_list: list):
    """
    :return: Max drawdown of the financial series.
    """
    return_list = return_list.copy()
    # return_list.insert(0, 1)
    return_list = np.exp(np.cumsum(np.array(return_list)))
    df = pd.DataFrame(return_list)
    roll_max = df.expanding().max()
    drawdown = df / roll_max
    imin = np.argmin(drawdown)  # 得到最小值索引
    roll_max_before_min = drawdown[:imin]  # 得到最小值之前的最大值
    nearest_1_index = roll_max_before_min[roll_max_before_min == 1.0].last_valid_index()
    maxDrawdown = -1 * np.min(drawdown - 1)  # 计算得到最大回撤
    if nearest_1_index is None:
        maxDrawdown_period = 0
    else:
        maxDrawdown_period = imin - nearest_1_index  # 计算最大回撤周期
    return maxDrawdown, maxDrawdown_period


def calmar_Ratio(return_list: list):
    """
    :return: Calmar ratio.
    """
    return_list = return_list.copy()
    annualReturn = annual_Return(return_list)
    maxDrawdown = max_Drawdown(return_list)[0]
    calmarRatio = annualReturn / (maxDrawdown + 1e-8)
    return calmarRatio


def annual_DownsideStd(return_list: list):
    """
    :return:Annual downside standard deviation.
    """
    return_list = return_list.copy()
    df = pd.DataFrame(return_list)
    # num = （df<0).sum().item()  # 计算小于0的收益率个数
    dailyDownsideStd = np.sqrt(
        np.mean(np.square(df[df < 0]))
    )  # 计算日度下行波动率) # 计算出日度下行波动率
    annualDownsideStd = dailyDownsideStd * np.sqrt(250)
    return annualDownsideStd


def sortino_Ratio(return_list: list):
    """
    :return: Sortino ratio.
    """
    return_list = return_list.copy()
    annualReturn = annual_Return(return_list)
    annualDownsideStd = annual_DownsideStd(return_list)
    sortinoRatio = annualReturn / annualDownsideStd
    return sortinoRatio


# if __name__ == "__main__":
#     mr = [0.1, -0.2, 0.3, -0.4, 0.2, 0.2, 0.1]
#     print(sortino_Ratio(mr))

# print(annual_Std(mr))
# print(annual_Return(mr))
# print(cum_Return(mr))
# print(sharpe_Ratio(mr))
# print(max_Drawdown(mr))
# print(calmar_Ratio(mr))
# print(annual_DownsideStd(mr))
# print(sortino_Ratio(mr))
# print(1.08 ** (250 / 7))


def skewness(return_list: list):
    """
    :return: The skewness of the return.
    """
    return scipy.stats.skew(return_list)


def kurtosis(return_list: list):
    """
    :return: The kurtosis of the return.
    """
    return scipy.stats.kurtosis(return_list)


def metric(return_list, policy_name=None, benchmark=None):
    """计算风险调整指标（支持基准对比），按分析逻辑排序结果

    Args:
        return_list: 对数收益率序列
        policy_name: 策略名称
        benchmark: 基准收益率序列（可选）

    Returns:
        pd.DataFrame: 包含指标的数据框，按分析逻辑排序列

    指标顺序设计逻辑：
    1. 基础收益 -> 2. 风险指标 -> 3. 风险调整比率 -> 4. 分布特征
    """
    # 计算超额收益（如果提供基准）
    mr = (
        [return_list[i] - benchmark[i] for i in range(len(return_list))]
        if benchmark
        else return_list.copy()
    )

    metrics = {}
    metrics["policy"] = policy_name  # 策略标识

    # 1. 收益指标
    metrics["cum_Return"] = cum_Return(mr)  # 累计收益（首要指标）
    metrics["annual_Return"] = annual_Return(mr)  # 年化收益
    metrics["utility"] = annual_Return(mr) - 0.5 * annual_Std(mr)  # 效用函数

    # 2. 风险指标
    metrics["annual_Std"] = annual_Std(mr)  # 总波动率
    metrics["annual_DownsideStd"] = annual_DownsideStd(mr)  # 下行波动
    metrics["max_Drawdown"] = max_Drawdown(mr)[0]  # 最大回撤
    metrics["max_Drawdown_period"] = max_Drawdown(mr)[1]  # 回撤持续时间

    # 3. 风险调整收益
    metrics["sharpe_Ratio"] = sharpe_Ratio(mr)  # 夏普（总风险调整）
    metrics["sortino_Ratio"] = sortino_Ratio(mr)  # 索提诺（下行风险调整）
    metrics["calmar_Ratio"] = calmar_Ratio(mr)  # Calmar（回撤调整）

    # 4. 收益分布特征
    metrics["skewness"] = skewness(mr)  # 偏度（收益不对称性）
    metrics["kurtosis"] = kurtosis(mr)  # 峰度（极端风险）

    return pd.DataFrame([metrics])


# mr = [0.1, -0.2, 0.3, -0.4, 0.2, 0.2, 0.1]
# t = metric(mr)
# print(t)


def data_plot(data, stock_names=0, train_pos=100, test_pos=2000):
    # 用于画出股票走势图
    # 传入数据类型应该为 length * N  return数据
    # train_pos 为训练集 测试集划分位置
    # stock_nams 为股票名称 用于legend
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#000000",
    ]
    data = data + 1
    test = np.cumprod(data, axis=0)
    max_num = np.max(test)
    fig, ax = plt.subplots()
    for i, row in enumerate(test.T):
        ax.plot(row, label=stock_names[i], colors=colors[i], linewidth=0.5)
    ax.set_title("trend", fontsize=18)
    ax.set_xlabel("time", fontsize=18, fontfamily="sans-serif", fontstyle="italic")
    ax.set_ylabel("return", fontsize="x-large", fontstyle="oblique")
    ax.vlines(
        [train_pos, test_pos],
        ymin=0,
        ymax=max_num + 1,
        linestyles="dashed",
        colors="red",
    )
    ax.set_aspect("auto")  # 设置 x 轴方向拉长两倍
    ax.legend()
    return fig, ax


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(
        self,
        state_hist,
        state_last,
        action,
        reward,
        next_state_hist,
        next_state_last,
        done,
    ):
        self.buffer.append(
            (
                state_hist,
                state_last,
                action,
                reward,
                next_state_hist,
                next_state_last,
                done,
            )
        )

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        (
            state_hist,
            state_last,
            action,
            reward,
            next_state_hist,
            next_state_last,
            done,
        ) = zip(*transitions)
        return (
            np.array(state_hist),
            np.array(state_last),  # 出现警告与错误 因为长度不等？？
            np.array(action),
            np.array(reward),
            np.array(next_state_hist),
            np.array(next_state_last),
            np.array(done),
        )

    def clear(
        self,
    ):
        self.buffer.clear()

    def size(self):
        return len(self.buffer)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_image_to_tensorboard(
    writer, episode_actions, i_episode, stock_names, save_path=None, flag="train"
):
    """后台线程处理动作曲线生成和添加到 TensorBoard 的过程"""
    # 将动作序列转换为 NumPy 数组用于绘制
    episode_actions_np = np.concatenate(episode_actions, axis=0)  # 维度为 (2106, 11)

    # 检查 stock_names 的长度与动作维度匹配
    assert (
        episode_actions_np.shape[1] == len(stock_names) + 1
    ), "stock_names 的长度应为 action_dim - 1。"

    # 创建曲线图
    fig, ax = plt.subplots()
    cmap = cm.get_cmap("tab10", len(stock_names) + 1)
    # 绘制第一个维度（无风险资产）
    ax.plot(
        episode_actions_np[:, 0],
        color=cmap(-1),
        label="riskless_asset",
        linewidth=0.25,
    )

    # 绘制其余维度（有风险资产），并使用 stock_names 作为标签
    for i in range(1, episode_actions_np.shape[1]):
        ax.plot(
            episode_actions_np[:, i],
            color=cmap(i - 1),
            label=stock_names[i - 1],
            linewidth=0.25,
        )

    # 设置标题、标签和图例
    ax.set_title(f"Epoch {i_episode} Action Trends")
    ax.set_xlabel("Steps")
    ax.set_ylabel(f"{flag} Action Value")
    # 调整坐标轴位置以给图例留出空间
    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0, box.width * 0.8, box.height]
    )  # 缩小原图让出图例位置

    # 将图例放在图表外面
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 6})
    # 保存图片到指定路径
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)  # 确保目录存在
        fig.savefig(f"{save_path}/action_trends_epoch_{i_episode}.png")
    # 将图保存到内存并转换为 RGB 图像格式
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf).convert("RGB")  # 确保图像为 RGB 格式

    # 添加图像到 TensorBoard
    writer.add_image(
        f"{flag} Action Trends {i_episode}",
        np.array(image),
        global_step=i_episode,
        dataformats="HWC",
    )

    # 关闭图像以释放内存
    plt.close(fig)
    buf.close()


def update_agent_statistics(
    agent,
    Train_env,
    eval_episodes,
    test_env,
):
    """
    更新智能体的统计信息，并通过TensorBoard记录这些更新。

    参数:
        agent: 智能体对象，包含eta, eta_sigma, value_bias等属性及writer方法。
        temp_mu: 临时均值。
        temp_var: 临时方差。
        temp_value_bias: 临时价值偏差。
        i_episode: 当前的episode编号，用于记录到TensorBoard。
    """
    # 使用agent.eta_lr作为平滑参数
    with torch.no_grad():
        episode_reward, temp_var, temp_value_bias = mv_evaluate_policy(
            agent, Train_env, episodes=eval_episodes
        )
        temp_mu = np.mean(episode_reward)
        temp_cum_return = np.sum(episode_reward)
        episode_reward_test, _, _ = mv_evaluate_policy(
            agent, test_env, episodes=eval_episodes
        )
        temp_mu = np.mean(episode_reward)
        temp_cum_return_test = np.sum(episode_reward_test)

    if agent.eta == 0:
        # 如果eta为0，则直接设置新的统计值
        agent.eta = temp_mu
        agent.eta_sigma = temp_var  # 乘以学习率
        agent.value_bias = temp_value_bias  # 乘以学习率
    else:
        # 使用平滑参数更新统计值
        agent.eta = (
            agent.eta_lr * temp_mu + (1 - agent.eta_lr) * agent.eta
        )  # 乘以学习率
        agent.eta_sigma = (
            agent.eta_lr * temp_var + (1 - agent.eta_lr) * agent.eta_sigma
        )  # 乘以学习率
        agent.value_bias = (
            agent.eta_lr * temp_value_bias + (1 - agent.eta_lr) * agent.value_bias
        )  # 乘以学习率

    # 记录到TensorBoard
    agent.writer.add_scalar(
        "train_env_return", temp_cum_return, global_step=agent.counter
    )
    agent.writer.add_scalar(
        "test_env_return", temp_cum_return_test, global_step=agent.counter
    )
    agent.writer.add_scalar("Eta", agent.eta, global_step=agent.counter)
    agent.writer.add_scalar("Eta Sigma", agent.eta_sigma, global_step=agent.counter)
    agent.writer.add_scalar("Value Bias", agent.value_bias, global_step=agent.counter)
    agent.writer.add_scalar(
        "J",
        agent.eta - agent.beta * agent.eta_sigma,
        global_step=agent.counter,
    )


# 调用示例
# 假设我们已经有了agent实例，以及相应的参数
# update_agent_statistics(agent, new_mu, new_var, new_value_bias, current_episode)


def plot_agent_statistics(writer, save_path=None):
    """
    从 CustomSummaryWriter 中获取统计数据并为每个统计项绘制单独的曲线。

    参数:
        writer: CustomSummaryWriter 实例，包含要绘制的数据。
        save_path: 如果提供，则保存图像到该路径。如果不提供，则仅显示图像。
    """
    # 定义要绘制的标签
    tags = [
        "train_env_return",
        "test_env_return",
        "Eta",
        "Eta Sigma",
        "Value Bias",
        "J",
    ]

    for tag in tags:
        data = writer.get_data(tag)

        if not data:
            print(f"No data found for tag '{tag}'. Skipping.")
            continue

        steps, values = zip(*data)  # 解压步骤和值

        # 创建单个子图
        fig, ax = plt.subplots(figsize=(10, 5))

        # 绘制数据
        ax.plot(steps, values, label=tag)
        ax.set_title(f"{tag} over time")
        ax.set_xlabel("Global Step")
        ax.set_ylabel(tag)
        ax.legend()
        ax.grid(True)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)  # 确保目录存在
            filename = os.path.join(save_path, f"{tag}.png")  # 使用 tag 命名文件
            fig.savefig(filename)
            print(f"Plot saved to {filename}")
        else:
            plt.show()

        plt.close(fig)


def mvddpg_alg(env, agent, replay_buffer, test_env, config):
    eta_update_freq = config["eta_update_freq"]
    num_episodes = config["num_episodes"]
    minimal_size = config["minimal_size"]
    batch_size = config["batch_size"]
    eval_interval = config["eval_interval"]
    eval_episodes = config["eval_episodes"]
    # tau = config.get("tau", 0.1)  # 如果没有提供，则使用默认值
    image_interval = config.get("image_interval", 5)  # 如果没有提供，则使用默认值
    stock_names = config.get("stock_names", None)
    save_path = config.get("folder_path", None)
    num_update_steps = config.get("num_update_steps", 10000)
    return_list = []
    test_return_list = []
    N_ITER = 1
    N_show = 1
    agent.eta = 0
    agent.eta_sigma = 0
    agent.eval()
    train_action_save_path = f"{config['folder_path']}train_images/"
    test_action_save_path = f"{config['folder_path']}test_images/"
    for i in range(N_ITER):
        if agent.counter >= num_update_steps:
            break
        with tqdm(total=int(num_episodes / N_ITER), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 1)):
                # 如果达到最大更新步数 就停止 至此 num episodes大于num update steps 不再生效
                if agent.counter >= num_update_steps:
                    break
                episode_return = []
                state, info = env.reset()
                done = False
                counter = 0
                episode_actions = []  # 存储该 episode 的所有动作
                while not done:
                    counter += 1
                    action = agent.take_action(state)
                    episode_actions.append(action.detach().cpu().numpy())  # 记录动作
                    next_state, reward, done, _, _ = env.step(
                        action.detach().cpu().numpy()
                    )
                    replay_buffer.add(
                        state["history"],
                        state["weight"],
                        action.detach().cpu().numpy(),
                        reward,
                        next_state["history"],
                        next_state["weight"],
                        done,
                    )
                    state = next_state
                    episode_return.append(reward)

                    if replay_buffer.size() > minimal_size and (
                        counter % config["update_freq"] == 0
                    ):
                        b_s_h, b_s_l, b_a, b_r, b_ns_h, b_ns_l, b_d = (
                            replay_buffer.sample(batch_size)
                        )
                        transition_dict = {
                            "states_hist": b_s_h,
                            "states_last": b_s_l,
                            "actions": b_a,
                            "next_states_hist": b_ns_h,
                            "next_states_last": b_ns_l,
                            "rewards": b_r,
                            "dones": b_d,
                        }
                        agent.update(transition_dict)
                    if (
                        replay_buffer.size() > minimal_size
                        and agent.counter % eta_update_freq == 0
                    ):
                        update_agent_statistics(
                            agent,
                            copy.deepcopy(env),
                            eval_episodes,
                            copy.deepcopy(test_env),
                        )
                    # 如果达到最大更新步数 就停止 至此 num episodes大于num update steps 不再生效
                    if agent.counter >= num_update_steps:
                        break
                agent.writer.add_scalar(
                    "lr",
                    agent.actor_optimizer.state_dict()["param_groups"][0]["lr"],
                    global_step=i_episode,
                )
                # 每隔 image_interval 轮生成一次图像
                if (
                    (i_episode % image_interval == 0) or (i_episode == num_episodes - 1)
                ) and episode_actions:
                    threading.Thread(
                        target=add_image_to_tensorboard,
                        args=(
                            agent.writer,
                            copy.deepcopy(env.action_memory),
                            i_episode,
                            stock_names,
                            train_action_save_path,
                        ),
                    ).start()

                episode_total_reward = np.sum(episode_return)
                agent.writer.add_scalar(
                    "Episode Reward", episode_total_reward, global_step=i_episode
                )

                # 保存模型
                # if (i_episode + 1) % 10 == 0:
                #     agent.save_checkpoint(
                #         f"{config['folder_path']}checkpoint/checkpoint_{i_episode}.pth"
                #     )
                # # 记录均值和方差
                # if agent.mv:
                #     agent.writer.add_scalar("Eta", agent.eta, global_step=agent.counter)
                #     agent.writer.add_scalar(
                #         "Eta Sigma", agent.eta_sigma, global_step=agent.counter
                #     )

                return_list.append(episode_return)
                with torch.no_grad():
                    test_reward, _, fig, _, _, test_action_list, _ = evaluate_policy(
                        agent, test_env, episodes=1
                    )
                    test_return_list.append(test_reward)
                    agent.writer.add_scalar(
                        "Test Episode Reward", test_reward, global_step=i_episode
                    )
                    plt.close(fig)  # 关闭fig
                    # 每隔 image_interval 轮生成一次图像
                    if (
                        (i_episode % image_interval == 0)
                        or (i_episode == num_episodes - 1)
                    ) and episode_actions:
                        threading.Thread(
                            target=add_image_to_tensorboard,
                            args=(
                                agent.writer,
                                copy.deepcopy(test_env.action_memory),
                                i_episode,
                                stock_names,
                                test_action_save_path,
                                "eva",
                            ),
                        ).start()

                avg_len = config["avg_len"]
                if (i_episode + 1) % N_show == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d"
                            % (num_episodes / N_ITER * i + i_episode + 1),
                            "avg_return": "%.3f"
                            % np.mean([np.sum(i) for i in return_list[-avg_len:]]),
                            "cur_return": "%.3f" % episode_total_reward,
                            "test_return_avg": "%.3f"
                            % np.mean(test_return_list[-avg_len:]),
                        }
                    )
                pbar.update(1)
                if agent.counter >= num_update_steps:
                    break
    agent.writer.close()

    return [np.sum(i) for i in return_list], test_return_list, return_list[-1]


def evaluate_policy(agent, env, episodes=1):
    total_rewards = 0
    reward_list = []
    save_return_list = []
    save_max_return = -1e10
    for i in range(episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                action = agent.take_action(state, "eva")  # 使用eva 模式 减去动作随机性
            next_state, reward, done, _, _ = env.step(action.detach().cpu().numpy())
            episode_reward += reward
            state = next_state
        if episode_reward > save_max_return:
            save_max_return = episode_reward
            save_return_list = env.asset_memory
            save_action_list = env.action_memory
        total_rewards += episode_reward
        reward_list.append(episode_reward)
    # np.save(reward_list,"evaluate_reward.txt")
    average_reward = total_rewards / episodes
    fig, ax = plt.subplots()
    ax.plot(save_return_list, label="max return series")
    ax.set_title("trend")
    ax.set_xlabel("time", fontfamily="sans-serif", fontstyle="italic")
    ax.set_ylabel("return", fontstyle="oblique")
    return (
        average_reward,
        reward_list,
        fig,
        ax,
        save_return_list,
        save_action_list,
        save_max_return,
    )


def mv_evaluate_policy(agent, env, episodes=1):
    for i in range(episodes):
        state, info = env.reset()
        done = False
        episode_reward = []
        value_record = []
        while not done:
            with torch.no_grad():
                action = agent.take_action(state, "eva")  # train 模式 动作随机性
                next_state, reward, done, _, _ = env.step(action.detach().cpu().numpy())
                episode_reward.append(reward)
                states_hist = torch.tensor(state["history"], dtype=torch.float).to(
                    agent.device
                )
                states_hist = agent.lsre(states_hist)
                states_last = torch.tensor(state["weight"], dtype=torch.float).to(
                    agent.device
                )
                value_record.append(
                    agent.critic((states_hist, states_last), action).item()
                )
                state = next_state
    return episode_reward, np.var(episode_reward), np.mean(value_record)


def stock_preview(data, stock_names=0, train_pos=100, test_pos=2000):
    # 用于画出股票走势图
    # 传入数据类型应该为 length * N  return数据
    # train_pos 为训练集 测试集划分位置
    # stock_nams 为股票名称 用于legend

    data = data + 1
    test = np.cumprod(data, axis=0)
    max_num = np.max(test)
    cmap = cm.get_cmap("tab10", len(stock_names))  # 根据股票数量获取颜色映射
    fig, ax = plt.subplots()
    for i, row in enumerate(test.T):
        ax.plot(row, label=stock_names[i], color=cmap(i), linewidth=0.25)
    ax.set_title("Trend of Dataset", fontsize=14)
    ax.set_xlabel("Date", fontsize=14, fontfamily="DejaVu Sans", fontstyle="normal")
    ax.set_ylabel("Return", fontsize=14, fontfamily="DejaVu Sans", fontstyle="normal")
    ax.vlines(
        [train_pos],
        ymin=0,
        ymax=max_num + 1,
        linestyles="dashed",
        colors="red",
        linewidth=0.5,
        # 设置字体大小
    )
    ax.set_aspect("auto")  # 设置 x 轴方向拉长两倍
    # 将图例放在图表外面
    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0, box.width * 0.8, box.height]
    )  # 缩小原图让出图例位置
    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 5}
    )  # 图例放置在右侧外部
    return fig, ax


def EvaALL(
    stock_num,
    train_env,
    test_env,
    agent,
    file_head,
    wtime,
    dpi,
    path="/home/psdz/Lin PJ/rlpm/ddpg_cnn/figsave/",
    subfix="before",
):
    with torch.no_grad():
        equal_action = np.zeros((1, stock_num + 1))
        equal_action[0, 1:] = 1 / stock_num
        equal_return_train_set, train_set_ews_reward_list = train_env.policy_return(
            equal_action
        )
        equal_return_test_set, test_set_ews_reward_list = test_env.policy_return(
            equal_action
        )
        print(
            f"equal reward train set:{equal_return_train_set},test set:{equal_return_test_set}"
        )

        (
            avg_reward,
            eva_reward_list,
            fig,
            ax,
            save_return_list_train_set_before,
            save_action_list,
            max_return,
        ) = evaluate_policy(env=train_env, agent=agent)
        print(f"avg_reward {subfix} train, train set {avg_reward}")
        print(
            f"train env policy return action {train_env.policy_return(save_action_list)[0]},max return :{max_return}"
        )  #!! 依然存在可能的问题 需要细致查找 policy return 与 evaluate policy max不一致
        fig.savefig(
            f"{path}{file_head}{subfix}_train_eva" + wtime + ".png",
            dpi=dpi,
        )
        plt.close(fig)
        (
            avg_reward,
            eva_reward_list,
            fig,
            ax,
            save_return_list_test_set_before,
            save_action_list,
            max_return,
        ) = evaluate_policy(env=test_env, agent=agent, episodes=1)
        print(f"avg_reward {subfix} train, test set {avg_reward}")
        print(
            f"test env policy return action {test_env.policy_return(save_action_list)[0]},max return :{max_return}"
        )
        fig.savefig(
            f"{path}{file_head}{subfix}_test_eva" + wtime + ".png",
            dpi=dpi,
        )
        plt.close(fig)
        train_diff_reward = []
        test_diff_reward = []
        train_ews_diff_reward = []
        test_ews_diff_reward = []
        for i in range(1, len(save_return_list_train_set_before)):
            train_diff_reward.append(
                np.log(
                    save_return_list_train_set_before[i]
                    / save_return_list_train_set_before[i - 1]
                )
            )
            train_ews_diff_reward.append(
                np.log(train_set_ews_reward_list[i] / train_set_ews_reward_list[i - 1])
            )
            if i < len(save_return_list_test_set_before):
                test_diff_reward.append(
                    np.log(
                        save_return_list_test_set_before[i]
                        / save_return_list_test_set_before[i - 1]
                    )
                )
                test_ews_diff_reward.append(
                    np.log(
                        test_set_ews_reward_list[i] / test_set_ews_reward_list[i - 1]
                    )
                )

        train_metric_df = metric(
            train_diff_reward,
            policy_name=f"{subfix}_train",
            # benchmark=train_ews_diff_reward,
        )
        test_metric_df = metric(
            test_diff_reward,
            policy_name=f"{subfix}_test",
            # benchmark=test_ews_diff_reward,
        )
        train_vars_dict = {
            "tcost": train_env.tcost_record,
            "turnover": train_env.turnover_record,
            "action": train_env.action_memory,
            "asset_memory": train_env.asset_memory,
        }
        test_vars_dict = {
            "tcost": test_env.tcost_record,
            "turnover": test_env.turnover_record,
            "action": test_env.action_memory,
            "asset_memory": test_env.asset_memory,
        }
        # 保存
        with open(f"{path}{subfix}train_env.pkl", "wb") as f:
            pickle.dump(train_vars_dict, f)
        with open(f"{path}{subfix}test_env.pkl", "wb") as f:
            pickle.dump(test_vars_dict, f)
        return (
            train_set_ews_reward_list,
            test_set_ews_reward_list,
            eva_reward_list,
            save_return_list_train_set_before,
            save_return_list_test_set_before,
            train_metric_df,
            test_metric_df,
        )


def ews_reward_df(train_set_ews_reward_list, test_set_ews_reward_list):
    train_ews_diff_reward = []
    test_ews_diff_reward = []
    for i in range(1, len(train_set_ews_reward_list)):
        train_ews_diff_reward.append(
            np.log(train_set_ews_reward_list[i] / train_set_ews_reward_list[i - 1])
        )
        if i < len(test_set_ews_reward_list):
            test_ews_diff_reward.append(
                np.log(test_set_ews_reward_list[i] / test_set_ews_reward_list[i - 1])
            )
    ews_metric_df_train = metric(
        train_ews_diff_reward,
        policy_name=f"ews_train",
        # benchmark=train_ews_diff_reward,
    )
    ews_metric_df_test = metric(
        test_ews_diff_reward,
        policy_name=f"ews_test",
        # benchmark=train_ews_diff_reward,
    )
    return ews_metric_df_train, ews_metric_df_test


def result_plot(
    return_list,
    test_return_list,
    critic_loss_list,
    actor_loss_list,
    save_return_list_train_set_before,
    save_return_list_train_set_after,
    train_set_ews_reward_list,
    save_return_list_test_set_before,
    save_return_list_test_set_after,
    test_set_ews_reward_list,
    file_head,
    wtime,
    dpi,
    env_name,
    path="/home/psdz/Lin PJ/rlpm/ddpg_cnn/figsave/",
):
    episodes_list = list(range(len(return_list)))
    # 绘制训练与测试结果
    (
        fig,
        ax1,
    ) = plt.subplots()
    ax1.plot(episodes_list, return_list, linewidth=0.5, label="train episode return")
    ax1.plot(episodes_list, test_return_list, linewidth=0.5, label="test return record")
    ax1.set_xlabel("Episodes")  # no xlable
    ax1.set_ylabel("Returns")
    ax1.set_title(f"{file_head} on {env_name}")
    ax1.legend()
    plt.savefig(f"{path}agent_performance" + wtime + ".png", dpi=dpi)
    # plt.show()
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(critic_loss_list, linewidth=0.5, label="crritic loss")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("citicloss")
    ax1.set_title("loss")
    ax2.plot(actor_loss_list, linewidth=0.5, label="actor loss")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("actorloss")
    ax1.legend()
    ax2.legend()
    plt.savefig(f"{path}agent_loss_" + wtime + ".png", dpi=dpi)
    # plt.show()
    plt.close(fig)

    # 绘制训练前后的return对比图
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(save_return_list_train_set_before, linewidth=0.5, label="random policy")
    ax1.plot(save_return_list_train_set_after, linewidth=0.5, label="ddpg policy")
    ax1.plot(train_set_ews_reward_list, linewidth=0.5, label="EWS")
    ax2.plot(save_return_list_test_set_before, linewidth=0.5, label="random policy")
    ax2.plot(save_return_list_test_set_after, linewidth=0.5, label="ddpg policy")
    ax2.plot(test_set_ews_reward_list, linewidth=0.5, label="EWS")
    ax1.set_xlabel("t")
    ax1.set_ylabel("asset")
    ax1.set_title("train set comparison")
    ax2.set_xlabel("t")
    ax2.set_ylabel("asset")
    ax1.legend()
    ax2.legend()
    plt.savefig(
        f"{path}agent_train_test_comparison" + wtime + ".png",
        dpi=dpi,
    )
    # plt.show()
    plt.close(fig)

    var_dict = {
        "return_list": return_list,
        "test_return_list": test_return_list,
        "critic_loss": critic_loss_list,
        "actor_loss": actor_loss_list,
        "save_return_list_train_set_before": save_return_list_train_set_before,
        "save_return_list_train_set_after": save_return_list_train_set_after,
        "train_set_ews_reward_list": train_set_ews_reward_list,
        "save_return_list_test_set_before": save_return_list_test_set_before,
        "save_return_list_test_set_after": save_return_list_test_set_after,
        "test_set_ews_reward_list": test_set_ews_reward_list,
    }

    # 将变量保存到文件
    with open(f"{path}vars.pkl", "wb") as f:
        pickle.dump(var_dict, f)

    # # 从文件加载变量
    # with open('vars.pkl', 'rb') as f:
    #     var1, var2, var3 = pickle.load(f)


def copy_current_script_to_folder(file, destination_folder):
    # 获取当前脚本的绝对路径
    current_file = file
    # 如果使用的是某些 IDE（如 PyCharm），__file__ 可能会指向 .pyc 文件，需要处理这种情况
    if current_file.endswith(".pyc"):
        current_file = current_file[:-1]  # 去掉最后的 'c' 字符

    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 构建目标文件的完整路径
    destination_path = os.path.join(destination_folder, os.path.basename(current_file))

    try:
        # 复制文件
        shutil.copy2(current_file, destination_path)  # 使用 copy2() 保留元数据
        print(f"成功复制文件到 {destination_path}")
    except Exception as e:
        print(f"复制文件时发生错误: {e}")
