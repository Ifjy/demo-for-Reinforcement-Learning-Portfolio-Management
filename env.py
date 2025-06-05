from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Porfolio_Env(gym.Env):
    """rlpm env

    Args:
        data (numpy.ndarray): 数据，形状为 T * N * M
        config (dict): 配置参数字典
    """

    def __init__(self, data, config):
        # 从配置字典中提取参数
        self.data = data
        self.terminal_time = data.shape[0]
        self.feature_num = data.shape[2]
        self.window_size = config["window_size"]
        self.transaction_cost = config["transaction_cost"]
        self.stock_num = config["N_stock"]
        self.t = self.window_size
        self.close_pos = config[
            "close_pos"
        ]  # 收盘价列在数组中的坐标 如 第x列 close_pos=x-1
        self.state = None
        self.reward = None
        self.done = False
        self.action = None
        self.init_wealth = config["init_wealth"]
        self.asset_memory = [self.init_wealth]
        self.action_memory = []
        self.reward_memory = [0]
        self.tcost_record = []
        self.turnover_record = []
        self.rf = config["risk_free"]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.stock_num + 1,), dtype=np.float64
        )
        self.truncated = False
        self.observation_space = spaces.Dict(
            {
                "history": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        self.window_size,
                        self.stock_num,
                        self.feature_num,
                    ),
                    dtype=np.float64,
                ),
                "weight": self.action_space,
            }
        )
        self.info = {"asset": self.asset_memory[-1], "time": self.t}

        if data.shape[0] < self.window_size * 2:
            raise Exception("data length < window_size")

    def normalization(self, indata):
        # 数据归一化
        mu = np.mean(indata, axis=0)
        sigma = np.std(indata, axis=0)
        return (indata - mu) / sigma

    def reset(self, seed=None):
        self.t = self.window_size
        self.asset_memory = [self.init_wealth]
        self.action_memory = []
        self.reward_memory = [0]
        self.reward = 0
        self.done = False
        self.action = np.zeros((1, self.stock_num + 1))
        self.action[0, 0] = 1  # 设定初始动作为all cash
        self.action_memory.append(self.action)
        self.state = {
            "history": self.data[0 : self.t, :, :],
            "weight": self.action,
        }
        self.truncated = False
        self.info = {"asset": self.asset_memory[-1], "time": self.t}
        self.tcost_record = []
        self.turnover_record = []
        return (self.state, self.info)

    def step(self, action):
        if len(action.shape) == 1:
            action = np.reshape(action, (1, self.stock_num + 1))
        # t初 进入市场，观察st = (self.t-W,self.t) 左闭右开 at在t末实现，并计算rt
        if self.t + 1 == self.terminal_time:  # 此时为末时刻T初，无需比较action
            self.done = True
            action = np.zeros((1, self.stock_num + 1))  # 将末期action变为 纯cash
            action[0, 0] = 1

        asset_prime = self.asset_memory[-1] * (
            1
            + np.sum(self.action[0, 1:] * self.data[self.t, :, self.close_pos])
            + self.action[0, 0] * self.rf
        )
        action_prime = (  # 因为价格的波动 期末的action会变动为action_prime  = a' = a*y/a^T*y
            self.asset_memory[-1]
            * self.action[0, 1:]
            * (1 + self.data[self.t, :, self.close_pos])
            / asset_prime
        )
        tcost = np.sum(
            -np.abs(action[0, 1:] - action_prime) * self.transaction_cost * asset_prime
        )
        self.tcost_record.append(tcost)  # 记录交易成本
        self.turnover_record.append(
            np.sum(np.abs(action[0, 1:] - action_prime))
        )  # 记录换手率 利用向量间差值比上1
        new_asset = asset_prime + tcost
        if not self.done:  # 没结束前 t增长 在done时 为了方便 会返回上一时刻的state
            self.t = self.t + 1
        self.state = {
            "history": self.data[self.t - self.window_size : self.t, :, :],
            "weight": action,
        }
        temp_reward = np.log(new_asset / self.asset_memory[-1])
        self.reward_memory.append(temp_reward)
        self.info = {"asset": self.asset_memory[-1], "time": self.t}
        self.asset_memory.append(new_asset)
        self.action_memory.append(action)
        self.action = action

        # todo 需要对final round 进行处理 final round
        return (
            self.state,
            self.reward_memory[-1],
            self.done,
            self.truncated,
            self.info,
        )

    def policy_return(self, action):
        # 给定一个策略返回asset list
        # input :steady action
        # 1 * N+1
        # self.terminal - self.windows * N+1
        data = self.data + 1
        data = data[self.window_size :, :, :]
        action = np.array(action)
        wlen = self.terminal_time - self.window_size
        if len(action.shape) == 3:
            action = np.squeeze(action)
        if np.array(action).shape[0] == 1:
            action = np.tile(action, (wlen, 1))
        save_asset_ = []
        init_wealth = 1
        save_init = init_wealth
        save_asset_.append(init_wealth)
        # todo 需要对final round 进行处理 final round
        for i in range(wlen):
            asset_prime = init_wealth * (
                1
                + np.sum(
                    action[i, 1:] * self.data[self.window_size + i, :, self.close_pos]
                )
                + action[i, 0] * self.rf
            )
            if i != wlen - 1:
                temp_action = action[i + 1, 1:]
            else:
                temp_action = np.zeros((1, action.shape[1] - 1))
            action_prime = (
                init_wealth
                * action[i, 1:]
                * (1 + self.data[self.window_size + i, :, self.close_pos])
                / asset_prime
            )
            tcost = np.sum(
                -np.abs(temp_action - action_prime)
                * self.transaction_cost
                * asset_prime
            )
            init_wealth = asset_prime + tcost
            save_asset_.append(init_wealth)
        # return_list = [
        #     np.log(save_asset_[i + 1] / save_asset_[i])
        #     for i in range(len(save_asset_) - 1)
        # ]
        return np.log(init_wealth / save_init), save_asset_  # return log reward


# if __name__ == "__main__":
#     data2 = np.random.randn(10000, 10, 10)
#     from stable_baselines3.common.env_checker import check_env

#     # 如果你安装了pytorch，则使用上面的，如果你安装了tensorflow，则使用from stable_baselines.common.env_checker import check_env
#     env = Porfolio_Env(data2)
#     check_env(env)
#     # from gymnasium.spaces import Dict, Box, Discrete

#     # observation_space = Dict({"test1":Box(-1, 1, shape=(2,)),"test2": Box(-1, 1, shape=(2,))}, seed=42)
#     # print(observation_space.sample())


class VecPorfolio_Env(gym.vector.VectorEnv):
    # rlpm vec env
    def __init__(self, env_class, num_envs, kwargs):
        self.envs = [env_class(kwargs) for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        observations = [env.reset() for env in self.envs]
        history = [obs[0]["history"] for obs in observations]
        weight = [obs[0]["weight"] for obs in observations]
        return np.array(history), np.array(weight)

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        observations, rewards, dones, infos = zip(*results)
        return np.array(observations), np.array(rewards), np.array(dones), infos
