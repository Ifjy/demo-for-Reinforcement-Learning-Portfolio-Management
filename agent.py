import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import numpy as np
import copy
import os


class DDPG_multitask:
    def __init__(
        self, actor, critic, lsre, writer, config  # 传入包含其他参数的配置字典
    ):
        self.actor = actor.to(config["device"])
        self.critic = critic.to(config["device"])
        self.lsre = lsre.to(config["device"])
        self.target_actor = copy.deepcopy(actor).to(config["device"])
        self.target_critic = copy.deepcopy(critic).to(config["device"])
        self.writer = writer
        self.device = config["device"]

        # 直接从 config 中读取参数
        self.sigma = config["sigma"]
        self.actor_lr = config["actor_lr"]
        self.actor_weight_decay = config["actor_weight_decay"]
        self.critic_lr = config["critic_lr"]
        self.critic_weight_decay = config["critic_weight_decay"]
        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.beta = config["beta"]
        self.mv = config["mv_or_not"]
        self.entropy_loss = config["entropy_loss"]
        self.entropy_coef = config["entropy_coef"]
        self.decay_rate = config["sigma_decay_rate"]
        self.param_record_interval = config["param_record_interval"]
        self.entropy_disappear_step = config["entropy_disappear_step"]

        # 初始化额外参数
        self.counter = 0
        self.critic_loss_list = []
        self.actor_loss_list = []
        self.eta = 0  # 初始化 eta
        self.eta_sigma = 0  # 初始化 eta_sigma
        self.value_coef = config["value_coef"]  # 价值函数bias系数
        self.value_bias = 0  # 价值函数bias
        self.eta_lr = config["eta_lr"]  # 学习率系数
        self.num_update_steps = config["num_update_steps"]  # 更新步数
        # 初始化优化器和学习率调度器
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.lsre.parameters()),
            lr=self.actor_lr,
            weight_decay=self.actor_weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()),
            lr=self.critic_lr,
            weight_decay=self.critic_weight_decay,
        )
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(
            self.actor_optimizer,
            step_size=config["actor_scheduler_step_size"],
            gamma=config["actor_scheduler_gamma"],
        )
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
            self.critic_optimizer,
            step_size=config["critic_scheduler_step_size"],
            gamma=config["critic_scheduler_gamma"],
        )

    def take_action(self, state, train_or_eva="train"):
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        state_hist, last_action = torch.tensor(state["history"], dtype=torch.float).to(
            self.device
        ), torch.tensor(state["weight"], dtype=torch.float).to(self.device)
        action = self.actor((self.lsre(state_hist), last_action))
        # action = torch.clamp(action, -0.25, 0.25)
        # if torch.sum(F.softmax(action,dim=1),dim=0)!=torch.tensors()
        # 给动作添加噪声，增加探索
        # if train_or_eva == "train":
        #     # 使得sigma 指数衰减 降低探索力度
        #     # 使用输出标准差动态调整噪声幅度
        #     decayed_sigma = self.sigma * np.exp(-self.decay_rate * self.counter)
        #     noise = torch.normal(0, decayed_sigma * torch.ones_like(action)).to(
        #         self.device
        #     )
        #     action = (1 - decayed_sigma) * action + decayed_sigma * F.softmax(
        #         noise, dim=1
        #     )  # 加上softmax 噪声不就好了？
        return action

    def soft_update(self, tnet, target_net):
        for param_target, param in zip(target_net.parameters(), tnet.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, transition_dict):
        self.train()
        self.counter = self.counter + 1
        states_hist = torch.tensor(
            transition_dict["states_hist"], dtype=torch.float
        ).to(self.device)
        states_hist = self.lsre(states_hist)
        states_last = torch.tensor(
            transition_dict["states_last"], dtype=torch.float
        ).to(self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.float).to(
            self.device
        )  # actions 存在大量nan
        batch_reward = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        if self.mv == 2:
            rewards = (
                batch_reward
                - self.beta * (batch_reward - self.eta) ** 2
                - (self.eta - self.beta * self.eta_sigma)
            )
        elif self.mv == 1:
            rewards = batch_reward - self.eta
        else:
            rewards = batch_reward
        next_states_hist = torch.tensor(
            transition_dict["next_states_hist"], dtype=torch.float
        ).to(self.device)
        next_states_hist = self.lsre(next_states_hist)
        next_states_last = torch.tensor(
            transition_dict["next_states_last"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        temp_action = self.target_actor((next_states_hist, next_states_last))
        # temp_action = F.softmax(temp_action, dim=1)
        next_q_values = self.target_critic(
            (next_states_hist, next_states_last),
            temp_action,
        )
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        temp_ = self.critic((states_hist, states_last), actions)
        critic_loss = torch.mean(
            F.mse_loss(temp_, q_targets - self.value_coef * self.value_bias)
        )
        self.writer.add_scalar(
            "Critic Loss", critic_loss.item(), global_step=self.counter
        )
        self.critic_loss_list.append(critic_loss.detach().cpu().numpy())
        # print([x.grad for x in self.critic_optimizer.param_groups[0]["params"]])
        self.critic_optimizer.zero_grad()
        # print([x.grad for x in self.critic_optimizer.param_groups[0]["params"]])
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        # 记录 critic 参数和梯度
        if self.counter % self.param_record_interval == 0:
            for name, param in self.critic.named_parameters():
                self.writer.add_histogram(
                    f"Critic/{name}",
                    param.clone().cpu().data.numpy(),
                    global_step=self.counter,
                )
                if param.grad is not None:
                    self.writer.add_histogram(
                        f"Critic/{name}_grad",
                        param.grad.clone().cpu().data.numpy(),
                        global_step=self.counter,
                    )

        # if self.counter % 100000 == 0:
        #     print([x.grad for x in self.critic_optimizer.param_groups[0]["params"]])
        self.critic_optimizer.step()

        temp_action2 = self.actor((states_hist, states_last))
        # temp_action2 = F.softmax(temp_action2, dim=1)
        actor_loss = -torch.mean(self.critic((states_hist, states_last), temp_action2))
        if self.entropy_loss == 1 and self.counter <= self.entropy_disappear_step:
            entropy_loss = torch.mean(
                -torch.sum(temp_action2 * torch.log(temp_action2 + 1e-8), dim=1)
            )
            actor_loss -= 0.01 * entropy_loss
        self.writer.add_scalar(
            "Actor Loss", actor_loss.item(), global_step=self.counter
        )
        self.actor_loss_list.append(actor_loss.detach().cpu().numpy())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()  # backward 之后很小
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        if self.counter % self.param_record_interval == 0:
            for name, param in self.actor.named_parameters():
                self.writer.add_histogram(
                    f"Actor/{name}",
                    param.clone().cpu().data.numpy(),
                    global_step=self.counter,
                )
                if param.grad is not None:
                    self.writer.add_histogram(
                        f"Actor/{name}_grad",
                        param.grad.clone().cpu().data.numpy(),
                        global_step=self.counter,
                    )
        # print([x.grad for x in self.actor_optimizer.param_groups[0]["params"]])
        self.actor_optimizer.step()
        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络/
        # 用于打印lr 查看 已验证scheduler 有效
        # print(f"step:{self.counter},actor lr:{self.actor_optimizer.param_groups[0]['lr']},critic lr:{self.critic_optimizer.param_groups[0]['lr']}")
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.eval()

    def train(self):
        # 将 actor、critic 和 lsre 设置为训练模式
        self.actor.train()
        self.critic.train()
        self.lsre.train()

    def eval(self):
        # 将 actor、critic 和 lsre 设置为评估模式
        self.actor.eval()
        self.critic.eval()
        self.lsre.eval()

    def save_checkpoint(self, checkpoint_path):
        # 检测地址 没有就创建
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "lsre_state_dict": self.lsre.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
                "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
                "counter": self.counter,
                "critic_loss_list": self.critic_loss_list,
                "actor_loss_list": self.actor_loss_list,
                "eta": self.eta,
                "eta_sigma": self.eta_sigma,
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path):

        # 没有地址报错
        assert os.path.exists(checkpoint_path), "Checkpoint {} not found!".format(
            checkpoint_path
        )
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.lsre.load_state_dict(checkpoint["lsre_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler_state_dict"])
        self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])
        self.counter = checkpoint["counter"]
        self.critic_loss_list = checkpoint["critic_loss_list"]
        self.actor_loss_list = checkpoint["actor_loss_list"]
        self.eta = checkpoint["eta"]
        self.eta_sigma = checkpoint["eta_sigma"]
