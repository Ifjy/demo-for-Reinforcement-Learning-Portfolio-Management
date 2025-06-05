import random
import collections
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs


class LayerNorm(nn.Module):
    """构造一个layernorm模块"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        "Norm"
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Add+Norm"""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "add norm"
        return self.norm(x + self.dropout(sublayer(x)))


class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=512, dropout_rate=0.03):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x


class PolicyNet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        in_features,
        embed_dim=100,
        feed_dim=1000,
        feed_hidden=512,
        num_actions=11,
        dropout_rate=0.05,
    ):
        super(PolicyNet, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.num_actions = num_actions
        self.feed_dim = (self.num_actions - 1) * embed_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.ffn = FeedForward(self.feed_dim, hidden_features=feed_hidden)
        self.subc = SublayerConnection(self.feed_dim, dropout_rate)
        self.lnorm = nn.LayerNorm(embed_dim)
        self.embed_fc = nn.Linear(self.in_features, 100)
        self.fc2 = nn.Linear(self.feed_dim + num_actions, 512)
        self.fc3 = nn.Linear(512, num_actions)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=10)

        # nn.init.kaiming_uniform_(self.conv1.weight, mode="fan_in", nonlinearity="tanh")
        # nn.init.kaiming_uniform_(self.conv2.weight, mode="fan_in", nonlinearity="tanh")
        # nn.init.kaiming_uniform_(self.conv3.weight, mode="fan_in", nonlinearity="tanh")
        # nn.init.kaiming_uniform_(self.conv4.weight, mode="fan_in", nonlinearity="tanh")
        # nn.init.kaiming_uniform_(self.fc2.weight, mode="fan_in", nonlinearity="tanh")
        # nn.init.kaiming_uniform_(self.fc3.weight, mode="fan_in", nonlinearity="tanh")

    def forward(self, x):
        # input should be  B * W * N * f
        # x tuple of (states_hist,states_last)
        hist_state, last_action = x
        if len(hist_state.shape) == 3:
            hist_state = torch.unsqueeze(hist_state, 0)
        embed_state = self.embed_fc(hist_state)  # B * W * N * embed
        embed_state = F.relu(embed_state)
        embed_state = embed_state.permute(0, 2, 1, 3)  # B * N * W * embed
        t_hist = [
            embed_state[i, :, :, :] for i in range(hist_state.shape[0])
        ]  # [10*100*100 -> N * w * embed]
        # t_hist[i] - > N * W * feature
        t_out = [
            self.attn(t_hist[i], t_hist[i], t_hist[i], need_weights=False)[0][:, -1, :]
            + t_hist[i][:, -1, :]  # 取最近的1天进行concat
            for i in range(len(t_hist))
        ]  # t_out be [10*100*100 -> N * w[-10:] * embed]
        t_out = [self.lnorm(t) for t in t_out]  # layer norm
        t_out = [t.reshape(t.shape[0], -1) for t in t_out]  # [N * 1*100]
        b_t_out = torch.stack(t_out, dim=0)  # B * N * 1024
        if len(b_t_out.shape) == 2:
            b_t_out = torch.unsqueeze(b_t_out, 0)
        b_t_out = b_t_out.reshape(b_t_out.shape[0], -1)  # [N * 100] -> B * (N*100)
        b_t_out = self.subc(b_t_out, self.ffn)
        # last action 应该为 B* （N+1）
        if len(last_action.shape) == 1:
            last_action = torch.unsqueeze(last_action, 0)
        elif len(last_action.shape) == 3:
            last_action = torch.squeeze(last_action)
        b_t_out = torch.cat([b_t_out, last_action], dim=1)
        b_t_out = self.fc2(b_t_out)
        b_t_out = F.tanh(b_t_out)
        out = self.fc3(b_t_out)
        return F.tanh(out)


class Qnet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        in_features,
        embed_dim=100,
        feed_dim=1000,
        feed_hidden=512,
        num_actions=11,
        dropout_rate=0.05,
    ):
        super(Qnet, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.num_actions = num_actions
        self.feed_dim = (self.num_actions - 1) * embed_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.ffn = FeedForward(self.feed_dim, hidden_features=feed_hidden)
        self.subc = SublayerConnection(self.feed_dim, dropout_rate)
        self.lnorm = nn.LayerNorm(embed_dim)
        self.embed_fc = nn.Linear(self.in_features, 100)
        self.fc2 = nn.Linear(self.feed_dim + 2 * num_actions, 512)
        self.fc3 = nn.Linear(512, num_actions)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=10)

        # nn.init.kaiming_uniform_(self.conv1.weight, mode="fan_in", nonlinearity="tanh")
        # nn.init.kaiming_uniform_(self.conv2.weight, mode="fan_in", nonlinearity="tanh")
        # nn.init.kaiming_uniform_(self.conv3.weight, mode="fan_in", nonlinearity="tanh")
        # nn.init.kaiming_uniform_(self.conv4.weight, mode="fan_in", nonlinearity="tanh")
        # nn.init.kaiming_uniform_(self.fc2.weight, mode="fan_in", nonlinearity="tanh")
        # nn.init.kaiming_uniform_(self.fc3.weight, mode="fan_in", nonlinearity="tanh")

    def forward(self, x, a):
        # input should be  B * W * N * f
        # x tuple of (states_hist,states_last)
        hist_state, last_action = x
        if len(hist_state.shape) == 3:
            hist_state = torch.unsqueeze(hist_state, 0)
        embed_state = self.embed_fc(hist_state)  # B * W * N * embed
        embed_state = F.relu(embed_state)
        embed_state = embed_state.permute(0, 2, 1, 3)  # B * N * W * embed
        t_hist = [
            embed_state[i, :, :, :] for i in range(hist_state.shape[0])
        ]  # [10*100*100 -> N * w * embed]
        # t_hist[i] - > N * W * feature
        t_out = [
            self.attn(t_hist[i], t_hist[i], t_hist[i], need_weights=False)[0][:, -1, :]
            + t_hist[i][:, -1, :]  # 取最近的1天进行concat
            for i in range(len(t_hist))
        ]  # t_out be [10*100*100 -> N * w[-10:] * embed]
        t_out = [self.lnorm(t) for t in t_out]  # layer norm
        t_out = [t.reshape(t.shape[0], -1) for t in t_out]  # [N * 1*100]
        b_t_out = torch.stack(t_out, dim=0)  # B * N * 1024
        if len(b_t_out.shape) == 2:
            b_t_out = torch.unsqueeze(b_t_out, 0)
        b_t_out = b_t_out.reshape(b_t_out.shape[0], -1)  # [N * 100] -> B * (N*100)
        b_t_out = self.subc(b_t_out, self.ffn)
        # last action 应该为 B* （N+1）
        if len(last_action.shape) == 1:
            last_action = torch.unsqueeze(last_action, 0)
        elif len(last_action.shape) == 3:
            last_action = torch.squeeze(last_action)
        if len(a.shape) == 1:
            a = torch.unsqueeze(a, 0)
        elif len(a.shape) == 3:
            a = torch.squeeze(a)
        b_t_out = torch.cat([b_t_out, last_action, a], dim=1)
        b_t_out = self.fc2(b_t_out)
        b_t_out = F.tanh(b_t_out)
        out = self.fc3(b_t_out)
        return F.tanh(out)


class cnn_actor(nn.Module):
    def __init__(self, in_channels, in_features, num_actions, hidden_size=512):
        super(cnn_actor, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        # input should be  B * W * N * f

        return


class cnn_critic(nn.Module):
    def __init__(self, in_channels, in_features, num_actions, hidden_size=512):
        super(cnn_critic, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x, a):

        return


class LSRE(torch.nn.Module):
    def __init__(
        self,
        window_size,
        in_features,
        embed_dim=100,
        num_actions=11,
    ):
        super(LSRE, self).__init__()
        self.window_size = window_size
        self.in_features = in_features
        self.num_assets = num_actions - 1
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.embed_fc = nn.Linear(self.in_features, 100)
        self.attn_list = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embed_dim, num_heads=10, batch_first=True
                )
                for _ in range(self.num_assets)
            ]
        )
        self.time_compress = nn.Linear(self.window_size, embed_dim // 10)
        self.embed_lnorm = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for i in range(num_actions - 1)]
        )
        self.lnorm_list = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for i in range(num_actions - 1)]
        )
        self.fc = nn.Linear(embed_dim * (embed_dim // 10), embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.03, 0.03)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.uniform_(m.in_proj_weight, -0.03, 0.03)
                if m.in_proj_bias is not None:
                    nn.init.uniform_(m.in_proj_bias, -0.03, 0.03)
                if m.out_proj.bias is not None:
                    nn.init.uniform_(m.out_proj.bias, -0.03, 0.03)
            elif isinstance(m, nn.LayerNorm):
                nn.init.uniform_(m.weight, -0.03, 0.03)
                nn.init.uniform_(m.bias, -0.03, 0.03)

    def forward(self, x):
        # input should be  B * W * N * f
        # x tuple of (states_hist,states_last)
        hist_state = x
        if len(hist_state.shape) == 3:
            hist_state = torch.unsqueeze(hist_state, 0)
        embed_state = self.embed_fc(hist_state)  # B * W * N * embed
        embed_state = F.relu(embed_state)
        embed_state = embed_state.permute(0, 2, 1, 3)  # B * N * W * embed
        asset_hist = [
            embed_state[:, i, :, :] for i in range(embed_state.shape[1])
        ]  #  B * w * embed]
        asset_hist = [
            self.embed_lnorm[i](t) for i, t in enumerate(asset_hist)
        ]  # layer norm B * W * embed
        asset_out = [
            self.attn_list[i](
                asset_hist[i], asset_hist[i], asset_hist[i], need_weights=False
            )[0]
            for i in range(len(asset_hist))
        ]  # B * W * embed
        asset_out = [asset_out[i] + asset_hist[i] for i in range(len(asset_hist))]
        asset_out = [
            self.lnorm_list[i](t) for i, t in enumerate(asset_out)
        ]  # layer norm B * W * embed
        b_t_out = torch.stack(asset_out, dim=1)  # B * N * W * embed
        b_t_out = b_t_out.permute(0, 1, 3, 2)  # B * N * embed * W
        b_t_out = self.time_compress(b_t_out)  # # B * N * embed * emb//10
        b_t_out = torch.reshape(
            b_t_out, (b_t_out.shape[0], b_t_out.shape[1], -1)
        )  # B * N * (embed//10 * embed)
        b_t_out = self.fc(b_t_out)  # B * N * embed
        b_t_out = F.relu(b_t_out)
        temp = b_t_out
        b_t_out = F.relu(self.fc2(b_t_out))  # B * N * embed
        b_t_out = self.fc3(b_t_out)  # B * N * embed
        b_t_out = temp + b_t_out  # B * N * embed
        b_t_out = self.ln(b_t_out)  # B * N * embed
        return b_t_out


class simpleLSRE(torch.nn.Module):
    def __init__(
        self,
        window_size,
        in_features,
        embed_dim=100,
        num_actions=11,
    ):
        super(simpleLSRE, self).__init__()
        self.window_size = window_size
        self.in_features = in_features
        self.num_assets = num_actions - 1
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.embed_fc = nn.Linear(self.in_features, 100)
        self.attn_list = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embed_dim, num_heads=10, batch_first=True
                )
                for _ in range(self.num_assets)
            ]
        )
        self.time_compress = nn.Linear(self.window_size, embed_dim // 10)
        # self.embed_lnorm = nn.ModuleList(
        #     [nn.LayerNorm(embed_dim) for i in range(num_actions - 1)]
        # )
        self.lnorm_list = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for i in range(num_actions - 1)]
        )
        self.fc = nn.Linear(embed_dim * (embed_dim // 10), embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.fc_simple = nn.Linear(embed_dim * self.window_size, self.embed_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.03, 0.03)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.uniform_(m.in_proj_weight, -0.03, 0.03)
                if m.in_proj_bias is not None:
                    nn.init.uniform_(m.in_proj_bias, -0.03, 0.03)
                if m.out_proj.bias is not None:
                    nn.init.uniform_(m.out_proj.bias, -0.03, 0.03)
            elif isinstance(m, nn.LayerNorm):
                nn.init.uniform_(m.weight, -0.03, 0.03)
                nn.init.uniform_(m.bias, -0.03, 0.03)

    def forward(self, x):
        # input should be  B * W * N * f
        # x tuple of (states_hist,states_last)
        hist_state = x
        if len(hist_state.shape) == 3:
            hist_state = torch.unsqueeze(hist_state, 0)
        embed_state = self.embed_fc(hist_state)  # B * W * N * embed
        embed_state = F.relu(embed_state)
        embed_state = embed_state.permute(0, 2, 1, 3)  # B * N * W * embed
        embed_state = torch.reshape(
            embed_state, (embed_state.shape[0], embed_state.shape[1], -1)
        )
        b_t_out = self.fc_simple(embed_state)
        return b_t_out


class Gru_LSRE(torch.nn.Module):
    def __init__(
        self,
        window_size,
        in_features,
        embed_dim=100,
        num_actions=11,
    ):
        super(Gru_LSRE, self).__init__()
        self.window_size = window_size
        self.in_features = in_features
        self.num_assets = num_actions - 1
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.embed_fc = nn.Linear(self.in_features, 100)
        self.Gru_list = nn.ModuleList(
            [
                nn.GRU(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)
                for _ in range(self.num_assets)
            ]
        )
        self.time_compress = nn.Linear(self.window_size, embed_dim // 10)
        # self.embed_lnorm = nn.ModuleList(
        #     [nn.LayerNorm(embed_dim) for i in range(num_actions - 1)]
        # )
        self.lnorm_list = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for i in range(num_actions - 1)]
        )
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.03, 0.03)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.uniform_(m.in_proj_weight, -0.03, 0.03)
                if m.in_proj_bias is not None:
                    nn.init.uniform_(m.in_proj_bias, -0.03, 0.03)
                if m.out_proj.bias is not None:
                    nn.init.uniform_(m.out_proj.bias, -0.03, 0.03)
            elif isinstance(m, nn.LayerNorm):
                nn.init.uniform_(m.weight, -0.03, 0.03)
                nn.init.uniform_(m.bias, -0.03, 0.03)

    def forward(self, x):
        # input should be  B * W * N * f
        # x tuple of (states_hist,states_last)
        hist_state = x
        if len(hist_state.shape) == 3:
            hist_state = torch.unsqueeze(hist_state, 0)
        embed_state = self.embed_fc(hist_state)  # B * W * N * embed
        embed_state = F.relu(embed_state)
        embed_state = embed_state.permute(0, 2, 1, 3)  # B * N * W * embed
        asset_hist = [
            embed_state[:, i, :, :] for i in range(embed_state.shape[1])
        ]  #  B * w * embed]
        asset_out = [
            torch.squeeze(
                self.Gru_list[i](
                    asset_hist[i],
                    torch.zeros(1, asset_hist[i].shape[0], asset_hist[i].shape[2]).to(
                        asset_hist[i].device
                    ),
                )[1],
                0,
            )
            for i in range(len(asset_hist))
        ]  # B * embed
        b_t_out = torch.stack(asset_out, dim=1)  # B * N * embed
        b_t_out = b_t_out.squeeze(2)  # B * N * embed
        b_t_out = self.fc(b_t_out)  # B * N * embed
        # b_t_out = F.relu(b_t_out)
        # temp = b_t_out
        # b_t_out = F.relu(self.fc2(b_t_out))  # B * N * embed
        # b_t_out = self.fc3(b_t_out)  # B * N * embed
        # b_t_out = temp + b_t_out  # B * N * embed
        # b_t_out = self.ln(b_t_out)  # B * N * embed
        return b_t_out


class BatchLSRE(torch.nn.Module):
    def __init__(
        self,
        window_size,
        in_features,
        embed_dim=100,
        num_actions=11,
    ):
        super(BatchLSRE, self).__init__()
        self.window_size = window_size
        self.in_features = in_features
        self.num_assets = num_actions - 1
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.embed_fc = nn.Linear(self.in_features, embed_dim)
        self.attn_list = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=10, batch_first=True
        )
        self.time_compress = nn.Linear(self.window_size, embed_dim // 10)
        self.embed_lnorm = nn.LayerNorm(embed_dim)
        self.lnorm_list = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim * (embed_dim // 10), embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.03, 0.03)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.uniform_(m.in_proj_weight, -0.03, 0.03)
                if m.in_proj_bias is not None:
                    nn.init.uniform_(m.in_proj_bias, -0.03, 0.03)
                if m.out_proj.bias is not None:
                    nn.init.uniform_(m.out_proj.bias, -0.03, 0.03)
            elif isinstance(m, nn.LayerNorm):
                nn.init.uniform_(m.weight, -0.03, 0.03)
                nn.init.uniform_(m.bias, -0.03, 0.03)

    def forward(self, x):
        # input should be  B * W * N * f
        # x tuple of (states_hist,states_last)
        hist_state = x
        flag = False
        if len(hist_state.shape) == 3:
            hist_state = torch.unsqueeze(hist_state, 0)
        elif len(hist_state.shape) > 4:
            flag = True
            tlen = hist_state.shape[0]
            batch = hist_state.shape[1]
            hist_state = hist_state.reshape(
                -1, self.window_size, self.num_assets, self.in_features
            )
        embed_state = self.embed_fc(hist_state)  # B * W * N * embed
        embed_state = F.relu(embed_state)
        embed_state = embed_state.permute(0, 2, 1, 3)  # B * N * W * embed
        b_size = embed_state.shape[0]
        N_size = embed_state.shape[1]
        embed_state = embed_state.reshape(
            b_size * N_size, self.window_size, self.embed_dim
        )  # B*N * W * embed
        asset_hist = self.embed_lnorm(embed_state)  # layer norm BN * W * embed
        asset_out = self.attn_list(
            asset_hist, asset_hist, asset_hist, need_weights=False
        )[
            0
        ]  # BN * W * embed
        torch.cuda.empty_cache()
        asset_out = asset_out + asset_hist
        asset_out = self.lnorm_list(asset_out)  # layer norm BN * W * embed
        # b_t_out = torch.stack(asset_out, dim=1)  # BN * W * embed
        b_t_out = asset_out.permute(0, 2, 1)  # BN * embed * W
        b_t_out = self.time_compress(b_t_out)  # # B * N * embed * emb//10
        b_t_out = torch.reshape(
            b_t_out, (b_size, N_size, -1)
        )  # B * N * (embed//10 * embed)
        b_t_out = self.fc(b_t_out)  # B * N * embed
        b_t_out = F.relu(b_t_out)
        temp = b_t_out
        b_t_out = F.relu(self.fc2(b_t_out))  # B * N * embed
        b_t_out = self.fc3(b_t_out)  # B * N * embed
        b_t_out = temp + b_t_out  # B * N * embed
        b_t_out = self.ln(b_t_out)  # B * N * embed
        if flag:
            b_t_out = b_t_out.reshape(tlen, batch, b_t_out.shape[1], b_t_out.shape[2])
        return b_t_out


class PolicyNet2(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        in_features,
        embed_dim=100,
        num_actions=11,
        hidden_size=512,
        portfolio_size=0.5,
    ):
        super(PolicyNet2, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.num_actions = num_actions
        self.num_assets = num_actions
        self.embed_dim = embed_dim
        self.portfolio_size = portfolio_size  # 用于定义有多少股票用于投资组合
        # 注意力机制
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=10, batch_first=True
        )
        # self.lstm = nn.LSTM(
        #     input_size=embed_dim,
        #     hidden_size=embed_dim,  # 保证输出维度一致
        #     batch_first=True,  # 与attention的batch_first对齐
        #     bidirectional=False,  # 双向LSTM需要调整hidden_size
        # )
        self.lnorm = nn.LayerNorm(embed_dim)

        # 嵌入层
        self.embed_fc = nn.Linear(embed_dim, embed_dim)

        # 全连接层，增加非线性能力
        self.fc1 = nn.Linear(embed_dim * (num_actions - 1) + num_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc5 = nn.Linear(hidden_size // 4, num_actions)
        self.fc6 = nn.Linear(num_actions * 2, num_actions)  # 输出动作概率分布
        self.TradeDecider = nn.Linear(
            embed_dim * (num_actions - 1) + num_actions, 1
        )  # 输出交易决策 帮助判断是否要进行交易
        # 归一化层
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(num_actions)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)  # 使用均匀分布初始化权重
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.03, 0.03)  # 使用均匀分布初始化偏置
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.kaiming_uniform_(
                    m.in_proj_weight, a=0, mode="fan_in", nonlinearity="relu"
                )
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        hist_state, last_action = x
        last_action = last_action.reshape(last_action.shape[0], -1)

        # 嵌入历史状态并通过注意力机制
        h = self.embed_fc(hist_state)  # B * N * embed
        attn_out = self.attn(h, h, h, need_weights=False)[0]
        # attn_out = self.lstm(h)[0]
        x = self.lnorm(attn_out + h)

        # 展平处理并连接上一步的动作
        x = x.reshape(x.shape[0], -1)  # B * (N*embed)
        x = torch.cat((x, last_action), dim=1)
        TradeSignal = F.tanh(
            self.TradeDecider(x)
        )  # 输出交易决策 帮助判断是否要进行交易
        trade_signal_mask = torch.ones_like(TradeSignal, dtype=torch.bool)
        # 全连接层带有 `tanh` 激活函数
        x = F.relu(self.fc1(x))
        temp = x
        x = F.relu(self.fc2(x))
        x = x + temp
        x = self.ln1(x)  # 第一层归一化
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)  # 输出动作概率分布
        x = torch.cat((x, last_action), dim=1)
        final_scores = self.fc6(x)  # 输出动作概率分布

        # 使用 `tanh` 限制输出
        # pm by deleting
        # Portfolio Management
        # 假设 TradeSignal 是一个张量，大小为 [batch_size, 1]

        # 对所有样本初始化 portfolio 为 last_action
        portfolio = last_action.clone()

        # 满足条件的样本索引
        valid_indices = torch.where(trade_signal_mask)[0]

        if self.portfolio_size != 1:
            num_winners = int(self.num_assets * self.portfolio_size)
            assert num_winners != 0 and num_winners <= self.num_assets

            # 只对满足交易信号的样本进行计算
            valid_scores = final_scores[valid_indices]
            rank = torch.argsort(valid_scores, descending=True, dim=1)
            winners = rank[:, :num_winners]

            # 创建 losers_mask
            losers_mask = torch.ones_like(valid_scores)
            for i in range(valid_scores.size(0)):
                losers_mask[i].scatter_(0, winners[i], 0)
            losers_mask *= -1e9

            # 计算 portfolio
            portfolio[valid_indices] = F.softmax(valid_scores + losers_mask, dim=1)
        else:
            # 直接对满足条件的样本计算 softmax
            valid_scores = final_scores[valid_indices]
            portfolio[valid_indices] = F.softmax(valid_scores, dim=1)

        return portfolio


class Critic2(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        in_features,
        embed_dim=100,
        num_actions=11,
        hidden_size=512,
    ):
        super(Critic2, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.num_actions = num_actions
        self.embed_dim = embed_dim

        # 注意力机制
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=10, batch_first=True
        )
        # self.lstm = nn.LSTM(
        #     input_size=embed_dim,
        #     hidden_size=embed_dim,  # 保证输出维度一致
        #     batch_first=True,  # 与attention的batch_first对齐
        #     bidirectional=False,  # 双向LSTM需要调整hidden_size
        # )
        self.lnorm = nn.LayerNorm(embed_dim)

        # 嵌入层
        self.embed_fc = nn.Linear(embed_dim, embed_dim)

        # 全连接层，与 `PolicyNet2` 保持一致
        self.fc1 = nn.Linear(
            embed_dim * (num_actions - 1) + num_actions * 2, hidden_size
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(
            hidden_size, hidden_size // 2
        )  # Critic 的输出为 Q 值，维度为 1
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc5 = nn.Linear(hidden_size // 4, 1)  # Critic 的输出为 V 值，维度为 1
        self.fc6 = nn.Linear(num_actions * 2 + 1, 1)
        # 归一化层
        self.ln1 = nn.LayerNorm(hidden_size)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)  # 使用均匀分布初始化权重
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.03, 0.03)  # 使用均匀分布初始化偏置
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.kaiming_uniform_(
                    m.in_proj_weight, a=0, mode="fan_in", nonlinearity="relu"
                )
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, a):
        hist_state, last_action = x
        last_action = last_action.reshape(last_action.shape[0], -1)
        a = a.reshape(a.shape[0], -1)  # 动作向量

        # 嵌入历史状态并通过注意力机制
        h = self.embed_fc(hist_state)  # B * N * embed
        attn_out = self.attn(h, h, h, need_weights=False)[0]
        # attn_out = self.lstm(h)[0]
        x = self.lnorm(attn_out + h)

        # 展平处理并连接动作
        x = x.reshape(x.shape[0], -1)  # B * (N*embed)
        x = torch.cat((x, last_action, a), dim=1)  # B * (N*embed + N + 1)

        # 全连接层，保持与 `PolicyNet2` 的结构一致
        x = F.relu(self.fc1(x))
        temp = x
        x = F.relu(self.fc2(x))
        x = x + temp
        x = self.ln1(x)  # 第一层归一化
        x = self.fc3(x)  # 输出 Q 值
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)  # 输出 V 值
        x = torch.cat((x, last_action, a), dim=1)  # B * (N*embed + N + 1)
        x = self.fc6(x)  # 输出 V 值
        return x


class PolicyNet3(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        in_features,
        embed_dim=100,
        num_actions=11,
        hidden_size=512,
    ):
        super(PolicyNet3, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.num_actions = num_actions
        self.embed_dim = embed_dim

        self.lsre = Gru_LSRE(100, in_features)
        # 注意力机制
        # self.attn = nn.MultiheadAttention(
        #     embed_dim=embed_dim, num_heads=10, batch_first=True
        # )
        # self.lnorm = nn.LayerNorm(embed_dim)

        # 嵌入层
        self.embed_fc = nn.Linear(embed_dim, embed_dim)

        # 全连接层，增加非线性能力
        self.fc1 = nn.Linear(embed_dim * (num_actions - 1) + num_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
        # self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        # self.fc5 = nn.Linear(hidden_size // 4, num_actions)
        # # 归一化层
        # self.ln1 = nn.LayerNorm(hidden_size)
        # self.ln2 = nn.LayerNorm(num_actions)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)  # 使用均匀分布初始化权重
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.03, 0.03)  # 使用均匀分布初始化偏置
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.kaiming_uniform_(
                    m.in_proj_weight, a=0, mode="fan_in", nonlinearity="relu"
                )
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        hist_state, last_action = x
        last_action = last_action.reshape(last_action.shape[0], -1)
        hist_state = self.lsre(hist_state)
        # 嵌入历史状态并通过注意力机制
        h = self.embed_fc(hist_state)  # B * N * embed
        # attn_out = self.attn(h, h, h, need_weights=False)[0]
        # x = self.lnorm(attn_out + h)
        x = h
        # 展平处理并连接上一步的动作
        x = x.reshape(x.shape[0], -1)  # B * (N*embed)
        x = torch.cat((x, last_action), dim=1)

        # 全连接层带有 `tanh` 激活函数
        x = F.relu(self.fc1(x))
        temp = x
        x = F.relu(self.fc2(x))
        x = x + temp
        # x = self.ln1(x)  # 第一层归一化
        x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        # x = self.fc5(x)  # 输出动作概率分布

        # 使用 `tanh` 限制输出
        return F.softmax(x, dim=1)


class Critic3(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        in_features,
        embed_dim=100,
        num_actions=11,
        hidden_size=512,
    ):
        super(Critic3, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.lsre = Gru_LSRE(100, in_features)
        # 注意力机制
        # self.attn = nn.MultiheadAttention(
        #     embed_dim=embed_dim, num_heads=10, batch_first=True
        # )
        # self.lnorm = nn.LayerNorm(embed_dim)

        # 嵌入层
        self.embed_fc = nn.Linear(embed_dim, embed_dim)

        # 全连接层，与 `PolicyNet2` 保持一致
        self.fc1 = nn.Linear(
            embed_dim * (num_actions - 1) + num_actions * 2, hidden_size
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Critic 的输出为 Q 值，维度为 1
        # self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        # self.fc5 = nn.Linear(hidden_size // 4, 1)  # Critic 的输出为 V 值，维度为 1
        # 归一化层
        # self.ln1 = nn.LayerNorm(hidden_size)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)  # 使用均匀分布初始化权重
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.3, 0.03)  # 使用均匀分布初始化偏置
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.kaiming_uniform_(
                    m.in_proj_weight, a=0, mode="fan_in", nonlinearity="relu"
                )
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, a):
        hist_state, last_action = x
        last_action = last_action.reshape(last_action.shape[0], -1)
        a = a.reshape(a.shape[0], -1)  # 动作向量
        hist_state = self.lsre(hist_state)
        # 嵌入历史状态并通过注意力机制
        h = self.embed_fc(hist_state)  # B * N * embed
        # attn_out = self.attn(h, h, h, need_weights=False)[0]
        # x = self.lnorm(attn_out + h)
        x = h
        # 展平处理并连接动作
        x = x.reshape(x.shape[0], -1)  # B * (N*embed)
        x = torch.cat((x, last_action, a), dim=1)  # B * (N*embed + N + 1)

        # 全连接层，保持与 `PolicyNet2` 的结构一致
        x = F.relu(self.fc1(x))
        temp = x
        x = F.relu(self.fc2(x))
        x = x + temp
        # x = self.ln1(x)  # 第一层归一化
        x = self.fc3(x)  # 输出 Q 值
        # x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        # x = self.fc5(x)  # 输出 V 值
        return x


# 简单版本 尽可能简化你的网络
class PolicyNet4(torch.nn.Module):
    # simple policy network
    def __init__(
        self,
        in_channels,
        in_features,
        embed_dim=100,
        num_actions=11,
        hidden_size=512,
        portfolio_size=0.5,
    ):
        super(PolicyNet4, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.num_actions = num_actions
        self.num_assets = num_actions
        self.embed_dim = embed_dim
        self.portfolio_size = portfolio_size  # 用于定义有多少股票用于投资组合
        # 注意力机制
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=10, batch_first=True
        )
        self.lnorm = nn.LayerNorm(embed_dim)

        # 嵌入层
        self.embed_fc = nn.Linear(embed_dim, embed_dim)

        # 全连接层，增加非线性能力
        self.fc1 = nn.Linear(
            embed_dim * (num_actions - 1) + num_actions,
            (embed_dim * (num_actions - 1) + num_actions) * 2,
        )
        self.fc2 = nn.Linear(
            (embed_dim * (num_actions - 1) + num_actions) * 2,
            (embed_dim * (num_actions - 1) + num_actions),
        )
        self.fc3 = nn.Linear((embed_dim * (num_actions - 1) + num_actions), hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size // 4)
        self.fc5 = nn.Linear(hidden_size, num_actions)
        self.fc6 = nn.Linear(num_actions * 2, num_actions)  # 输出动作概率分布
        self.TradeDecider = nn.Linear(
            embed_dim * (num_actions - 1) + num_actions, 1
        )  # 输出交易决策 帮助判断是否要进行交易
        # 归一化层
        self.ln1 = nn.LayerNorm(embed_dim * (num_actions - 1) + num_actions)
        self.ln2 = nn.LayerNorm(num_actions)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)  # 使用均匀分布初始化权重
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.03, 0.03)  # 使用均匀分布初始化偏置
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.kaiming_uniform_(
                    m.in_proj_weight, a=0, mode="fan_in", nonlinearity="relu"
                )
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                # 假设你想要为 bias 设置一个具体的 float 值 1.0
        # bias_value = 1.0

        # # 创建一个包含该值的张量，并设置 requires_grad=True 如果你想让其参与梯度计算
        # bias_tensor = torch.tensor(bias_value, requires_grad=True)

        # # 将张量包装成 Parameter
        # bias_param = nn.Parameter(bias_tensor)
        # self.TradeDecider.bias = bias_param  # really?

    def forward(self, x):
        hist_state, last_action = x
        last_action = last_action.reshape(last_action.shape[0], -1)

        # 嵌入历史状态并通过注意力机制
        h = self.embed_fc(hist_state)  # B * N * embed
        attn_out = self.attn(h, h, h, need_weights=False)[0]
        x = attn_out + h

        # 展平处理并连接上一步的动作
        x = x.reshape(x.shape[0], -1)  # B * (N*embed)
        x = torch.cat((x, last_action), dim=1)
        # x = F.relu(self.fc0(x))
        # TradeSignal = self.TradeDecider(x)
        # # 输出交易决策 帮助判断是否要进行交易  如果后续效果依然不行 将考虑换回tanh激活
        # # 全连接层带有 `tanh` 激活函数
        # # 假设 TradeSignal 是一个张量，大小为 [batch_size, 1]
        # # todo 尝试非二值输出情况下 但是初始让他为0————即初始全用新策略而不是last_action
        # trade_signal_mask = torch.sigmoid(
        #     TradeSignal
        # )  # 限制输出范围在 0-1 之间  初始 该sigmoid值<0.5
        # temp = x
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # x = x + temp
        # x = self.ln1(x)  # 第一层归一化
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc5(x)  # 输出动作概率分布
        x = torch.cat((x, last_action), dim=1)
        final_scores = self.fc6(x)  # 输出动作概率分布

        # 使用 `tanh` 限制输出
        # pm by deleting
        # Portfolio Management

        # 对所有样本初始化 portfolio 为 last_action
        portfolio = last_action.clone()

        # # 满足条件的样本索引
        # valid_indices = torch.where(trade_signal_mask)[0]

        if self.portfolio_size != 1:
            num_winners = int(self.num_assets * self.portfolio_size)
            assert num_winners != 0 and num_winners <= self.num_assets

            # 只对满足交易信号的样本进行计算
            valid_scores = final_scores
            rank = torch.argsort(valid_scores, descending=True, dim=1)
            winners = rank[:, :num_winners]

            # 创建 losers_mask
            losers_mask = torch.ones_like(valid_scores)
            for i in range(valid_scores.size(0)):
                losers_mask[i].scatter_(0, winners[i], 0)
            losers_mask *= -1e9

            # 计算 portfolio
            portfolio = F.softmax(valid_scores + losers_mask, dim=1)
        else:
            # 直接对满足条件的样本计算 softmax
            valid_scores = final_scores
            portfolio = F.softmax(valid_scores, dim=1)

        return portfolio


class Critic4(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        in_features,
        embed_dim=100,
        num_actions=11,
        hidden_size=512,
    ):
        super(Critic4, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.num_actions = num_actions
        self.embed_dim = embed_dim

        # 注意力机制
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=10, batch_first=True
        )
        self.lnorm = nn.LayerNorm(embed_dim)

        # 嵌入层
        self.embed_fc = nn.Linear(embed_dim, embed_dim)

        # 全连接层，与 `PolicyNet2` 保持一致
        self.fc0 = nn.Linear(
            embed_dim * (num_actions - 1) + num_actions * 2,
            embed_dim * (num_actions - 1) + num_actions * 2,
        )
        self.fc1 = nn.Linear(
            embed_dim * (num_actions - 1) + num_actions * 2,
            (embed_dim * (num_actions - 1) + num_actions * 2) * 2,
        )
        self.fc2 = nn.Linear(
            (embed_dim * (num_actions - 1) + num_actions * 2) * 2,
            (embed_dim * (num_actions - 1) + num_actions * 2),
        )
        self.fc3 = nn.Linear(
            (embed_dim * (num_actions - 1) + num_actions * 2), hidden_size
        )
        self.fc4 = nn.Linear(hidden_size, hidden_size // 4)
        self.fc5 = nn.Linear(hidden_size, 1)  # Critic 的输出为 V 值，维度为 1
        self.fc6 = nn.Linear(num_actions * 2 + 1, 1)
        # 归一化层
        self.ln1 = nn.LayerNorm(embed_dim * (num_actions - 1) + num_actions * 2)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.03, 0.03)  # 使用均匀分布初始化权重
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.03, 0.03)  # 使用均匀分布初始化偏置
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.kaiming_uniform_(
                    m.in_proj_weight, a=0, mode="fan_in", nonlinearity="relu"
                )
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, a):
        hist_state, last_action = x
        last_action = last_action.reshape(last_action.shape[0], -1)
        a = a.reshape(a.shape[0], -1)  # 动作向量

        # 嵌入历史状态并通过注意力机制
        h = self.embed_fc(hist_state)  # B * N * embed
        attn_out = self.attn(h, h, h, need_weights=False)[0]
        x = attn_out + h

        # 展平处理并连接动作
        x = x.reshape(x.shape[0], -1)  # B * (N*embed)
        x = torch.cat((x, last_action, a), dim=1)  # B * (N*embed + N + 1)
        # x = F.relu(self.fc0(x))
        # temp = x
        # 全连接层，保持与 `PolicyNet2` 的结构一致
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # x = x + temp
        # x = self.ln1(x)  # 第一层归一化
        x = self.fc3(x)  # 输出 Q 值
        # x = F.relu(x)
        # x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)  # 输出 V 值
        x = torch.cat((x, last_action, a), dim=1)  # B * (N*embed + N + 1)
        x = self.fc6(x)  # 输出 V 值
        return x
