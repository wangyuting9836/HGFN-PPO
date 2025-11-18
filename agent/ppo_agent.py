# _*_ coding: UTF-8 _*_

import numpy as np
import torch
import torch_scatter
from torch import nn
from torch.distributions import Categorical
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import mask_to_index

from fjsp_env import FJSPState
from hgnn import HGNN
from utils import IndexedDataset


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, is_orthogonal_init=False):
        super(MLPActor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.is_orthogonal_init = is_orthogonal_init

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linear_layers = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linear_layers.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

        # Trick: orthogonal initialization
        if self.is_orthogonal_init:
            if self.linear_or_not:
                # If linear model
                orthogonal_init(self.linear)
            else:
                # If MLP
                for layer in range(self.num_layers - 1):
                    orthogonal_init(self.linear_layers[layer])

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linear_layers[layer](h))
                # h = F.relu(self.linear_layers[layer](h))
            return self.linear_layers[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, is_orthogonal_init=False):
        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.is_orthogonal_init = is_orthogonal_init

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linear_layers = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linear_layers.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

        # Trick: orthogonal initialization
        if self.is_orthogonal_init:
            if self.linear_or_not:
                # If linear model
                orthogonal_init(self.linear)
            else:
                # If MLP
                for layer in range(self.num_layers - 1):
                    orthogonal_init(self.linear_layers[layer])

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linear_layers[layer](h)))
                '''
                h = torch.tanh(self.linear_layers[layer](h))
                # h = F.relu(self.linear_layers[layer](h))
            return self.linear_layers[self.num_layers - 1](h)


class Memory:
    def __init__(self):
        self.batch_graph = []
        self.graph_is_available = []
        self.log_action_prob = []
        self.state_value = []
        self.select_action_index_mask = []
        self.available_action_mask = []

        self.reward = []
        self.true_reward = []
        self.done = []
        self.terminal = []

    def clear(self):
        del self.batch_graph[:]
        del self.graph_is_available[:]
        del self.log_action_prob[:]
        del self.state_value[:]
        del self.select_action_index_mask[:]
        del self.available_action_mask[:]

        del self.reward[:]
        del self.true_reward[:]
        del self.done[:]
        del self.terminal[:]


class FJSPPolicy(nn.Module):
    def __init__(self, train_args):
        super(FJSPPolicy, self).__init__()
        self.device = train_args.device

        self.operation_in_dim = train_args.operation_in_dim
        self.machine_in_dim = train_args.machine_in_dim
        self.vehicle_in_dim = train_args.vehicle_in_dim
        self.o_to_m_edge_in_dim = train_args.o_to_m_edge_in_dim
        self.o_to_v_edge_in_dim = train_args.o_to_v_edge_in_dim
        self.ope_embedding_hidden_dim = train_args.ope_embedding_hidden_dim
        self.lstm_hidden_dim = train_args.lstm_hidden_dim
        self.operation_out_dim = train_args.operation_out_dim
        self.machine_out_dim = train_args.machine_out_dim
        self.vehicle_out_dim = train_args.vehicle_out_dim
        self.heads = train_args.heads
        self.hgnn_num_layers = train_args.hgnn_num_layers

        self.actor_hidden_layers = train_args.actor_hidden_layers
        self.critic_hidden_layers = train_args.critic_hidden_layers
        self.actor_in_dim = self.operation_out_dim * 2 + self.machine_out_dim * 2 + self.vehicle_out_dim * 2
        self.critic_in_dim = self.operation_out_dim + self.machine_out_dim + self.vehicle_out_dim
        self.actor_hidden_dim = train_args.actor_hidden_dim
        self.critic_hidden_dim = train_args.critic_hidden_dim
        self.actor_out_dim = 1
        self.critic_out_dim = 1

        self.hgnn = HGNN(self.operation_in_dim, self.machine_in_dim, self.vehicle_in_dim, self.o_to_m_edge_in_dim, self.o_to_v_edge_in_dim,
                         self.ope_embedding_hidden_dim, self.lstm_hidden_dim, self.operation_out_dim, self.machine_out_dim, self.vehicle_out_dim,
                         self.heads, self.hgnn_num_layers).to(self.device)
        self.actor = MLPActor(self.actor_hidden_layers, self.actor_in_dim, self.actor_hidden_dim, self.actor_out_dim).to(self.device)
        self.critic = MLPCritic(self.critic_hidden_layers, self.critic_in_dim, self.critic_hidden_dim, self.critic_out_dim).to(self.device)

    def get_action_and_value(self, state, memory, is_greedy, training, action_type='available_vehicle_tuple'):
        available_action_mask = None
        available_operation_actions = None
        available_machine_actions = None
        available_vehicle_actions = None
        available_batch_mask = None

        if action_type == 'available_vehicle_tuple':
            available_operation_actions, available_machine_actions, available_vehicle_actions, available_action_mask, available_batch_mask = \
                state.get_available_edge_info_1()
        elif action_type == 'all_tuple':
            available_operation_actions, available_machine_actions, available_vehicle_actions, available_action_mask, available_batch_mask = \
                state.get_available_edge_info_2()

        available_action_index = available_action_mask.nonzero().squeeze(-1)

        batch_graph = state.batch_graph
        hgnn_out = self.hgnn(batch_graph)
        pooled_operation = global_mean_pool(hgnn_out['operation'], batch_graph.batch_dict['operation'])
        pooled_machine = global_mean_pool(hgnn_out['machine'], batch_graph.batch_dict['machine'])
        pooled_vehicle = global_mean_pool(hgnn_out['vehicle'], batch_graph.batch_dict['vehicle'])

        operation_batch = batch_graph.batch_dict['operation']
        available_action_batch = operation_batch[available_operation_actions]

        actor_input = torch.cat([hgnn_out['operation'][available_operation_actions],
                                 hgnn_out['machine'][available_machine_actions],
                                 hgnn_out['vehicle'][available_vehicle_actions],
                                 pooled_operation[available_action_batch],
                                 pooled_machine[available_action_batch],
                                 pooled_vehicle[available_action_batch]], dim=1)
        actor_out = self.actor(actor_input)
        actor_out_softmax_result = torch_scatter.scatter_softmax(actor_out.squeeze(1), available_action_batch, dim=0)

        available_batch = mask_to_index(available_batch_mask)

        select_action = torch.full((3, batch_graph.batch_size), -1, dtype=torch.long)

        if is_greedy:
            # DRL-G, greedily picking actions with the maximum probability
            action_prob, max_prob_index = (
                torch_scatter.scatter_max(actor_out_softmax_result, available_action_batch, dim_size=batch_graph.batch_size, dim=0))
            select_action[0, available_batch] = available_operation_actions[max_prob_index[available_batch]]
            select_action[1, available_batch] = available_machine_actions[max_prob_index[available_batch]]
            select_action[2, available_batch] = available_vehicle_actions[max_prob_index[available_batch]]
            select_action_index = available_action_index[max_prob_index[available_batch]]
        else:
            # DRL-S, sampling actions following \pi
            action_prob = torch.zeros(batch_graph.batch_size, dtype=torch.float)
            select_action_index = torch.zeros_like(available_batch, dtype=torch.long)
            for idx, b in enumerate(available_batch):
                b_probs = actor_out_softmax_result[available_action_batch == b]
                sampled_idx = torch.multinomial(b_probs, num_samples=1)
                select_action[0, b] = available_operation_actions[available_action_batch == b][sampled_idx]
                select_action[1, b] = available_machine_actions[available_action_batch == b][sampled_idx]
                select_action[2, b] = available_vehicle_actions[available_action_batch == b][sampled_idx]
                select_action_index[idx] = available_action_index[available_action_batch == b][sampled_idx]
                action_prob[b] = b_probs[sampled_idx]

        if training:
            # memory.batch_graph.append(batch_graph)
            # memory.graph_is_available.append(available_batch_mask)
            # memory.log_action_prob.append(torch.log(action_prob))
            select_action_index_mask = torch.zeros_like(available_action_mask, dtype=torch.bool)
            select_action_index_mask[select_action_index] = True
            # memory.select_action_index_mask.append(select_action_index_mask)
            # memory.available_action_mask.append(available_action_mask)

            critic_input = torch.cat([pooled_operation, pooled_machine, pooled_vehicle], dim=1)
            critic_out = self.critic(critic_input)

            return select_action, torch.log(action_prob), critic_out.squeeze(1), available_batch_mask, select_action_index_mask, available_action_mask
        else:
            return select_action, None, None, None, None, None

    def evaluate(self, memory_batch_graph, memory_batch_graph_is_available, memory_batch_select_action_index_mask, memory_batch_available_action_mask):
        available_action_index = mask_to_index(memory_batch_available_action_mask)

        all_operation_actions = memory_batch_graph.edge_index_dict[('operation', 'o_m_action', 'machine')][0]
        all_machine_actions = memory_batch_graph.edge_index_dict[('operation', 'o_m_action', 'machine')][1]
        all_vehicle_actions = memory_batch_graph.edge_index_dict[('operation', 'o_v_action', 'vehicle')][1]

        available_operation_actions = all_operation_actions[available_action_index]
        available_machine_actions = all_machine_actions[available_action_index]
        available_vehicle_actions = all_vehicle_actions[available_action_index]

        operation_batch = memory_batch_graph.batch_dict['operation']
        available_action_batch = operation_batch[available_operation_actions]

        hgnn_out = self.hgnn(memory_batch_graph)
        pooled_operation = global_mean_pool(hgnn_out['operation'], memory_batch_graph.batch_dict['operation'])
        pooled_machine = global_mean_pool(hgnn_out['machine'], memory_batch_graph.batch_dict['machine'])
        pooled_vehicle = global_mean_pool(hgnn_out['vehicle'], memory_batch_graph.batch_dict['vehicle'])

        actor_input = torch.cat([hgnn_out['operation'][available_operation_actions],
                                 hgnn_out['machine'][available_machine_actions],
                                 hgnn_out['vehicle'][available_vehicle_actions],
                                 pooled_operation[available_action_batch],
                                 pooled_machine[available_action_batch],
                                 pooled_vehicle[available_action_batch]], dim=1)

        actor_out = self.actor(actor_input)
        actor_out_softmax_result = torch_scatter.scatter_softmax(actor_out.squeeze(1), available_action_batch, dim=0)

        available_batch = mask_to_index(memory_batch_graph_is_available)

        dist_entropy = torch.zeros_like(available_batch, dtype=torch.float)
        for idx, b in enumerate(available_batch):
            b_probs = actor_out_softmax_result[available_action_batch == b]
            b_dist = Categorical(b_probs)
            dist_entropy[idx] = b_dist.entropy()

        critic_input = torch.cat([pooled_operation[available_batch], pooled_machine[available_batch], pooled_vehicle[available_batch]], dim=1)
        critic_out = self.critic(critic_input)

        action_index = mask_to_index(memory_batch_select_action_index_mask[memory_batch_available_action_mask])
        action_log_prob = torch.log(actor_out_softmax_result[action_index])

        return action_log_prob, critic_out.squeeze(1), dist_entropy


class PPOAgent:
    def __init__(self, train_args, is_greedy=False):
        self.training = True
        self.is_greedy = is_greedy

        self.batch_size = train_args.batch_size

        self.device = train_args.device
        self.num_epochs = train_args.num_epochs
        self.learning_rate = train_args.learning_rate
        self.gamma = train_args.gamma
        self.gae_lambda = train_args.gae_lambda
        self.K_epochs = train_args.K_epochs
        self.clip_epsilon = train_args.clip_epsilon
        self.is_clip_v_loss = train_args.is_clip_v_loss
        self.A_coefficient = train_args.A_coefficient
        self.vf_coefficient = train_args.vf_coefficient
        self.entropy_coefficient = train_args.entropy_coefficient
        self.times_of_batch_size = train_args.times_of_batch_size

        self.is_anneal_lr = train_args.is_anneal_lr
        self.is_grad_clip = train_args.is_grad_clip
        self.max_grad_norm = train_args.max_grad_norm
        self.is_advantage_norm = train_args.is_advantage_norm

        self.policy = FJSPPolicy(train_args).to(self.device)
        # self.old_policy = copy.deepcopy(self.policy)
        # self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def get_action_and_value(self, state: FJSPState, action_type, memory: Memory = None):
        # return self.old_policy.take_action(state, memory, self.is_greedy, self.training)
        return self.policy.get_action_and_value(state, memory, self.is_greedy, self.training, action_type)

    def update(self, memory, epoch):
        memory_rewards = torch.stack(memory.reward, dim=0)
        memory_true_rewards = torch.stack(memory.true_reward, dim=0)
        memory_values = torch.stack(memory.state_value, dim=0)
        memory_dones = torch.stack(memory.done, dim=0)
        memory_terminals = torch.stack(memory.terminal, dim=0)

        # bootstrap value GAE
        memory_advantages = torch.zeros_like(memory_rewards)
        step = memory_rewards.shape[0]
        delta = memory_rewards[step - 1] * ~memory_terminals[step - 1] - memory_values[step - 1] * ~memory_terminals[step - 1]
        memory_advantages[step - 1] = delta
        for t in reversed(range(step - 1)):
            delta = memory_rewards[t] * ~memory_terminals[t] + self.gamma * memory_values[t + 1] * ~memory_dones[t] - \
                    memory_values[t] * ~memory_terminals[t]
            memory_advantages[t] = delta + self.gamma * self.gae_lambda * memory_advantages[t + 1]
        memory_returns = memory_advantages + memory_values
        total_returns_all_batch = torch.sum(memory_returns).item()

        # 计算每个action的真实奖励
        memory_true_returns = torch.zeros_like(memory_true_rewards)
        memory_true_returns[step - 1] = memory_true_rewards[step - 1] * ~memory_terminals[step - 1]
        for t in reversed(range(step - 1)):
            memory_true_returns[t] = memory_true_rewards[t] * ~memory_terminals[t] + self.gamma * memory_true_returns[t + 1] * ~memory_dones[t]
        total_true_returns_all_batch = torch.sum(memory_true_returns).item()

        # Trick: advantage normalization todo batch or mini batch
        if self.is_advantage_norm:
            memory_advantages = (memory_advantages - memory_advantages.mean()) / (memory_advantages.std() + 1e-8)

        memory_loader = DataLoader(IndexedDataset(memory.batch_graph), self.times_of_batch_size, shuffle=True, generator=torch.Generator(self.device))

        memory_graph_is_available = torch.stack(memory.graph_is_available, dim=0)
        memory_log_action_prob = torch.stack(memory.log_action_prob, dim=0)
        memory_select_action_index_mask = torch.stack(memory.select_action_index_mask, dim=0)
        memory_available_action_mask = torch.stack(memory.available_action_mask, dim=0)

        loss_item = 0
        p_loss_item = 0
        v_loss_tem = 0
        entropy_loss_item = 0
        old_approx_kl_item = 0
        approx_kl_item = 0
        clip_fractions = []
        mini_batch_count = 0
        for _ in range(self.K_epochs):
            for memory_mini_batch_graph, mini_batch_indices in memory_loader:
                memory_mini_batch_graph_is_available = memory_graph_is_available[mini_batch_indices].reshape(-1)
                memory_mini_batch_log_action_prob = memory_log_action_prob[mini_batch_indices].reshape(-1)[memory_mini_batch_graph_is_available]
                memory_mini_batch_select_action_index_mask = memory_select_action_index_mask[mini_batch_indices].reshape(-1)
                memory_mini_batch_available_action_mask = memory_available_action_mask[mini_batch_indices].reshape(-1)

                theta_log_prob, state_value, dist_entropy = self.policy.evaluate(memory_mini_batch_graph,
                                                                                 memory_mini_batch_graph_is_available,
                                                                                 memory_mini_batch_select_action_index_mask,
                                                                                 memory_mini_batch_available_action_mask)

                memory_mini_batch_advantages = memory_advantages[mini_batch_indices][~memory_terminals[mini_batch_indices]]
                log_ratio = theta_log_prob - memory_mini_batch_log_action_prob
                ratio = torch.exp(log_ratio)

                print('{} ratio mean: {:.5f}, ratio std: {:.5f}'.format(_, ratio.mean(), ratio.std()))
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fractions += [((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()]

                # Policy loss
                p_loss1 = -memory_mini_batch_advantages * ratio
                p_loss2 = -memory_mini_batch_advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                p_loss = torch.max(p_loss1, p_loss2).mean()

                # Value loss
                memory_mini_batch_returns = memory_returns[mini_batch_indices][~memory_terminals[mini_batch_indices]]
                if self.is_clip_v_loss:
                    # Trick: Value loss clipping
                    memory_mini_batch_values = memory_values[mini_batch_indices][~memory_terminals[mini_batch_indices]]
                    v_loss_un_clipped = (state_value - memory_mini_batch_returns) ** 2
                    v_clipped = memory_mini_batch_values + torch.clamp(
                        state_value - memory_mini_batch_values,
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    v_loss_clipped = (v_clipped - memory_mini_batch_returns) ** 2
                    v_loss_max = torch.max(v_loss_un_clipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((state_value - memory_mini_batch_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = dist_entropy.mean()

                loss = self.A_coefficient * p_loss + self.vf_coefficient * v_loss - self.entropy_coefficient * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                # Trick Gradient clip
                if self.is_grad_clip:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                loss_item += loss.item()
                p_loss_item += p_loss.item()
                v_loss_tem += v_loss.item()
                entropy_loss_item += -entropy_loss.item()
                old_approx_kl_item += old_approx_kl.item()
                approx_kl_item += approx_kl.item()

                mini_batch_count += 1

        # Trick: Annealing the learning rate
        if self.is_anneal_lr:
            self.anneal_learning_rate(epoch)

        # Refer to https://zhuanlan.zhihu.com/p/22530231545
        y_pred, y_target = memory_values[~memory_terminals].cpu().numpy(), memory_returns[~memory_terminals].cpu().numpy()
        var_y = np.var(y_target)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_target - y_pred) / var_y

        return loss_item / mini_batch_count, \
            p_loss_item / mini_batch_count, \
            v_loss_tem / mini_batch_count, \
            entropy_loss_item / mini_batch_count, \
            explained_var, \
            np.mean(clip_fractions), \
            old_approx_kl_item / mini_batch_count, \
            approx_kl_item / mini_batch_count, \
            total_returns_all_batch / torch.count_nonzero(~memory_terminals).item(), \
            total_true_returns_all_batch / torch.count_nonzero(~memory_terminals).item() \


    def train(self):
        self.training = True
        self.policy.train()
        # self.old_policy.train()

    def eval(self):
        self.training = False
        self.policy.eval()
        # self.old_policy.eval()

    def anneal_learning_rate(self, epoch):
        frac = 1.0 - epoch / self.num_epochs
        self.learning_rate = frac * self.learning_rate
        self.optimizer.param_groups[0]["lr"] = self.learning_rate
