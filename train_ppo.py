# _*_ coding: UTF-8 _*_
import os
import time
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from visdom import Visdom

from agent import PPOAgent, Memory
from agent.normlization import Normalization, RewardScaling
from config.parse_args import parse_train_ppo_args
from fjsp_env.fjsp_env import FJSPEnv
from fjsp_lib import generate_train_data, read_data
from fjsp_lib.fjsp_graph_data import ObserveNormalizationRunningMeanStd, ObserveNormalizationGraph
from utils.train_dataset import TrainDataset, collate_fn


def train_ppo(train_args):
    current_directory = Path(__file__).resolve().parent

    # logger = create_logger(str(current_directory) + '/log', 'train_log')
    # logger.info('------Begin Training Model------')

    # Use visdom to visualize the training process
    viz_window = None
    is_visualize = True
    if is_visualize:
        viz_window = Visdom(env="training process")

    train_args.num_epochs = 2000
    operation_in_dim = train_args.operation_in_dim
    machine_in_dim = train_args.machine_in_dim
    vehicle_in_dim = train_args.vehicle_in_dim
    batch_size = train_args.batch_size

    train_instance_feature = [
        {"num_jobs": 10, "num_machines": 5, "num_vehicles": 2, "num_operations_per_job": (4, 6), "num_machine_per_operation": (1, 5)},
        # {"num_jobs": 20, "num_machines": 5, "num_vehicles": 5, "num_operations_per_job": (4, 6), "num_machine_per_operation": (1, 5)},
        # {"num_jobs": 20, "num_machines": 10, "num_vehicles": 10, "num_operations_per_job": (8, 12), "num_machine_per_operation": (1, 10)}
    ]

    # validation_data_dir_path = str(current_directory) + "/validation_data/{0}{1}/".format(str.zfill(str(10), 2), str.zfill(str(5), 2))

    validation_data_dir_path = str(current_directory) + "/validation_data/1005/"
    validation_datas = read_data(validation_data_dir_path, train_args.operation_in_dim, train_args.machine_in_dim, train_args.vehicle_in_dim)
    validation_batch_size = len(validation_datas)
    validation_batch = Batch.from_data_list([v_data[0] for v_data in validation_datas])
    validation_instances = [v_data[1] for v_data in validation_datas]

    validation_env = FJSPEnv(validation_batch, validation_instances, validation_batch_size, render_mode=False)

    train_data_size = train_args.train_data_size
    train_datas = generate_train_data(train_data_size, train_instance_feature, operation_in_dim, machine_in_dim, vehicle_in_dim)

    loader = DataLoader(TrainDataset(train_datas), train_args.batch_size, shuffle=True, generator=torch.Generator(train_args.device), collate_fn=collate_fn)

    agent = PPOAgent(train_args, is_greedy=False)
    memory = Memory()

    str_time = time.strftime('%Y-%m-%d-%H-%M')
    loss_return_file_path = str(current_directory) + '/result/loss_return_{}.csv'.format(str_time)
    make_span_file_path = str(current_directory) + '/result/make_span_{}.csv'.format(str_time)

    df = pd.DataFrame(columns=['iteration', 'loss', 'return'])
    df.to_csv(loss_return_file_path, index=False)
    df = pd.DataFrame(columns=['iteration', 'make span'])
    df.to_csv(make_span_file_path, index=False)

    best_make_span_mean = float('inf')
    max_num_of_save_model = 1
    best_models = deque()
    save_path = str(current_directory) + '/save_model'

    episode_index = 0
    update_episode_index = 0
    validation_episode_index = 0
    total_time = 0
    for epoch in range(train_args.num_epochs):
        for batch_graph, batch_instance in loader:
            env = FJSPEnv(batch_graph, batch_instance, batch_graph.batch_size, render_mode=False)
            state = env.reset()

            if train_args.observer_norm == 'observer_norm_graph':
                # Trick: observer normalization one graph
                state.batch_graph = observe_norm_graph(state.batch_graph)
            elif train_args.observer_norm == 'observer_norm_running':
                # Trick: observer normalization running mean std
                state.batch_graph = observe_norm_running(state.batch_graph)

            if train_args.reward_norm == 'reward_norm_scaling':
                # Trick: reward scaling
                reward_norm_scaling.reset()

            done = torch.zeros(train_args.batch_size, dtype=torch.bool)
            last_time = time.time()
            while not done.all():
                with torch.no_grad():
                    action, log_action_prob, value, available_batch_mask, select_action_index_mask, available_action_mask = \
                        agent.get_action_and_value(state, action_type=train_args.action_type, memory=memory)
                    next_state, reward, done, terminal = env.step(action, reward_type=train_args.reward_type, action_type=train_args.action_type)

                    if train_args.reward_norm == 'reward_norm_running':
                        # Trick: reward normalization
                        norm_reward = reward_norm_running(reward)
                    elif train_args.reward_norm == 'reward_norm_scaling':
                        # Trick: reward scaling
                        norm_reward = reward_norm_scaling(reward)

                    memory.batch_graph.append(state.batch_graph)
                    memory.graph_is_available.append(available_batch_mask)
                    memory.log_action_prob.append(log_action_prob)
                    memory.state_value.append(value)
                    memory.select_action_index_mask.append(select_action_index_mask)
                    memory.available_action_mask.append(available_action_mask)
                    memory.reward.append(norm_reward)
                    memory.true_reward.append(reward)
                    memory.done.append(done)
                    memory.terminal.append(terminal)

                    if train_args.observer_norm == 'observer_norm_graph':
                        # Trick: observer normalization one graph
                        next_state.batch_graph = observe_norm_graph(next_state.batch_graph)
                    elif train_args.observer_norm == 'observer_norm_running':
                        # Trick: observer normalization running mean std
                        next_state.batch_graph = observe_norm_running(next_state.batch_graph)

                    state = next_state

            cur_time = time.time()
            episode_time = cur_time - last_time
            total_time += episode_time
            episode_index += 1
            print('{}th episode spend time:  {:.4}'.format(episode_index, cur_time - last_time))

            if episode_index == 1000:
                avg_time_per_episode = total_time / episode_index
                print(f"Average time for 1000 episodes: {avg_time_per_episode:.4f} seconds")

            if episode_index % train_args.policy_update_episodes == 0:
                update_episode_index += 1
                last_time = time.time()
                loss, p_loss, v_loss, entropy_loss, explained_var, clip_fraction, old_approx_kl, approx_kl, avg_return, avg_true_return = \
                    agent.update(memory, epoch)
                memory.clear()
                cur_time = time.time()

                # print("return: ", '%.3f' % avg_return, "; loss: ", '%.3f' % loss, "time: ", '%.4f' % (cur_time - last_time))
                print('{}th update model spend time:  {:.4}, loss: {:.3f}, return: {:.3f}'.
                      format(update_episode_index, cur_time - last_time, loss, avg_return))
                # logger.log('{}th update model spend time:  {:.4}, loss: {:.3f}, avg_return: {:.3f}'.
                #       format(update_index, cur_time - last_time, loss, avg_return))

                data = {'Iteration': update_episode_index, 'loss': loss, 'return': avg_return}
                df = pd.DataFrame([data])
                df.to_csv(loss_return_file_path, mode='a', header=False, index=False)

                if is_visualize:
                    viz_window.line(X=np.array([episode_index]), Y=np.array([avg_return]),
                                    win='window{}'.format(0), update='append', opts=dict(title='return'))
                    viz_window.line(X=np.array([episode_index]), Y=np.array([avg_true_return]),
                                    win='window{}'.format(1), update='append', opts=dict(title='true return'))
                    viz_window.line(X=np.array([episode_index]), Y=np.array([loss]),
                                    win='window{}'.format(2), update='append', opts=dict(title='loss'))
                    viz_window.line(X=np.array([episode_index]), Y=np.array([p_loss]),
                                    win='window{}'.format(3), update='append', opts=dict(title='policy loss'))
                    viz_window.line(X=np.array([episode_index]), Y=np.array([v_loss]),
                                    win='window{}'.format(4), update='append', opts=dict(title='value loss'))
                    viz_window.line(X=np.array([episode_index]), Y=np.array([entropy_loss]),
                                    win='window{}'.format(5), update='append', opts=dict(title='entropy loss'))
                    viz_window.line(X=np.array([episode_index]), Y=np.array([explained_var]),
                                    win='window{}'.format(6), update='append', opts=dict(title='explained variance'))
                    viz_window.line(X=np.array([episode_index]), Y=np.array([clip_fraction]),
                                    win='window{}'.format(7), update='append', opts=dict(title='clip fraction'))
                    viz_window.line(X=np.array([episode_index]), Y=np.array([approx_kl]),
                                    win='window{}'.format(8), update='append', opts=dict(title='Approximating KL Divergence'))

            if episode_index % train_args.validation_episodes == 0:
                validation_episode_index += 1
                print('\nStart validating')
                last_time = time.time()
                mean_make_span = validate_ppo_model(validation_env, agent, validation_batch_size, train_args)
                cur_time = time.time()

                # print('validating time: ', '%.4f' % (cur_time - last_time), '\n')
                print('\n{}th validation spend time: {:.4f}, make span: {:.3f}\n'.format(validation_episode_index, cur_time - last_time, mean_make_span))
                # logger.log('\n{}th validation spend time: {:.4f}, make span: {:.3f}\n'.format(validation_index, cur_time - last_time, mean_make_span))

                data = {'Iteration': validation_episode_index, 'make span': mean_make_span}
                df = pd.DataFrame([data])
                df.to_csv(make_span_file_path, mode='a', header=False, index=False)

                if is_visualize:
                    viz_window.line(
                        X=np.array([episode_index]), Y=np.array([mean_make_span]),
                        win='window{}'.format(9), update='append', opts=dict(title='make span of validation instances'))

                if mean_make_span < best_make_span_mean:
                    best_make_span_mean = mean_make_span
                    if len(best_models) == max_num_of_save_model:
                        delete_file = best_models.popleft()
                        os.remove(delete_file)

                    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                    save_file = '{0}/save_best_{1}.pt'.format(save_path, str_time)
                    best_models.append(save_file)
                    torch.save(agent.policy.state_dict(), save_file)


# todo 小数
def validate_ppo_model(validation_env, agent, validation_batch_size, train_args):
    """
    Validate the policy during training, and the process is similar to test
    """
    agent.eval()
    agent.is_greedy = True

    print('There are {0} validation instances.'.format(validation_batch_size))  # validation set is also called development set
    state = validation_env.reset()

    if train_args.observer_norm == 'observer_norm_graph':
        # Trick: observer normalization one graph
        state.batch_graph = observe_norm_graph(state.batch_graph)
    elif train_args.observer_norm == 'observer_norm_running':
        # Trick: observer normalization running mean std
        state.batch_graph = observe_norm_running(state.batch_graph, update=False)

    done = torch.zeros(validation_batch_size, dtype=torch.bool)

    while not done.all():
        with torch.no_grad():
            actions, _, _, _, _, _ = agent.get_action_and_value(state, action_type=train_args.action_type, memory=None)
            next_state, _, done, _ = validation_env.step(actions, reward_type=train_args.reward_type, action_type=train_args.action_type)

            if train_args.observer_norm == 'observer_norm_graph':
                # Trick: observer normalization one graph
                next_state.batch_graph = observe_norm_graph(next_state.batch_graph)
            elif train_args.observer_norm == 'observer_norm_running':
                # Trick: observer normalization running mean std
                next_state.batch_graph = observe_norm_running(next_state.batch_graph, update=False)

            state = next_state

    agent.train()
    agent.is_greedy = False
    return torch.mean(validation_env.make_span_of_batch, dim=0).item()


if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float)
    else:
        torch.set_default_dtype(torch.float)

    args = parse_train_ppo_args()
    args.device = device

    # Trick: observer normalization one graph
    observe_norm_graph = ObserveNormalizationGraph()
    # Trick: observer normalization running mean std
    observe_norm_running = ObserveNormalizationRunningMeanStd(args.operation_in_dim, args.machine_in_dim, args.vehicle_in_dim,
                                                              args.o_to_m_edge_in_dim, args.o_to_v_edge_in_dim, clip=10.0)
    # Trick: reward normalization
    reward_norm_running = Normalization(shape=1, clip=10.0)
    # Trick: reward scaling
    reward_norm_scaling = RewardScaling(shape=1, gamma=args.gamma, clip=10.0)

    train_ppo(args)
