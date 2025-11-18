# -*-coding:gbk-*-

import argparse


def parse_train_ppo_args():
    parser = argparse.ArgumentParser(description="Train")

    # Model Parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--operation_in_dim', type=int, default=8, help="Operation raw feature dimension")
    model_group.add_argument('--machine_in_dim', type=int, default=3, help="Machine raw feature dimension")
    model_group.add_argument('--vehicle_in_dim', type=int, default=3, help="Vehicle raw feature dimension")
    model_group.add_argument('--o_to_m_edge_in_dim', type=int, default=2, help="Operation to machine arc raw feature dimension")
    model_group.add_argument('--o_to_v_edge_in_dim', type=int, default=1, help="Operation to vehicle arc raw feature dimension")
    model_group.add_argument('--ope_embedding_hidden_dim', type=int, default=128, help="OperationNodeEmbedding MLP hidden layer dimension")
    model_group.add_argument('--lstm_hidden_dim', type=int, default=64, help="OperationNodeEmbedding LSTM hidden layer dimension")
    model_group.add_argument('--operation_out_dim', type=int, default=10, help="OperationNodeEmbedding output feature dimension")
    model_group.add_argument('--machine_out_dim', type=int, default=10, help="MachineNodeEmbeddingLayer output feature dimension")
    model_group.add_argument('--vehicle_out_dim', type=int, default=10, help="VehicleNodeEmbeddingLayer output feature dimension")
    model_group.add_argument('--heads', type=int, default=4, help="MachineNodeEmbeddingLayer GAT heads number")
    model_group.add_argument('--hgnn_num_layers', type=int, default=2, help="NodeEmbedding layers number")
    model_group.add_argument('--actor_hidden_layers', type=int, default=3, help="Actor MLP hidden layers number")
    model_group.add_argument('--critic_hidden_layers', type=int, default=3, help="Critic MLP hidden layers number")
    model_group.add_argument('--actor_hidden_dim', type=int, default=64, help="Actor MLP hidden layer dimension")
    model_group.add_argument('--critic_hidden_dim', type=int, default=64, help="Critic MLP hidden layer dimension")

    # Training Parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--learning_rate', type=float, default=0.0002, help="Learning rate")
    training_group.add_argument('--batch_size', type=int, default=20, help="Batch size")
    # training_group.add_argument('--validation_batch_size', type=int, default=100, help="Number of batches used for validation")
    training_group.add_argument('--num_epochs', type=int, default=20, help="Number of training epochs")
    training_group.add_argument('--policy_update_episodes', type=int, default=1, help="How many episodes to update the policy")
    training_group.add_argument('--validation_episodes', type=int, default=10, help="How many episodes to validate the model")
    training_group.add_argument('--gamma', type=float, default=0.99, help="gamma")
    training_group.add_argument('--gae_lambda', type=float, default=0.95, help="gae_lambda")
    training_group.add_argument('--K_epochs', type=int, default=3, help="K_epochs")
    training_group.add_argument('--clip_epsilon', type=float, default=0.2, help="clip_epsilon")
    training_group.add_argument('--A_coefficient', type=float, default=1.0, help="A_coefficient")
    training_group.add_argument('--vf_coefficient', type=float, default=0.5, help="vf_coefficient")
    training_group.add_argument('--entropy_coefficient', type=float, default=0.01, help="entropy_coefficient")
    training_group.add_argument('--times_of_batch_size', type=int, default=25, help="Batch size when updating the model")

    training_group.add_argument("--reward_type", type=str, default='increased_make_span', help="reward setting method",
                                choices=['estimate_make_span', 'increased_make_span'])

    training_group.add_argument("--action_type", type=str, default='available_vehicle_tuple', help="action setting method",
                                choices=['available_vehicle_tuple', 'all_tuple'])

    training_group.add_argument("--is_advantage_norm", type=bool, default=True, help="Trick 1: advantage normalization")
    training_group.add_argument("--observer_norm", type=str, default='observer_norm_graph', help="Trick 2: observer normalization",
                                choices=['observer_norm_graph', 'observer_norm_running'])
    training_group.add_argument("--reward_norm", type=str, default='reward_norm_running', help="Trick 3: reward normalization",
                                choices=['reward_norm_running', 'reward_norm_scaling', 'None'])
    training_group.add_argument("--is_anneal_lr", type=bool, default=True, help="Trick 4: annealing learning rate ")
    training_group.add_argument("--is_grad_clip", type=bool, default=True, help="Trick 5: gradient clip")
    training_group.add_argument('--max_grad_norm', type=float, default=0.5, help="the maximum norm for the gradient clipping")
    training_group.add_argument("--is_orthogonal_init", type=bool, default=True, help="Trick 6: orthogonal initialization")
    training_group.add_argument('--is_clip_v_loss', type=bool, default=True, help="Trick 7: value loss clipping")

    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--data_path', type=str, default='./data', help="Path to the dataset")
    data_group.add_argument('--train_data_size', type=int, default='1000', help="Training data size")

    return parser.parse_args()
