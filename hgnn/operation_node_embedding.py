import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor


class GraphLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, bidirectional=True):
        super(GraphLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.path_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x_operation, edge_index_dict):
        o_j_edge_index = edge_index_dict[('operation', 'o_j', 'job')]
        src_nodes, dst_nodes = o_j_edge_index[0], o_j_edge_index[1]
        unique_jobs, length_of_jobs = torch.unique(dst_nodes, return_counts=True)
        max_seq_len = max(length_of_jobs)

        sequences = torch.split(x_operation[src_nodes], length_of_jobs.tolist())
        padded_sequences = pad_sequence(sequences, batch_first=True)
        packed_sequences = pack_padded_sequence(
            padded_sequences,
            length_of_jobs,
            batch_first=True,
            enforce_sorted=False
        )

        lstm_out, (hidden, _) = self.path_lstm(packed_sequences)
        unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        if self.bidirectional:
            forward_output = unpacked_out[:, :, :self.hidden_dim]
            backward_output = unpacked_out[:, :, self.hidden_dim:]
            unpacked_out = torch.cat([forward_output, backward_output], dim=-1)

        range_tensor = torch.arange(max_seq_len).unsqueeze(0)  # [1, max_seq_len]
        length_tensor = length_of_jobs.unsqueeze(1)  # [num_of_jobs, 1]
        mask = range_tensor < length_tensor

        node_embeddings = torch.zeros(x_operation.size(0), unpacked_out.size(-1))
        node_embeddings[src_nodes] = unpacked_out[mask]

        return node_embeddings


class OperationNodeEmbedding(MessagePassing):

    def __init__(self, operation_in_dim, machine_in_dim, vehicle_in_dim, hidden_dim=128, lstm_hidden_dim=64, operation_out_dim=10):
        super().__init__(aggr='add')

        self.mlp_theta1 = nn.Sequential(
            nn.Linear(operation_in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, operation_out_dim),
        )

        self.mlp_theta2 = nn.Sequential(
            nn.Linear(machine_in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, operation_out_dim),
        )

        self.mlp_theta3 = nn.Sequential(
            nn.Linear(vehicle_in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, operation_out_dim),
        )

        self.mlp_theta0 = nn.Sequential(
            nn.ELU(),
            nn.Linear(operation_out_dim * 3, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, operation_out_dim),
        )

        self.graph_lstm = GraphLSTM(input_dim=operation_out_dim, hidden_dim=lstm_hidden_dim, num_layers=2, bidirectional=True)

        self.final_proj = nn.Sequential(nn.Linear(2*lstm_hidden_dim, hidden_dim),
                                        nn.ELU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ELU(),
                                        nn.Linear(hidden_dim, operation_out_dim))

    def forward(self, x_dict, edge_index_dict, train_mask_dict):
        o_to_m_mask = train_mask_dict[('operation', 'o_m', 'machine')]
        o_to_v_mask = train_mask_dict[('operation', 'o_v', 'vehicle')]

        self.flow = 'source_to_target'
        x = self.mlp_theta1(x_dict['operation'])
        self_embedding = self.propagate(edge_index_dict[('operation', 'self_loop', 'operation')], x=x)

        self.flow = 'source_to_target'
        x_src = self.mlp_theta2(x_dict['machine'])
        x_dst = x_dict['operation']
        m_to_o_embedding = self.propagate(torch.flip(edge_index_dict[('operation', 'o_m', 'machine')][:, o_to_m_mask], dims=[0]), x=(x_src, x_dst))

        self.flow = 'source_to_target'
        x_agv = self.mlp_theta3(x_dict['vehicle'])
        x_dst = x_dict['operation']
        v_to_o_embedding = self.propagate(torch.flip(edge_index_dict[('operation', 'o_v', 'vehicle')][:, o_to_v_mask], dims=[0]), x=(x_agv, x_dst))

        node_mlp_embedding = self.mlp_theta0(torch.cat([self_embedding, m_to_o_embedding, v_to_o_embedding], dim=1))

        lstm_out = self.graph_lstm(node_mlp_embedding, edge_index_dict)

        out = self.final_proj(lstm_out)

        return {"operation": out}

    def message(self, x_j: Tensor):
        return x_j

    def edge_update(self) -> Tensor:
        pass

    def message_and_aggregate(self, edge_index: Adj) -> Tensor:
        pass
