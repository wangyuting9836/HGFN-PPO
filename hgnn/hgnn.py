import torch.nn as nn
from hgnn.machine_node_embedding import MachineNodeEmbeddingLayer
from hgnn.vehicle_node_embeding import VehicleNodeEmbedding
from hgnn.operation_node_embedding import OperationNodeEmbedding

class HGNN(nn.Module):
    def __init__(self, operation_in_dim, machine_in_dim, vehicle_in_dim, o_to_m_edge_in_dim, o_to_v_edge_in_dim, hidden_dim, lstm_hidden_dim,
                 operation_out_dim, machine_out_dim, vehicle_out_dim, heads, num_layers):
        super(HGNN, self).__init__()

        self.operation_batch_norm_layer = nn.BatchNorm1d(operation_in_dim, eps=1e-5)
        self.machine_batch_norm_layer = nn.BatchNorm1d(machine_in_dim, eps=1e-5)
        self.vehicle_batch_norm_layer = nn.BatchNorm1d(vehicle_in_dim, eps=1e-5)
        self.process_by_edge_batch_norm_layer = nn.BatchNorm1d(o_to_m_edge_in_dim, eps=1e-5)
        self.transport_by_edge_batch_norm_layer = nn.BatchNorm1d(o_to_v_edge_in_dim, eps=1e-5)

        self.hgnn_net = nn.ModuleList([])
        layer = nn.ModuleList([
            MachineNodeEmbeddingLayer(operation_in_dim, machine_in_dim, o_to_m_edge_in_dim, machine_out_dim, heads),
            VehicleNodeEmbedding(operation_in_dim, vehicle_in_dim, o_to_v_edge_in_dim, vehicle_out_dim, heads),
            OperationNodeEmbedding(operation_in_dim, machine_out_dim, vehicle_out_dim, hidden_dim, lstm_hidden_dim, operation_out_dim)
        ])
        self.hgnn_net.append(layer)
        for _ in range(num_layers - 1):
            layer = nn.ModuleList([
                MachineNodeEmbeddingLayer(operation_out_dim, machine_out_dim, o_to_m_edge_in_dim, machine_out_dim, heads),
                VehicleNodeEmbedding(operation_out_dim, vehicle_out_dim, o_to_v_edge_in_dim, vehicle_out_dim, heads),
                OperationNodeEmbedding(operation_out_dim, machine_out_dim, vehicle_out_dim, hidden_dim, lstm_hidden_dim, operation_out_dim)
            ])
            self.hgnn_net.append(layer)

    def forward(self, hetero_data):
        x_dict, edge_index_dict, edge_attr_dict, train_mask_dict = hetero_data.x_dict, hetero_data.edge_index_dict, hetero_data.edge_attr_dict, hetero_data.train_mask_dict

        for layer in self.hgnn_net:

            machine_embedding = layer[0]
            out_dict = machine_embedding(x_dict, edge_index_dict, edge_attr_dict, train_mask_dict)
            x_dict["machine"] = out_dict["machine"]

            vehicle_embedding = layer[1]
            out_dict = vehicle_embedding(x_dict, edge_index_dict, edge_attr_dict, train_mask_dict)
            x_dict["vehicle"] = out_dict["vehicle"]

            operation_embedding = layer[2]
            out_dict = operation_embedding(x_dict, edge_index_dict, train_mask_dict)
            x_dict["operation"] = out_dict["operation"]

        return x_dict
