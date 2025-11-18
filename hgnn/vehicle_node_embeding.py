# _*_ coding: UTF-8 _*_

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv

from hgnn.hetero_gatv2_conv import HeteroGATv2Conv


class VehicleNodeEmbedding(nn.Module):
    def __init__(self, operation_in_dim, vehicle_in_dim, o_to_v_edge_in_dim=1, vehicle_out_dim=10, heads=4):
        super().__init__()
        self.m_gat_conv = HeteroGATv2Conv(in_channels=(operation_in_dim, vehicle_in_dim), out_channels=vehicle_out_dim, edge_dim=o_to_v_edge_in_dim,
                                          heads=heads, concat=False)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, train_mask_dict):
        o_to_v_mask = train_mask_dict[('operation', 'o_v', 'vehicle')]

        v_embedding = self.m_gat_conv(
            x_dict['operation'],
            x_dict['vehicle'],
            edge_index_dict[('operation', 'o_v', 'vehicle')][:, o_to_v_mask],
            edge_index_dict[('vehicle', 'self_loop', 'vehicle')],
            edge_attr_dict[('operation', 'o_v', 'vehicle')][o_to_v_mask, :]
        )

        return {"vehicle": v_embedding.sigmoid()}
