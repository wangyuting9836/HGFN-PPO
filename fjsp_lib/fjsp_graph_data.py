# _*_ coding: UTF-8 _*_

import numpy as np
import torch
import torch_scatter
from torch_geometric.data import HeteroData

from agent.normlization import Normalization
from fjsp_lib import FJSPInstance


def generate_graph_data(instance: FJSPInstance, operation_in_dim, machine_in_dim, vehicle_in_dim):
    # data = FJSPHeteroGraphData()
    data = HeteroData()
    num_all_jobs = instance.num_jobs
    num_all_operations = instance.num_operations

    data['job'].x = torch.zeros((num_all_jobs, 1), dtype=torch.float)
    data['operation'].x = torch.zeros((num_all_operations + 1, operation_in_dim), dtype=torch.float)
    data[('operation', 'self_loop', 'operation')].edge_index = torch.arange(1, num_all_operations + 1, dtype=torch.long).repeat(2, 1)

    num_machines = instance.num_machines
    data['machine'].x = torch.zeros((num_machines + 1, machine_in_dim), dtype=torch.float)
    data[('machine', 'self_loop', 'machine')].edge_index = torch.arange(1, num_machines + 1, dtype=torch.long).repeat(2, 1)

    num_vehicles = instance.num_vehicles
    data['vehicle'].x = torch.zeros((num_vehicles, vehicle_in_dim), dtype=torch.float)
    data[('vehicle', 'self_loop', 'vehicle')].edge_index = torch.arange(num_vehicles, dtype=torch.long).repeat(2, 1)

    o_o_edges = []
    o_j_edges = []

    o_m_edges = []
    o_m_edges_times = []

    o_v_edges = []
    o_v_edges_times = []

    process_info_all_jobs = instance.process_info_of_job
    operation_index = 0

    operation_index += 1
    for job in range(instance.num_jobs):
        num_operations = instance.num_operations_of_job[job]
        for op in range(num_operations):
            process_info = process_info_all_jobs[job][op]
            o_j_edges.append([operation_index, job])
            if op == 0:
                for machine, process_time in process_info.available_machine_process_info:
                    o_m_transport_time = instance.transportation_time_matrix[0][machine]
                    o_m_edges.append([operation_index, machine])
                    o_m_edges_times.append([o_m_transport_time, process_time])
                for vehicle in range(num_vehicles):
                    o_v_edges.append([operation_index, vehicle])
                    o_v_edges_times.append([0])
            else:
                o_o_edges.append([operation_index - 1, operation_index])
                pre_operation_process_info = process_info_all_jobs[job][op - 1]
                pre_machine_locations = [m for m, _ in pre_operation_process_info.available_machine_process_info]
                for machine, process_time in process_info.available_machine_process_info:
                    o_m_transport_time = np.average(instance.transportation_time_matrix[pre_machine_locations, machine])
                    o_m_edges.append([operation_index, machine])
                    o_m_edges_times.append([o_m_transport_time, process_time])
                for vehicle in range(num_vehicles):
                    o_v_transport_time = np.average(instance.transportation_time_matrix[0, pre_machine_locations])
                    o_v_edges.append([operation_index, vehicle])
                    o_v_edges_times.append([o_v_transport_time])

            operation_index += 1

    data[('operation', 'o_o', 'operation')].edge_index = torch.tensor(o_o_edges, dtype=torch.long).t().contiguous()
    data[('operation', 'o_j', 'job')].edge_index = torch.tensor(o_j_edges, dtype=torch.long).t().contiguous()

    data[('operation', 'o_m', 'machine')].edge_index = torch.tensor(o_m_edges, dtype=torch.long).t().contiguous()
    data[('operation', 'o_m', 'machine')].edge_attr = torch.tensor(o_m_edges_times, dtype=torch.float).contiguous()
    data[('operation', 'o_m', 'machine')].train_mask = torch.ones(len(o_m_edges), dtype=torch.bool)

    data[('operation', 'o_v', 'vehicle')].edge_index = torch.tensor(o_v_edges, dtype=torch.long).t().contiguous()
    data[('operation', 'o_v', 'vehicle')].edge_attr = torch.tensor(o_v_edges_times, dtype=torch.float).contiguous()
    data[('operation', 'o_v', 'vehicle')].train_mask = torch.ones(len(o_v_edges), dtype=torch.bool)

    o_m_edge = data.edge_index_dict[('operation', 'o_m', 'machine')]

    if o_m_edge.shape[0] == 0:
        data[('operation', 'o_m_action', 'machine')].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data[('operation', 'o_v_action', 'vehicle')].edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        o_m_action = o_m_edge.repeat(1, instance.num_vehicles)
        vehicle_action = torch.arange(0, instance.num_vehicles).repeat_interleave(o_m_edge.size(1)).unsqueeze(0)
        o_v_action = torch.cat([o_m_action[0].unsqueeze(0), vehicle_action], dim=0)
        data[('operation', 'o_m_action', 'machine')].edge_index = o_m_action
        data[('operation', 'o_v_action', 'vehicle')].edge_index = o_v_action
    return data


class ObserveNormalizationGraph:
    def __call__(self, batch_graph: HeteroData):
        for key, x in batch_graph.x_dict.items():
            node_batch = batch_graph.batch_dict[key]
            node_feature_batch_mean = torch_scatter.scatter_mean(x, node_batch, dim=0)
            node_feature_batch_std = torch_scatter.scatter_std(x, node_batch, dim=0)
            batch_graph[key].x = (x - node_feature_batch_mean[node_batch]) / (node_feature_batch_std[node_batch] + 1e-5)
            # normalized_data[key].x = (x - node_feature_batch_mean[node_batch]) / (node_feature_batch_std[node_batch] + 1e-5)
        for key, edge_attr in batch_graph.edge_attr_dict.items():
            edge_batch = batch_graph.batch_dict[key[0]][batch_graph.edge_index_dict[key][0]]
            edge_attr_batch_mean = torch_scatter.scatter_mean(edge_attr, edge_batch, dim=0)
            edge_attr_batch_std = torch_scatter.scatter_std(edge_attr, edge_batch, dim=0)
            batch_graph[key].edge_attr = (edge_attr - edge_attr_batch_mean[edge_batch]) / (edge_attr_batch_std[edge_batch] + 1e-5)
            # batch_graph[key].edge_attr = (edge_attr - edge_attr.mean(dim=0)) / (edge_attr.std(dim=0) + 1e-5)
            # normalized_data[key].edge_attr = (edge_attr - edge_attr_batch_mean[edge_batch]) / (edge_attr_batch_std[edge_batch] + 1e-5)
        return batch_graph


class ObserveNormalizationRunningMeanStd:
    def __init__(self, operation_in_dim, machine_in_dim, vehicle_in_dim, o_to_m_edge_in_dim, o_to_v_edge_in_dim, clip=10.0):
        self.operation_feature_norm = Normalization(shape=operation_in_dim, clip=clip)
        self.machine_feature_norm = Normalization(shape=machine_in_dim, clip=clip)
        self.vehicle_feature_norm = Normalization(shape=vehicle_in_dim, clip=clip)
        self.o_to_m_feature_norm = Normalization(shape=o_to_m_edge_in_dim, clip=clip)
        self.o_to_v_feature_norm = Normalization(shape=o_to_v_edge_in_dim, clip=clip)

    def __call__(self, batch_graph: HeteroData, update=True):
        batch_graph['operation'].x = self.operation_feature_norm(batch_graph['operation'].x, update=update)
        batch_graph['machine'].x = self.machine_feature_norm(batch_graph['machine'].x, update=update)
        batch_graph['vehicle'].x = self.vehicle_feature_norm(batch_graph['vehicle'].x, update=update)

        batch_graph[('operation', 'o_m', 'machine')].edge_attr = self.o_to_m_feature_norm(batch_graph[('operation', 'o_m', 'machine')].edge_attr, update=update)
        batch_graph[('operation', 'o_v', 'vehicle')].edge_attr = self.o_to_v_feature_norm(batch_graph[('operation', 'o_v', 'vehicle')].edge_attr, update=update)
        return batch_graph
