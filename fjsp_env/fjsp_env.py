# _*_ coding: UTF-8 _*_
import copy
from typing import List, Union, Tuple

import numpy as np
import torch
import torch_scatter
from torch_geometric.data import HeteroData

from fjsp_lib import FJSPInstance, PartialSolution
from fjsp_lib import OperationScheduleInfo
from fjsp_lib.fjsp_partial_solution import VehicleScheduleInfo


class FJSPState:
    batch_size: int
    batch_graph: HeteroData

    next_schedule_operation_of_job: torch.LongTensor
    unfinished_mask_of_job: torch.BoolTensor

    idle_mask_of_vehicle: torch.BoolTensor
    transporting_job_of_vehicle: torch.LongTensor

    job_location: torch.BoolTensor
    vehicle_location: torch.BoolTensor

    def __init__(self, batch_size, batch_graph, next_schedule_operation_of_job, unfinished_mask_of_job, idle_mask_of_vehicle, job_location,
                 vehicle_location):
        self.batch_size = batch_size
        self.batch_graph = batch_graph

        self.next_schedule_operation_of_job = next_schedule_operation_of_job
        self.unfinished_mask_of_job = unfinished_mask_of_job

        self.job_location = job_location
        self.vehicle_location = vehicle_location

        self.idle_mask_of_vehicle = idle_mask_of_vehicle

    def get_available_edge_info_1(self):
        """
        version1: Consider the available AGVs
        """
        operation_batch = self.batch_graph.batch_dict['operation']

        available_operation = self.next_schedule_operation_of_job[self.unfinished_mask_of_job]
        available_vehicle = torch.nonzero(self.idle_mask_of_vehicle, as_tuple=True)[0]

        all_operation_actions = self.batch_graph.edge_index_dict[('operation', 'o_m_action', 'machine')][0]
        all_machine_actions = self.batch_graph.edge_index_dict[('operation', 'o_m_action', 'machine')][1]
        all_vehicle_actions = self.batch_graph.edge_index_dict[('operation', 'o_v_action', 'vehicle')][1]

        available_action_mask = torch.isin(all_operation_actions, available_operation) & torch.isin(all_vehicle_actions, available_vehicle)
        available_operation_actions = all_operation_actions[available_action_mask]
        available_machine_actions = all_machine_actions[available_action_mask]
        available_vehicle_actions = all_vehicle_actions[available_action_mask]

        available_batch_mask = torch_scatter.scatter(available_action_mask.int(), operation_batch[all_operation_actions], dim=0, dim_size=self.batch_size,
                                                     reduce='max').bool()

        return available_operation_actions, available_machine_actions, available_vehicle_actions, available_action_mask, available_batch_mask

    def get_available_edge_info_2(self):
        """
        version2: Consider all AGVs
        """
        operation_batch = self.batch_graph.batch_dict['operation']

        available_operation = self.next_schedule_operation_of_job[self.unfinished_mask_of_job]

        all_operation_actions = self.batch_graph.edge_index_dict[('operation', 'o_m_action', 'machine')][0]
        all_machine_actions = self.batch_graph.edge_index_dict[('operation', 'o_m_action', 'machine')][1]
        all_vehicle_actions = self.batch_graph.edge_index_dict[('operation', 'o_v_action', 'vehicle')][1]

        available_action_mask = torch.isin(all_operation_actions, available_operation)
        available_operation_actions = all_operation_actions[available_action_mask]
        available_machine_actions = all_machine_actions[available_action_mask]
        available_vehicle_actions = all_vehicle_actions[available_action_mask]

        available_batch_mask = torch_scatter.scatter(available_action_mask.int(), operation_batch[all_operation_actions], dim=0, dim_size=self.batch_size,
                                                     reduce='max').bool()

        return available_operation_actions, available_machine_actions, available_vehicle_actions, available_action_mask, available_batch_mask


class FJSPEnv:
    batch_graph: HeteroData
    batch_instance: Union[List[FJSPInstance], None]
    batch_size: int
    render_mode: bool

    num_of_all_jobs: int
    num_of_all_operations: int
    num_of_all_machines: int
    num_of_all_o_m_v: int

    # 下面绘制甘特图用
    partial_solution_of_batch: Union[List[PartialSolution], None]
    operation_index_to_j_o: List[Tuple[int, int]]
    machine_index_to_machine: List[int]
    vehicle_index_to_vehicle: List[int]

    state: FJSPState
    original_state: FJSPState

    cumulative_conjunctive_arc: torch.Tensor
    cumulative_conjunctive_arc_mask: torch.Tensor

    operation_belongs_to_job: torch.Tensor
    virtual_operation_mask: torch.Tensor
    pre_operation_index: torch.Tensor
    suc_operation_index: torch.Tensor

    num_operations_of_job: torch.Tensor
    first_operation_of_job: torch.Tensor
    end_operation_of_job: torch.Tensor
    batch_of_job: torch.Tensor
    next_schedule_operation_of_job: torch.Tensor
    unfinished_mask_of_job: torch.Tensor
    job_location: torch.Tensor

    work_time_of_machine: torch.Tensor

    idle_mask_of_vehicle: torch.Tensor
    transporting_job_of_vehicle: torch.Tensor
    work_time_of_vehicle: torch.Tensor
    vehicle_location: torch.Tensor

    # Diagonalised transport matrix
    transport_time: torch.Tensor

    o_v_all_edges: torch.Tensor
    o_v_all_edges_time: torch.Tensor

    time_of_batch: torch.Tensor
    make_span_of_batch: torch.Tensor
    partial_make_span_of_batch: torch.Tensor
    reward_of_batch: torch.Tensor
    done_of_batch: torch.Tensor
    terminal_of_batch: torch.Tensor

    def __init__(self, batch_graph, batch_instance, batch_size, render_mode=False):
        self.batch_graph = batch_graph
        self.batch_instance = batch_instance
        self.batch_size = batch_size
        self.render_mode = render_mode

        self.partial_solution_of_batch = [PartialSolution(instance.num_jobs, instance.num_machines, instance.num_vehicles) for instance in batch_instance]
        self.operation_index_to_j_o = []
        self.machine_index_to_machine = []
        self.vehicle_index_to_vehicle = []

        self.num_of_all_jobs = sum([instance.num_jobs + 1 for instance in self.batch_instance])  # Add a virtual job to each instance
        self.num_of_all_operations = sum([instance.num_operations + 1 for instance in self.batch_instance])  # Add a virtual operation to each instance
        self.num_of_all_machines = sum([instance.num_machines + 1 for instance in self.batch_instance])  # Add a virtual machine to each instance
        self.num_of_all_vehicles = sum([instance.num_vehicles for instance in self.batch_instance])
        self.num_of_all_o_m_v = sum([instance.num_o_m_v for instance in self.batch_instance])

        self.init_environment_by_instances()

        '''
        operation features, dynamic:
            0: Status, a binary value indicates whether O_{ij} has been scheduled(1)  or not(0) till step t
            1: Number of neighboring machines
            2: Processing time, if O_{ij} is scheduled p_{ijk}, otherwise  \\bar{p_{ij}} 
            3: Number of unscheduled operations in the job
            4: Job completion time
            5: Start time     
            6: Transport time. 
            7: Num of neighboring vehicles       
        '''
        # The virtual process is set as scheduled
        self.batch_graph.x_dict['operation'][self.virtual_operation_mask, 0] = 1

        self.batch_graph.x_dict['operation'][:, 1] = torch.bincount(
            self.batch_graph.edge_index_dict[('operation', 'o_m', 'machine')][0], minlength=self.num_of_all_operations)

        self.batch_graph.x_dict['operation'][:, 2] = torch_scatter.scatter(
            self.batch_graph.edge_attr_dict[('operation', 'o_m', 'machine')][:, 1],
            self.batch_graph.edge_index_dict[('operation', 'o_m', 'machine')][0],
            dim=0,
            dim_size=self.num_of_all_operations,
            reduce='mean').squeeze()

        self.batch_graph.x_dict['operation'][:, 3] = torch.gather(self.num_operations_of_job, 0, self.operation_belongs_to_job)

        # The average transportation time from each machine to all possible processing-machine locations of the predecessor operations
        t1 = torch_scatter.scatter(
            self.o_v_all_edges_time,
            self.o_v_all_edges[0],
            dim=0,
            dim_size=self.num_of_all_operations,
            reduce='mean').squeeze()

        # The average transportation time from all possible processing-machine locations of the predecessor operation
        # to all possible processing-machine locations of the current operation
        t2 = torch_scatter.scatter(
            self.batch_graph.edge_attr_dict[('operation', 'o_m', 'machine')][:, 0],
            self.batch_graph.edge_index_dict[('operation', 'o_m', 'machine')][0],
            dim=0,
            dim_size=self.num_of_all_operations,
            reduce='mean').squeeze()

        self.batch_graph.x_dict['operation'][:, 6] = t1 + t2

        self.batch_graph.x_dict['operation'][:, 7] = torch.bincount(
            self.batch_graph.edge_index_dict[('operation', 'o_v', 'vehicle')][0], minlength=self.num_of_all_operations)

        cumulative_time = self.batch_graph.x_dict['operation'][:, 6] + self.batch_graph.x_dict['operation'][:, 2]

        # Estimate the start time of the operations
        self.batch_graph.x_dict['operation'][:, 5] = torch_scatter.scatter(
            cumulative_time[self.cumulative_conjunctive_arc[0]],
            self.cumulative_conjunctive_arc[1], dim=0, dim_size=self.num_of_all_operations,
            reduce='sum') + self.batch_graph.x_dict['operation'][:, 6]

        job_complete_time = torch.gather(
            self.batch_graph.x_dict['operation'][:, 5] + self.batch_graph.x_dict['operation'][:, 2],
            0,
            self.end_operation_of_job)

        self.batch_graph.x_dict['operation'][:, 4] = torch.gather(job_complete_time, 0, self.operation_belongs_to_job)

        '''            
        machine features, dynamic:
            0: Number of neighboring operations
            1: Available time
            2: Utilization, ratio of the non idle time to the total production time of M_k till T(t) and is within the range [0,1]
        '''
        self.batch_graph.x_dict['machine'][:, 0] = torch.bincount(
            self.batch_graph.edge_index_dict[('operation', 'o_m', 'machine')][1], minlength=self.num_of_all_machines)

        '''
        AGV features, dynamic:
           0: Number of neighboring operations
           1: Available time
           2: Utilization
        '''
        self.batch_graph.x_dict['vehicle'][:, 0] = torch.bincount(
            self.batch_graph.edge_index_dict[('operation', 'o_v', 'vehicle')][1],
            minlength=self.num_of_all_vehicles)

        self.time_of_batch = torch.zeros(self.batch_size)

        self.make_span_of_batch = torch_scatter.scatter(self.batch_graph.x_dict['operation'][:, 4], self.batch_graph.batch_dict['operation'],
                                                        dim=0, dim_size=self.batch_size, reduce='max')
        self.partial_make_span_of_batch = torch.zeros(self.batch_size)

        self.reward_of_batch = torch.zeros(self.batch_size)

        self.terminal_of_batch = torch.zeros(self.batch_size, dtype=torch.bool)
        self.done_of_batch = torch.zeros(self.batch_size, dtype=torch.bool)

        self.original_state = FJSPState(self.batch_size, self.batch_graph, self.next_schedule_operation_of_job,
                                        self.unfinished_mask_of_job, self.idle_mask_of_vehicle,
                                        self.job_location, self.vehicle_location)

    def reset(self):
        # for partial_solution in self.partial_solution_of_batch:
        #     partial_solution.reset()
        self.cumulative_conjunctive_arc_mask[:] = True
        self.work_time_of_machine[:] = 0
        self.work_time_of_vehicle[:] = 0
        self.time_of_batch[:] = 0
        self.reward_of_batch[:] = 0
        self.done_of_batch[:] = False
        self.terminal_of_batch[:] = False
        self.state = copy.deepcopy(self.original_state)
        return copy.deepcopy(self.state)

    def step(self, actions: torch.Tensor, reward_type, action_type):
        full_o_m_edge = self.state.batch_graph.edge_index_dict[('operation', 'o_m', 'machine')]
        full_o_v_edge = self.state.batch_graph.edge_index_dict[('operation', 'o_v', 'vehicle')]

        operation_batch = self.state.batch_graph.batch_dict['operation']
        machine_batch = self.state.batch_graph.batch_dict['machine']
        vehicle_batch = self.state.batch_graph.batch_dict['vehicle']
        o_m_edge_batche = operation_batch[full_o_m_edge[0]]
        o_v_edge_batche = operation_batch[full_o_v_edge[0]]

        select_operation = actions[0]
        select_machine = actions[1]
        select_vehicle = actions[2]
        action_operation = select_operation[select_operation != - 1]
        action_machine = select_machine[select_machine != - 1]
        action_vehicle = select_vehicle[select_vehicle != - 1]

        select_job = torch.full((self.state.batch_size,), -1, dtype=torch.long)
        select_job[select_operation != - 1] = self.operation_belongs_to_job[action_operation]
        action_job = select_job[select_job != - 1]

        select_batch = torch.full((self.state.batch_size,), -1, dtype=torch.long)
        select_batch[select_operation != - 1] = operation_batch[action_operation]
        action_batch = select_batch[select_batch != -1]

        action_vehicle_old_location = self.state.vehicle_location[action_vehicle]
        action_job_old_location = self.state.job_location[action_job]

        pre_operation_of_select_operation = torch.full((self.state.batch_size,), -1, dtype=torch.long)
        pre_operation_of_select_operation[select_batch != -1] = self.pre_operation_index[action_operation]
        pre_operation_of_action_operation = pre_operation_of_select_operation[pre_operation_of_select_operation != -1]

        suc_operation_of_select_operation = torch.full((self.state.batch_size,), -1, dtype=torch.long)
        suc_operation_of_select_operation[select_batch != -1] = self.suc_operation_index[action_operation]
        suc_operation_of_action_operation = suc_operation_of_select_operation[suc_operation_of_select_operation != - 1]
        valid_suc_operation_mask = ~self.virtual_operation_mask[suc_operation_of_action_operation]

        # select_o_m_edge_mask = torch.isin(full_o_m_edge[0], action_operation) & torch.isin(full_o_m_edge[1], action_machine)
        select_o_m_edge_mask = (full_o_m_edge[0] == select_operation[o_m_edge_batche]) & (full_o_m_edge[1] == select_machine[o_m_edge_batche])
        select_o_m_edge_index = select_o_m_edge_mask.nonzero().view(-1)  # mask_to_index

        process_time = self.state.batch_graph.edge_attr_dict[('operation', 'o_m', 'machine')][select_o_m_edge_index][:, 1]

        select_o_v_edge_mask = (full_o_v_edge[0] == select_operation[o_v_edge_batche]) & (full_o_v_edge[1] == select_vehicle[o_v_edge_batche])
        select_o_v_edge_index = select_o_v_edge_mask.nonzero().view(-1)  # mask_to_index

        complete_time_of_pre_operation = (self.state.batch_graph.x_dict['operation'][pre_operation_of_action_operation, 5] +
                                          self.state.batch_graph.x_dict['operation'][pre_operation_of_action_operation, 2])
        # Determine whether the original position of the job is the same as that of the current processing machine
        is_not_on_same_machine = self.state.job_location[action_job] != action_machine

        # The transportation time from AGV's current position to the machine where the predecessor operation is located
        # 0 for the same machine
        transport_time_1 = (self.state.batch_graph.edge_attr_dict[('operation', 'o_v', 'vehicle')][select_o_v_edge_index].squeeze() *
                            is_not_on_same_machine)
        arrive_time_to_select_pre_operation = self.state.batch_graph.x_dict['vehicle'][action_vehicle, 1] + transport_time_1

        # The transportation time for a job to move from the machine of the previous operation to the selected machine
        # 0 for the same machine
        transport_time_2 = (self.state.batch_graph.edge_attr_dict[('operation', 'o_m', 'machine')][select_o_m_edge_index][:, 0] *
                            is_not_on_same_machine)

        o_m_arc_deleted = (full_o_m_edge[0] == select_operation[o_m_edge_batche]) & (full_o_m_edge[1] != select_machine[o_m_edge_batche])
        self.state.batch_graph.train_mask_dict[('operation', 'o_m', 'machine')][o_m_arc_deleted] = False

        o_v_arc_deleted = (full_o_v_edge[0] == select_operation[o_v_edge_batche]) & (full_o_v_edge[1] != select_vehicle[o_v_edge_batche])
        self.state.batch_graph.train_mask_dict[('operation', 'o_v', 'vehicle')][o_v_arc_deleted] = False

        self.state.next_schedule_operation_of_job[action_job] += 1
        self.state.unfinished_mask_of_job = torch.where(
            self.state.next_schedule_operation_of_job > self.end_operation_of_job,
            False,
            self.state.unfinished_mask_of_job)

        self.state.job_location[action_job] = action_machine

        # Update the working time of the selected machine
        self.work_time_of_machine[action_machine] += process_time

        # Update whether the selected AGV is idle
        self.state.idle_mask_of_vehicle[action_vehicle] = ~is_not_on_same_machine

        # Update the location of the selected AGV.
        # If the previous and current operations are processed on the same machine,
        # no transportation is required and the AGV's location remains unchanged.
        self.state.vehicle_location[action_vehicle] = torch.where(is_not_on_same_machine, action_machine, self.state.vehicle_location[action_vehicle])

        # Update the working time of the selected AGV, considering whether the machines are the same
        work_time_of_action_vehicle = torch.where(
            (arrive_time_to_select_pre_operation < complete_time_of_pre_operation) & is_not_on_same_machine,
            transport_time_1 + complete_time_of_pre_operation - arrive_time_to_select_pre_operation + transport_time_2,
            transport_time_1 + transport_time_2)
        self.work_time_of_vehicle[action_vehicle] += work_time_of_action_vehicle

        self.state.batch_graph.x_dict['operation'][action_operation, 0] = 1

        self.state.batch_graph.x_dict['operation'][action_operation, 1] = 1

        # Update the processing time of the selected operation, replacing the average value with the actual value
        self.state.batch_graph.x_dict['operation'][action_operation, 2] = process_time

        # operation_of_action_job = torch.isin(self.operation_belongs_to_job, action_job).nonzero().view(-1)
        operation_of_action_job = (self.operation_belongs_to_job == select_job[operation_batch]).nonzero().view(-1)
        self.state.batch_graph.x_dict['operation'][operation_of_action_job, 3] -= 1

        self.state.batch_graph.x_dict['vehicle'][action_vehicle, 1] = self.state.batch_graph.x_dict['vehicle'][action_vehicle, 1] + work_time_of_action_vehicle

        available_time_of_action_machine = self.state.batch_graph.x_dict['machine'][action_machine, 1]

        self.state.batch_graph.x_dict['operation'][action_operation, 5] = torch.where(
            (self.state.batch_graph.x_dict['vehicle'][action_vehicle, 1] > available_time_of_action_machine) & is_not_on_same_machine,
            self.state.batch_graph.x_dict['vehicle'][action_vehicle, 1],
            available_time_of_action_machine)

        # Update the transportation time of the selected operation, replacing the average value with the actual value
        self.state.batch_graph.x_dict['operation'][action_operation, 6] = work_time_of_action_vehicle  # transport_time_1 + transport_time_2

        # O-M edges related to the immediate successor operations of the selected operation
        o_m_edge_mask_of_suc_operation = full_o_m_edge[0] == suc_operation_of_select_operation[o_m_edge_batche]

        self.state.batch_graph.edge_attr_dict[('operation', 'o_m', 'machine')][o_m_edge_mask_of_suc_operation, 0] \
            = torch.sum(self.transport_time[action_machine][:, full_o_m_edge[1, o_m_edge_mask_of_suc_operation]], dim=0)

        # The average transportation time from each machine to the processing-machine location of the predecessor operation
        t1 = torch_scatter.scatter(
            torch.sum(self.transport_time[:, action_machine], dim=1),
            machine_batch,
            dim=0,
            dim_size=self.batch_size,
            reduce='mean')[action_batch][valid_suc_operation_mask]

        # The average transportation time from the processing-machine location of the selected operation
        # to all possible processing-machine locations of the successor operation.
        t2 = torch_scatter.scatter(
            self.state.batch_graph.edge_attr_dict[('operation', 'o_m', 'machine')][o_m_edge_mask_of_suc_operation, 0],
            full_o_m_edge[0, o_m_edge_mask_of_suc_operation],
            dim=0,
            dim_size=self.num_of_all_operations,
            reduce='mean')[suc_operation_of_action_operation[valid_suc_operation_mask]]

        # Update the average transportation time of the immediate successor operations of the selected operation,
        # since the processing machine for the current operation has now been determined
        self.state.batch_graph.x_dict['operation'][suc_operation_of_action_operation[valid_suc_operation_mask], 6] = t1 + t2

        # O–V edges associated with the immediate successor operations of the selected operation
        o_v_edge_mask_of_suc_operation = full_o_v_edge[0] == suc_operation_of_select_operation[o_v_edge_batche]

        self.state.batch_graph.edge_attr_dict[('operation', 'o_v', 'vehicle')][o_v_edge_mask_of_suc_operation, 0] \
            = torch.sum(self.transport_time[self.state.vehicle_location[full_o_v_edge[1, o_v_edge_mask_of_suc_operation]]][:, action_machine], dim=1)

        # Unscheduled operations
        un_scheduled_operation = torch.where(self.state.batch_graph.x_dict['operation'][:, 0] == 0)[0]
        # Unscheduled operations(excluding the immediate successor operations)
        un_scheduled_operation_without_suc = un_scheduled_operation[~torch.isin(un_scheduled_operation, suc_operation_of_action_operation)]

        # The predecessor operations of unscheduled operations(excluding the immediate successor operations)
        pre_operation_of_un_scheduled_operation = self.pre_operation_index[un_scheduled_operation_without_suc]

        # the predecessor operation of an unscheduled operation is also unscheduled
        # /the operation is the first operation of the job
        mask_pre_un_scheduled = (self.state.batch_graph.x_dict['operation'][pre_operation_of_un_scheduled_operation, 0] == 0) | (
            self.virtual_operation_mask[pre_operation_of_un_scheduled_operation])

        o_v_edge_mask_of_need_update = (torch.isin(full_o_v_edge[0], un_scheduled_operation_without_suc[mask_pre_un_scheduled])
                                        & (full_o_v_edge[1] == select_vehicle[o_v_edge_batche]))

        o_v_all_edge_mask = (torch.isin(self.o_v_all_edges[0], un_scheduled_operation_without_suc[mask_pre_un_scheduled]) &
                             torch.isin(self.o_v_all_edges[1], self.state.vehicle_location[action_vehicle]))

        # Update the transportation time on the O–V edges
        # that are jointly related to the selected AGV and the unscheduled operations (predecessors are also unscheduled),
        # excluding the immediate successor operations.
        self.state.batch_graph.edge_attr_dict[('operation', 'o_v', 'vehicle')][o_v_edge_mask_of_need_update] = \
            self.o_v_all_edges_time[o_v_all_edge_mask]

        o_v_edge_mask_of_need_update = torch.isin(full_o_v_edge[0], un_scheduled_operation_without_suc[~mask_pre_un_scheduled]) & (
                full_o_v_edge[1] == select_vehicle[o_v_edge_batche])

        mask_o_m_of_pre_operation = torch.isin(full_o_m_edge[0, self.state.batch_graph.train_mask_dict[('operation', 'o_m', 'machine')]],
                                               pre_operation_of_un_scheduled_operation[~mask_pre_un_scheduled])
        machine_process_pre_operation = full_o_m_edge[1, self.state.batch_graph.train_mask_dict[('operation', 'o_m', 'machine')]][mask_o_m_of_pre_operation]

        self.state.batch_graph.edge_attr_dict[('operation', 'o_v', 'vehicle')][o_v_edge_mask_of_need_update, 0] = \
            torch.sum(self.transport_time[action_machine][:, machine_process_pre_operation], dim=0)

        mask1 = self.operation_belongs_to_job[self.cumulative_conjunctive_arc[1]] == select_job[operation_batch[self.cumulative_conjunctive_arc[1]]]
        mask2 = self.cumulative_conjunctive_arc[0] == pre_operation_of_select_operation[operation_batch[self.cumulative_conjunctive_arc[0]]]
        self.cumulative_conjunctive_arc_mask[mask1 & mask2] = False

        scheduled_operation_mask = self.state.batch_graph.x_dict['operation'][:, 0] == torch.tensor(1)

        start_times = self.state.batch_graph.x_dict['operation'][:, 5] * scheduled_operation_mask

        complete_times = start_times + self.state.batch_graph.x_dict['operation'][:, 2] * scheduled_operation_mask

        cumulative_time = (self.state.batch_graph.x_dict['operation'][:, 6] + self.state.batch_graph.x_dict['operation'][:, 2]) * ~scheduled_operation_mask

        estimate_times = torch_scatter.scatter(
            complete_times[self.cumulative_conjunctive_arc[0, self.cumulative_conjunctive_arc_mask]] +
            cumulative_time[self.cumulative_conjunctive_arc[0, self.cumulative_conjunctive_arc_mask]],
            self.cumulative_conjunctive_arc[1, self.cumulative_conjunctive_arc_mask], dim=0, dim_size=self.num_of_all_operations,
            reduce='sum') + self.state.batch_graph.x_dict['operation'][:, 6] * ~scheduled_operation_mask

        self.state.batch_graph.x_dict['operation'][:, 5] = start_times + estimate_times

        job_complete_time = torch.gather(
            self.state.batch_graph.x_dict['operation'][:, 5] + self.state.batch_graph.x_dict['operation'][:, 2],
            0,
            self.end_operation_of_job)

        self.state.batch_graph.x_dict['operation'][:, 4] = torch.gather(job_complete_time, 0, self.operation_belongs_to_job)

        self.state.batch_graph.x_dict['operation'][action_operation, 7] = 1

        # self.state.batch_graph.x_dict['machine'][action_machine, 0] = self.state.batch_graph.x_dict['machine'][action_machine, 0] - 1
        o_m_mask = self.state.batch_graph.train_mask_dict[('operation', 'o_m', 'machine')]
        self.state.batch_graph.x_dict['machine'][:, 0] = torch.bincount(
            self.state.batch_graph.edge_index_dict[('operation', 'o_m', 'machine')][1][o_m_mask],
            minlength=self.num_of_all_machines)

        partial_make_span = torch.max(self.state.batch_graph.x_dict['machine'][action_machine, 1])

        self.state.batch_graph.x_dict['machine'][action_machine, 1] = self.state.batch_graph.x_dict['operation'][action_operation, 5] + process_time

        self.state.batch_graph.x_dict['machine'][action_machine, 2] = self.work_time_of_machine[action_machine] / (torch.maximum(
            self.state.batch_graph.x_dict['machine'][action_machine, 1],
            self.time_of_batch[machine_batch[action_machine]]) + 1e-9)

        # self.state.batch_graph.x_dict['vehicle'][action_vehicle, 0] = self.state.batch_graph.x_dict['vehicle'][action_vehicle, 0] - 1
        o_v_mask = self.state.batch_graph.train_mask_dict[('operation', 'o_v', 'vehicle')]
        self.state.batch_graph.x_dict['vehicle'][:, 0] = torch.bincount(
            self.state.batch_graph.edge_index_dict[('operation', 'o_v', 'vehicle')][1][o_v_mask],
            minlength=self.num_of_all_vehicles)

        self.state.batch_graph.x_dict['vehicle'][action_vehicle, 2] = self.work_time_of_vehicle[action_vehicle] / (torch.maximum(
            self.state.batch_graph.x_dict['vehicle'][action_vehicle, 1],
            self.time_of_batch[vehicle_batch[action_vehicle]]) + 1e-9)

        new_make_span = torch_scatter.scatter(self.state.batch_graph.x_dict['operation'][:, 4], operation_batch,
                                              dim=0, dim_size=self.batch_size, reduce='max')

        new_partial_make_span = torch_scatter.scatter(self.state.batch_graph.x_dict['machine'][:, 1], machine_batch,
                                                      dim=0, dim_size=self.batch_size, reduce='max')

        if reward_type == 'estimate_make_span':
            self.reward_of_batch = self.make_span_of_batch - new_make_span
        elif reward_type == 'increased_make_span':
            self.reward_of_batch = self.partial_make_span_of_batch - new_partial_make_span

        self.make_span_of_batch = new_make_span
        self.partial_make_span_of_batch = new_partial_make_span

        self.terminal_of_batch = copy.deepcopy(self.done_of_batch)
        self.done_of_batch = torch_scatter.scatter((~self.state.unfinished_mask_of_job).int(), self.batch_of_job, dim=0, dim_size=self.batch_size,
                                                   reduce='min').bool()
        # self.reward_of_batch = -new_make_span * self.done_of_batch

        if self.render_mode:
            self.update_partial_solution(select_operation, select_machine, select_vehicle, action_vehicle_old_location, action_job_old_location,
                                         work_time_of_action_vehicle, transport_time_1, transport_time_2, complete_time_of_pre_operation,
                                         arrive_time_to_select_pre_operation, is_not_on_same_machine, select_batch)

        is_transit = self.get_transit_flag(action_type)
        while torch.any(is_transit):
            self.transit_time(is_transit)
            is_transit = self.get_transit_flag(action_type)

        return copy.deepcopy(self.state), self.reward_of_batch, self.done_of_batch, self.terminal_of_batch

    def get_transit_flag(self, action_type):
        if action_type == 'available_vehicle_tuple':
            _, _, _, _, available_batch_mask = self.state.get_available_edge_info_1()
            return (~available_batch_mask) & (~self.done_of_batch)
        elif action_type == 'all_tuple':
            _, _, _, _, available_batch_mask = self.state.get_available_edge_info_2()
            return (~available_batch_mask) & (~self.done_of_batch)

    def transit_time(self, is_transit):
        available_time_of_vehicle = self.state.batch_graph.x_dict['vehicle'][:, 1]

        next_time = torch.full((self.batch_size,), torch.inf)
        torch_scatter.scatter(available_time_of_vehicle, self.state.batch_graph.batch_dict['vehicle'], dim=0, out=next_time,
                              dim_size=self.batch_size, reduce='min')

        self.time_of_batch[is_transit] = next_time[is_transit]

        self.state.idle_mask_of_vehicle[available_time_of_vehicle == self.time_of_batch[self.state.batch_graph.batch_dict['vehicle']]] = True

        self.state.batch_graph.x_dict['machine'][:, 2] = self.work_time_of_machine / (torch.maximum(
            self.state.batch_graph.x_dict['machine'][:, 1],
            self.time_of_batch[self.state.batch_graph.batch_dict['machine']]) + 1e-9)

        self.state.batch_graph.x_dict['vehicle'][:, 2] = self.work_time_of_vehicle / (torch.maximum(
            self.state.batch_graph.x_dict['vehicle'][:, 1],
            self.time_of_batch[self.state.batch_graph.batch_dict['vehicle']]) + 1e-9)

    def init_environment_by_instances(self):
        self.operation_belongs_to_job = torch.full((self.num_of_all_operations,), -1, dtype=torch.long)
        self.virtual_operation_mask = torch.full((self.num_of_all_operations,), False, dtype=torch.bool)
        self.pre_operation_index = torch.full((self.num_of_all_operations,), -1, dtype=torch.long)
        self.suc_operation_index = torch.full((self.num_of_all_operations,), -1, dtype=torch.long)

        self.num_operations_of_job = torch.zeros(self.num_of_all_jobs, dtype=torch.long)
        self.first_operation_of_job = torch.zeros(self.num_of_all_jobs, dtype=torch.long)
        self.end_operation_of_job = torch.zeros(self.num_of_all_jobs, dtype=torch.long)
        self.batch_of_job = torch.zeros(self.num_of_all_jobs, dtype=torch.long)
        self.next_schedule_operation_of_job = torch.zeros(self.num_of_all_jobs, dtype=torch.long)
        self.unfinished_mask_of_job = torch.ones(self.num_of_all_jobs, dtype=torch.bool)
        self.job_location = torch.zeros(self.num_of_all_jobs, dtype=torch.long)

        self.work_time_of_machine = torch.zeros(self.num_of_all_machines)

        self.idle_mask_of_vehicle = torch.ones(self.num_of_all_vehicles, dtype=torch.bool)
        self.transporting_job_of_vehicle = torch.full((self.num_of_all_vehicles,), -1, dtype=torch.long)
        self.work_time_of_vehicle = torch.zeros(self.num_of_all_vehicles)
        self.vehicle_location = torch.zeros(self.num_of_all_vehicles, dtype=torch.long)

        transport_time_tensor_list = []

        all_o_v = []
        all_o_v_times = []

        conjunctive_arcs = []
        job_index = 0
        operation_index = 0
        machine_index = 0
        vehicle_index = 0
        batch_index = 0
        o_m_v_index = 0
        for instance in self.batch_instance:
            virtual_operation_index = operation_index
            virtual_job_index = job_index
            virtual_machine_index = machine_index

            self.operation_belongs_to_job[virtual_operation_index] = virtual_job_index
            self.virtual_operation_mask[virtual_operation_index] = True
            self.batch_of_job[virtual_job_index] = batch_index
            self.unfinished_mask_of_job[virtual_job_index] = False

            self.job_location[job_index: job_index + instance.num_jobs + 1] = virtual_machine_index
            self.vehicle_location[vehicle_index: vehicle_index + instance.num_vehicles] = virtual_machine_index
            self.operation_index_to_j_o.append((0, 0))
            operation_index += 1
            job_index += 1

            for job in range(instance.num_jobs):

                num_ops_of_job = instance.num_operations_of_job[job]
                self.num_operations_of_job[job_index] = num_ops_of_job
                self.first_operation_of_job[job_index] = operation_index
                self.end_operation_of_job[job_index] = operation_index + num_ops_of_job - 1
                self.batch_of_job[job_index] = batch_index
                self.next_schedule_operation_of_job[job_index] = operation_index
                self.job_location[job_index] = virtual_machine_index
                conjunctive_arcs.extend([[virtual_operation_index, operation_index + suc_op] for suc_op in range(num_ops_of_job)])
                conjunctive_arcs.extend(
                    [[operation_index + pre_op, operation_index + suc_op] for pre_op in range(num_ops_of_job) for suc_op in range(pre_op + 1, num_ops_of_job)])

                for op in range(num_ops_of_job):
                    self.operation_index_to_j_o.append((job, op))
                    self.operation_belongs_to_job[operation_index] = job_index
                    if op == 0:
                        self.pre_operation_index[operation_index] = virtual_operation_index
                    else:
                        self.pre_operation_index[operation_index] = operation_index - 1
                    if op == num_ops_of_job - 1:
                        self.suc_operation_index[operation_index] = virtual_operation_index
                    else:
                        self.suc_operation_index[operation_index] = operation_index + 1

                    if op == 0:
                        for machine in range(instance.num_machines + 1):
                            all_o_v.append([operation_index, machine_index + machine])
                            all_o_v_times.append([instance.transportation_time_matrix[machine, 0]])
                    else:
                        pre_operation_process_info = instance.process_info_of_job[job][op - 1]
                        pre_machine_locations = [m for m, _ in pre_operation_process_info.available_machine_process_info]  # 前一个工序可能的位置
                        for machine in range(instance.num_machines + 1):
                            o_v_transport_time = np.average(instance.transportation_time_matrix[machine, pre_machine_locations])  # 平均运输时间
                            all_o_v.append([operation_index, machine_index + machine])
                            all_o_v_times.append([o_v_transport_time])

                    operation_index += 1
                job_index += 1
            transport_time_tensor_list.append(torch.tensor(instance.transportation_time_matrix, dtype=torch.float))
            self.machine_index_to_machine.extend([mac for mac in range(instance.num_machines + 1)])  # 绘制甘特图用，把在多个批中的机器索引转成原始的索引，包含虚拟机器
            self.vehicle_index_to_vehicle.extend([veh for veh in range(instance.num_vehicles)])
            machine_index += instance.num_machines + 1
            vehicle_index += instance.num_vehicles
            o_m_v_index += instance.num_o_m_v
            batch_index += 1

        if len(conjunctive_arcs) == 0:
            self.cumulative_conjunctive_arc = torch.zeros((2, 0), dtype=torch.long)
            self.cumulative_conjunctive_arc_mask = torch.ones(0, dtype=torch.bool)
        else:
            self.cumulative_conjunctive_arc = torch.tensor(conjunctive_arcs, dtype=torch.long).t().contiguous()
            self.cumulative_conjunctive_arc_mask = torch.ones(self.cumulative_conjunctive_arc.shape[1], dtype=torch.bool)

        self.transport_time = torch.block_diag(*transport_time_tensor_list)

        self.o_v_all_edges = torch.tensor(all_o_v, dtype=torch.long).t().contiguous()
        self.o_v_all_edges_time = torch.tensor(all_o_v_times, dtype=torch.float).contiguous()

    def update_partial_solution(self, operation_index, machine_index, vehicle_index, vehicle_old_location, vehicle_middle_location,
                                work_time_of_action_vehicle, transport_time1, transport_time2, complete_time_of_pre_operation,
                                arrive_time_to_select_pre_operation, is_not_on_same_machine, batch_index):
        i = 0
        for idx in range(len(batch_index)):
            b_idx = batch_index[idx]
            if b_idx == -1:
                continue
            o_idx = operation_index[idx]
            m_idx = machine_index[idx]
            v_idx = vehicle_index[idx]

            job, operation = self.operation_index_to_j_o[o_idx]
            machine = self.machine_index_to_machine[m_idx]
            vehicle = self.vehicle_index_to_vehicle[v_idx]

            start_time = self.state.batch_graph.x_dict['operation'][o_idx, 5].item()
            complete_time = start_time + self.state.batch_graph.x_dict['operation'][o_idx, 2].item()

            old_location = self.machine_index_to_machine[vehicle_old_location[i]]
            middle_location = self.machine_index_to_machine[vehicle_middle_location[i]]

            operation_schedule_info = OperationScheduleInfo(job, operation, machine, start_time, complete_time)
            self.partial_solution_of_batch[b_idx].append_operation_schedule(operation_schedule_info)

            if is_not_on_same_machine[i]:
                v_start_time = self.state.batch_graph.x_dict['vehicle'][v_idx, 1].item() - work_time_of_action_vehicle[i].item()
                time2 = v_start_time + transport_time1[i].item()
                if complete_time_of_pre_operation[i] > arrive_time_to_select_pre_operation[i]:
                    time3 = time2 + complete_time_of_pre_operation[i].item() - arrive_time_to_select_pre_operation[i].item()
                else:
                    time3 = time2
                time4 = time3 + transport_time2[i].item()
                vehicle_schedule_info = VehicleScheduleInfo(vehicle, job, old_location, middle_location, machine, v_start_time, time2, time3, time4)
                self.partial_solution_of_batch[b_idx].append_vehicle_schedule(vehicle_schedule_info)

            i = i + 1
