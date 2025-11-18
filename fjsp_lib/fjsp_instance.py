import random
from pathlib import Path
from typing import List

import numpy as np

from .read_fjsp import file2lines, parse_job_line


class OperationInfo:
    node_index: int
    available_machine_process_info: List[tuple[int, int]]
    avg_process_time: float
    num_available_machine: int

    def __init__(self, node_index, process_info, avg_processing_time):
        self.node_index = node_index
        self.available_machine_process_info = process_info
        self.avg_process_time = avg_processing_time
        self.num_available_machine = len(self.available_machine_process_info)


class MachineInfo:
    node_index: int
    processed_operation_info: List[tuple[int, int]]

    def __init__(self, node_index):
        self.node_index = node_index
        self.processed_operation_info = []


class FJSPInstance:
    num_jobs: int
    num_operations: int
    num_machines: int
    num_vehicles: int
    num_operations_of_job: list[int]
    process_info_of_job: list[list[OperationInfo]]
    process_info_of_machine: list[MachineInfo]
    transportation_time_matrix: np.array  # ����ʱ�����
    num_o_m_v: int  # O-M����

    def __init__(self, *args, **kwargs):
        self.num_jobs = 0
        self.num_machines = 0
        self.num_vehicles = 0
        self.num_operations_of_job = []
        self.process_info_of_job = []
        self.process_info_of_machine = []
        self.transportation_time_matrix = np.ndarray([])
        self.num_o_m_v = 0
        if len(args) == 4:
            self.generate_from_random_data(args[0], args[1], args[2], args[3])
        elif len(args) == 1:
            self.num_vehicles = 10
            self.read_from_file(args[0])

        self.num_operations = sum(self.num_operations_of_job)
        self.read_layout_file()

    def generate_from_random_data(self, num_jobs, num_machines, num_vehicles, num_operations_each_job):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_vehicles = num_vehicles
        self.num_operations_of_job = num_operations_each_job

        process_time_min = 1
        process_time_max = 20
        process_time_dev = 0.2

        for machine in range(1, self.num_machines + 1):
            self.process_info_of_machine.append(MachineInfo(machine))

        operation_index = 1
        for j in range(self.num_jobs):
            self.process_info_of_job.append([])
            for op in range(self.num_operations_of_job[j]):
                process_time_mean = random.randint(process_time_min, process_time_max)
                low_bound = max(process_time_min, round(process_time_mean * (1 - process_time_dev)))
                high_bound = min(process_time_max, round(process_time_mean * (1 + process_time_dev)))

                available_machines = sorted(random.sample(range(1, self.num_machines + 1), random.randint(1, self.num_machines)))

                process_info = [(machine, random.randint(low_bound, high_bound)) for machine in available_machines]

                self.num_o_m_v += len(process_info) * self.num_vehicles

                avg_processing_time = sum([process_time for _, process_time in process_info]) / len(process_info)

                for machine, process_time in process_info:
                    self.process_info_of_machine[machine - 1].processed_operation_info.append((operation_index, process_time))

                self.process_info_of_job[j].append(OperationInfo(operation_index, process_info, avg_processing_time))

                operation_index += 1

    def read_from_file(self, loc: Path):
        lines = file2lines(loc)
        jobs_info = [parse_job_line(line) for line in lines[1:]]

        self.num_jobs, self.num_machines = lines[0][0], lines[0][1]
        self.num_operations_of_job = [len(operations) for operations in jobs_info]

        for machine in range(1, self.num_machines + 1):
            self.process_info_of_machine.append(MachineInfo(machine))

        operation_index = 1
        for job in range(self.num_jobs):
            self.process_info_of_job.append([])
            for op in range(self.num_operations_of_job[job]):
                process_info = jobs_info[job][op]

                self.num_o_m_v += len(process_info) * self.num_vehicles

                avg_processing_time = sum([process_time for _, process_time in process_info]) / len(process_info)

                for machine, process_time in process_info:
                    self.process_info_of_machine[machine - 1].processed_operation_info.append((operation_index, process_time))

                self.process_info_of_job[job].append(OperationInfo(operation_index, process_info, avg_processing_time))
                operation_index += 1

    def read_layout_file(self):
        root_directory = Path(__file__).resolve().parents[1]
        dir_path = str(root_directory) + "/FJSP-layouts/layout_{0}m.txt".format(self.num_machines)
        matrix_data = []
        with open(dir_path, 'r') as file:
            for line in file:
                if line.strip():
                    row = [float(num) for num in line.split()]
                    matrix_data.append(row)

        self.transportation_time_matrix = np.array(matrix_data)
