import os
import random
from pathlib import Path

from fjsp_lib import FJSPInstance, generate_graph_data
from fjsp_lib.write_fjsp import write_fjsp


def generate_train_data(train_data_size, train_instance_feature, operation_in_dim, machine_in_dim, vehicle_in_dim):
    # opes_per_job_min = int(num_machines * 0.8)
    # opes_per_job_max = int(num_machines * 1.2)
    # train_instances = []
    train_datas = []
    for instance_feather in train_instance_feature:
        num_jobs = instance_feather['num_jobs']
        num_machines = instance_feather['num_machines']
        num_vehicles = instance_feather['num_vehicles']
        opes_per_job_min = instance_feather['num_operations_per_job'][0]
        opes_per_job_max = instance_feather['num_operations_per_job'][1]
        for _ in range(train_data_size):
            num_operations_each_job = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            instance = FJSPInstance(num_jobs, num_machines, num_vehicles, num_operations_each_job)
            graph_data = generate_graph_data(instance, operation_in_dim, machine_in_dim, vehicle_in_dim)

            train_datas.append((graph_data, instance))

            # partial_schedule = generate_partial_schedule(instance)
            # partial_schedules.append(partial_schedule)

    return train_datas


def generate_validation_data(validation_data_size, num_jobs, num_machines):
    root_directory = Path(__file__).resolve().parents[1]

    opes_per_job_min = int(num_machines * 0.8)
    opes_per_job_max = int(num_machines * 1.2)
    dir_path = str(root_directory) + "/validation_data/{0}{1}/".format(str.zfill(str(num_jobs), 2), str.zfill(str(num_machines), 2))
    for i in range(validation_data_size):
        file_path = dir_path + "{}j_{}m_{}.fjs".format(str.zfill(str(num_jobs), 2), str.zfill(str(num_machines), 2), str.zfill(str(i + 1), 3))
        num_operations_each_job = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
        instance = FJSPInstance(num_jobs, num_machines, num_operations_each_job)
        write_fjsp(file_path, instance)


def read_data(dir_path, operation_in_dim, machine_in_dim, vehicle_in_dim):
    valid_data_files = os.listdir(dir_path)

    validation_datas = []
    for i in range(len(valid_data_files)):
        file_path = dir_path + valid_data_files[i]
        instance = FJSPInstance(file_path)

        graph_data = generate_graph_data(instance, operation_in_dim, machine_in_dim, vehicle_in_dim)
        validation_datas.append((graph_data, instance))

    return validation_datas
