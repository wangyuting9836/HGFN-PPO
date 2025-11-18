from pathlib import Path
from typing import Union, Tuple

import numpy as np


def parse_job_line(line: list[int]) -> list[list[Tuple[int, int]]]:
    num_operations = line[0]
    operations = []
    idx = 1

    for _ in range(num_operations):
        num_pairs = int(line[idx]) * 2
        machines = line[idx + 1: idx + 1 + num_pairs: 2]
        durations = line[idx + 2: idx + 2 + num_pairs: 2]
        operations.append([(m, d) for m, d in zip(machines, durations)])

        idx += 1 + num_pairs

    return operations


def file2lines(loc: Union[Path, str]) -> list[list[int]]:
    with open(loc, "r") as fh:
        lines = [line for line in fh.readlines() if line.strip()]

    def parse_num(word: str):
        return int(word) if "." not in word else int(float(word))

    return [[parse_num(x) for x in line.split()] for line in lines]


def read_fjsp_data(process_file, transport_file):

    lines = file2lines(process_file)
    jobs_info = [parse_job_line(line) for line in lines[1:]]

    num_jobs, num_machines = lines[0][0], lines[0][1]
    num_operations_of_job = [len(operations) for operations in jobs_info]

    p = {}  # Processing time of O_{i,j} on machine k.
    operation_set = {}  # Set of operation operations of job i.
    Delta = {}  # Set of eligible machines for O_{ij}.

    for i in np.arange(1, num_jobs + 1):
        p[i] = {}
        operation_set[i] = np.arange(1, num_operations_of_job[i - 1] + 1)
        for j in operation_set[i]:
            p[i][j] = {}
            Delta[i, j] = []
            process_info = jobs_info[i - 1][j - 1]
            for k, process_time in process_info:
                p[i][j][k] = process_time
                Delta[i, j].append(k)

    matrix_data = []
    with open(transport_file, 'r') as file:
        for line in file:
            if line.strip():
                row = [float(num) for num in line.split()]
                matrix_data.append(row)

    t_time_matrix = np.array(matrix_data)

    return num_jobs, num_machines, p, operation_set, Delta, t_time_matrix
