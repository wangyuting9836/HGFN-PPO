from pathlib import Path
from typing import Union, Type

ProcessingData = list[tuple[int, int]]
Arc = tuple[int, int]


def parse_job_line(line: list[int]) -> list[ProcessingData]:
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
