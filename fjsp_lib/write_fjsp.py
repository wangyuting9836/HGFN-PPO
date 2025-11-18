from pathlib import Path
from typing import Union

from fjsp_lib import FJSPInstance


def write_fjsp(where: Union[Path, str], instance: FJSPInstance):
    lines = []

    num_eligible = sum([len(task.available_machine_process_info) for ops in instance.process_info_of_job for task in ops])
    flexibility = round(num_eligible / instance.num_operations, 1)

    metadata = f"{instance.num_jobs} {instance.num_machines} {flexibility}"
    lines.append(metadata)

    process_info_all_jobs = instance.process_info_of_job

    for job in range(instance.num_jobs):
        num_operations = instance.num_operations_of_job[job]
        line_data = [num_operations]
        for op in range(num_operations):
            process_info = process_info_all_jobs[job][op]
            line_data.append(process_info.num_available_machine)
            for machine, process_time in process_info.available_machine_process_info:
                line_data.extend([machine + 1, process_time])

        line = " ".join(str(num) for num in line_data)
        lines.append(line)

    formatted = "\n".join(lines)

    with open(where, "w") as fh:
        fh.write(formatted)
