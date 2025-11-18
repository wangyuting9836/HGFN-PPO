class VehicleScheduleInfo:
    vehicle: int
    transport_job: int
    machine1: int
    machine2: int
    machine3: int
    time1: int
    time2: int
    time3: int
    time4: int

    def __init__(self, vehicle, transport_job, machine1, machine2, machine3, time1, time2, time3, time4):
        self.vehicle = vehicle
        self.transport_job = transport_job
        self.machine1 = machine1
        self.machine2 = machine2
        self.machine3 = machine3
        self.time1 = time1
        self.time2 = time2
        self.time3 = time3
        self.time4 = time4


class OperationScheduleInfo:
    job: int
    operation: int
    machine: int
    start_time: int
    complete_time: int

    def __init__(self, job, operation, machine, start_time, complete_time):
        self.job = job
        self.operation = operation
        self.machine = machine
        self.start_time = start_time
        self.complete_time = complete_time


class PartialSolution:
    operation_schedule_info_list: list[OperationScheduleInfo]
    vehicle_schedule_info_list: list[VehicleScheduleInfo]
    num_jobs: int
    num_machines: int
    num_vehicles: int

    def __init__(self, num_jobs, num_machines, num_vehicles):
        self.operation_schedule_info_list = []
        self.vehicle_schedule_info_list = []
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_vehicles = num_vehicles
        pass

    def append_operation_schedule(self, operation_schedule_info: OperationScheduleInfo):
        self.operation_schedule_info_list.append(operation_schedule_info)

    def append_vehicle_schedule(self, vehicle_schedule_info: VehicleScheduleInfo):
        self.vehicle_schedule_info_list.append(vehicle_schedule_info)

    def reset(self):
        self.operation_schedule_info_list.clear()
        self.vehicle_schedule_info_list.clear()
