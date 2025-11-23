import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB
import pandas as pd
from pygantt_agv import *
from read_data import read_fjsp_data


def solve_fjsp_agv(num_jobs, num_machines, num_vehicles, p, operation_set, Delta, t_time_matrix):
    try:
        h = 0x0000ffff

        job_set = np.arange(0, num_jobs + 1)
        operation_set[0] = np.array([0, 1])

        # Create a new model
        model = gp.Model("FJSP_AGV")
        # model.setParam(GRB.Param.IntFeasTol, 1e-9)
        model.setParam(GRB.Param.Presolve, 2)
        model.setParam(GRB.Param.LogFile, "result.txt")
        model.setParam(GRB.Param.TimeLimit, 600)

        # Create variables
        x = model.addVars(
            [
                (i, j, k)
                for i in job_set[1:]
                for j in operation_set[i]
                for k in Delta[i, j]
            ],
            vtype=GRB.BINARY, name="x"
        )
        y = model.addVars(
            [
                (i, j, i1, j1, k)
                for i in job_set[1:]
                for i1 in job_set[1:]
                for j in operation_set[i]
                for j1 in operation_set[i1]
                for k in set(Delta[i, j]) & set(Delta[i1, j1])
            ],
            vtype=GRB.BINARY, name="y"
        )
        w = model.addVars(
            [
                (i, j, i1, j1)
                for i in job_set
                for j in operation_set[i]
                for i1 in job_set
                for j1 in operation_set[i1]
            ],
            vtype=GRB.BINARY, name="w"
        )
        c = model.addVars(
            [
                (i, j)
                for i in job_set[1:]
                for j in operation_set[i]
            ],
            vtype=GRB.INTEGER, name="c", lb=0
        )
        a = model.addVars(
            [
                (i, j)
                for i in job_set[1:]
                for j in operation_set[i]
            ],
            vtype=GRB.INTEGER, name="a", lb=0
        )
        c_max = model.addVar(vtype=GRB.INTEGER, name="c_max")

        # Set objective
        model.setObjective(c_max, GRB.MINIMIZE)

        # model.setObjectiveN(c_max, 0, priority=2, abstol=0, reltol=0, name="makespan")
        # model.setObjectiveN(gp.quicksum(c[i, j] for i in job_set[1:] for j in operation_set[i]), 1, priority=1, abstol=0, reltol=0,
        #                     name="TT")
        # model.setObjectiveN(gp.quicksum(a[i, j] for i in job_set[1:] for j in operation_set[i]), 2, priority=0, abstol=0, reltol=0,
        #                     name="AA")

        # 1
        model.addConstrs(gp.quicksum(x[i, j, k] for k in Delta[i, j]) == 1
                         for i in job_set[1:]
                         for j in operation_set[i])

        # 2
        model.addConstrs(y[i, j, i1, j1, k] + y[i1, j1, i, j, k] <= x[i, j, k]
                         for i in job_set[1:]
                         for i1 in job_set[1:]
                         for j in operation_set[i]
                         for j1 in operation_set[i1]
                         if i < i1 or (i == i1 and j < j1)
                         for k in np.intersect1d(Delta[i, j], Delta[i1, j1]))

        # 3
        model.addConstrs(y[i, j, i1, j1, k] + y[i1, j1, i, j, k] <= x[i1, j1, k]
                         for i in job_set[1:]
                         for i1 in job_set[1:]
                         for j in operation_set[i]
                         for j1 in operation_set[i1]
                         if i < i1 or (i == i1 and j < j1)
                         for k in np.intersect1d(Delta[i, j], Delta[i1, j1]))

        # 4
        model.addConstrs(y[i, j, i1, j1, k] + y[i1, j1, i, j, k]
                         >= x[i, j, k] + x[i1, j1, k] - 1
                         for i in job_set[1:]
                         for i1 in job_set[1:]
                         for j in operation_set[i]
                         for j1 in operation_set[i1]
                         if i < i1 or (i == i1 and j < j1)
                         for k in np.intersect1d(Delta[i, j], Delta[i1, j1]))

        # 5
        model.addConstrs(c[i, j] >= c[i, j - 1] + gp.quicksum(x[i, j, k] * p[i, j, k] for k in Delta[i, j])
                         for i in job_set[1:]
                         for j in operation_set[i][1:])

        # 6
        model.addConstrs(c[i1, j1] >= c[i, j] + p[i1, j1, k] + (y[i, j, i1, j1, k] - 1) * h
                         for i in job_set[1:]
                         for i1 in job_set[1:]
                         for j in operation_set[i]
                         for j1 in operation_set[i1]
                         if i != i1 or (i == i1 and j != j1)
                         for k in np.intersect1d(Delta[i, j], Delta[i1, j1]))

        # 7
        model.addConstrs(w[0, 0, i, j] ==
                         gp.quicksum(y[i, j - 1, i, j, k]
                                     for k in np.intersect1d(Delta[i, j - 1], Delta[i, j]))
                         for i in job_set[1:]
                         for j in operation_set[i][1:])

        # 8
        model.addConstrs(w[i, j, 0, 0] ==
                         gp.quicksum(y[i, j - 1, i, j, k]
                                     for k in np.intersect1d(Delta[i, j - 1], Delta[i, j]))
                         for i in job_set[1:]
                         for j in operation_set[i][1:])

        # 9
        model.addConstrs(w[0, 0, i, 1] == 0
                         for i in job_set[1:])

        # 10
        model.addConstrs(w[i, 1, 0, 0] == 0
                         for i in job_set[1:])

        # 11
        model.addConstrs(gp.quicksum(gp.quicksum(w[i, j, i1, j1] for j in operation_set[i]) for i in job_set if i != i1)
                         + gp.quicksum(w[i1, j, i1, j1] for j in operation_set[i1] if j < j1) == 1
                         for i1 in job_set[1:]
                         for j1 in operation_set[i1])

        # 12
        model.addConstrs(gp.quicksum(gp.quicksum(w[i, j, i1, j1] for j1 in operation_set[i1]) for i1 in job_set if i1 != i)
                         + gp.quicksum(w[i, j, i, j1] for j1 in operation_set[i] if j1 > j) == 1
                         for i in job_set[1:]
                         for j in operation_set[i])
        # 13
        model.addConstr(gp.quicksum(gp.quicksum(w[i, j, 0, 1] for j in operation_set[i]) for i in job_set[1:]) <= num_vehicles)

        # 14
        model.addConstr(gp.quicksum(gp.quicksum(w[0, 1, i1, j1] for j1 in operation_set[i1]) for i1 in job_set[1:]) <= num_vehicles)

        # 15
        model.addConstr(gp.quicksum(gp.quicksum(w[i, j, 0, 1] for j in operation_set[i]) for i in job_set[1:]) ==
                        gp.quicksum(gp.quicksum(w[0, 1, i1, j1] for j1 in operation_set[i1]) for i1 in job_set[1:]))

        # 16
        model.addConstrs(c[i, j] >= a[i, j] + gp.quicksum(x[i, j, k] * p[i, j, k] for k in Delta[i, j])
                         for i in job_set[1:]
                         for j in operation_set[i])

        # 17
        model.addConstrs(a[i, j] >= c[i, j - 1] + t_time_matrix[k1][k]
                         + (x[i, j, k] + x[i, j - 1, k1] - w[0, 0, i, j] - 2) * h
                         for i in job_set[1:]
                         for j in operation_set[i][1:]
                         for k in Delta[i, j]
                         for k1 in Delta[i, j - 1]
                         if k != k1)

        # 18
        model.addConstrs(a[i1, j1] >= a[i, j] + t_time_matrix[k][k2] + t_time_matrix[k2][k1]
                         + (x[i, j, k] + x[i1, j1, k1]
                            + x[i1, j1 - 1, k2] + w[i, j, i1, j1] - 4) * h
                         for i in job_set[1:]
                         for i1 in job_set[1:]
                         for j in operation_set[i]
                         for j1 in operation_set[i1][1:]
                         if i != i1 or (i == i1 and j < j1)
                         for k in Delta[i, j]
                         for k1 in Delta[i1, j1]
                         for k2 in Delta[i1, j1 - 1]
                         if k1 != k2)

        # 19
        model.addConstrs(a[i1, 1] >= a[i, j] + t_time_matrix[k][0] + t_time_matrix[0][k1]
                         + (x[i, j, k] + x[i1, 1, k1] + w[i, j, i1, 1] - 3) * h
                         for i in job_set[1:]
                         for i1 in job_set[1:]
                         if i != i1
                         for j in operation_set[i]
                         for k in Delta[i, j]
                         for k1 in Delta[i1, 1])

        # 20
        model.addConstrs(a[i1, j1] >= t_time_matrix[0][k2] + t_time_matrix[k2][k1]
                         + (x[i1, j1, k1] + x[i1, j1 - 1, k2] + w[0, 1, i1, j1] - 3) * h
                         for i1 in job_set[1:]
                         for j1 in operation_set[i1][1:]
                         for k1 in Delta[i1, j1]
                         for k2 in Delta[i1, j1 - 1]
                         if k1 != k2)

        # 21
        model.addConstrs(a[i1, 1] >= t_time_matrix[0][k1]
                         + (x[i1, 1, k1] + w[0, 1, i1, 1] - 2) * h
                         for i1 in job_set[1:]
                         for k1 in Delta[i1, 1])

        # 22
        model.addConstrs(c_max >= c[i, operation_set[i][len(operation_set[i]) - 1]]
                         for i in job_set[1:])

        # Optimize model
        model.optimize()
        print('Obj: %g' % model.ObjVal)
        return model

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError as e:
        print('Encountered an attribute error')


def show_solution(num_jobs, num_machines, num_vehicles, p, operation_set, Delta, t_time_matrix, model, gantt_image_filename):
    job_set = np.arange(0, num_jobs + 1)
    vehicle_set = np.arange(1, num_vehicles + 1)

    x = {
        (i, j, k): model.getVarByName(f'x[{i},{j},{k}]').X
        for i in job_set[1:]
        for j in operation_set[i]
        for k in Delta[i, j]
    }
    w = {
        (i, j, i1, j1): model.getVarByName(f'w[{i},{j},{i1},{j1}]').X
        for i in job_set
        for j in operation_set[i]
        for i1 in job_set
        for j1 in operation_set[i1]
    }

    a = {
        (i, j): model.getVarByName(f'a[{i},{j}]').X
        for i in job_set[1:]
        for j in operation_set[i]
    }
    c = {
        (i, j): model.getVarByName(f'c[{i},{j}]').X
        for i in job_set[1:]
        for j in operation_set[i]
    }

    job_operation_array = [(i, j) for i in job_set[1:] for j in operation_set[i]]

    result_job = []
    result_agv = []
    for k in np.arange(1, num_machines + 1):
        result_job.append(
            {"bar_type": "PlaceholderBar",
             "machine": k,
             "label": "",
             "text": "",
             "color_category": 0,
             "start": 0,
             "finish": 0,
             "departure": 0
             })

    for i, j in job_operation_array:
        for k in Delta[i, j]:
            if x[i, j, k] >= 0.9:
                result_job.append(
                    {"bar_type": "NormalBar",
                     "machine": k,
                     "label": "Job" + str(i),
                     "text": "$O_{" + str(i) + "," + str(j) + "}$",
                     "color_category": i,
                     "start": c[i, j] - p[i, j, k],
                     "finish": c[i, j],
                     "departure": c[i, j]
                     })

    r = 1
    for i, j in job_operation_array:
        if w[0, 1, i, j] >= 0.9:
            machine = 0
            if j == 1:
                machine1 = 0
            else:
                for k1 in Delta[i, j - 1]:
                    if x[i, j - 1, k1] > 0.9:
                        machine1 = k1
                        break
            for k in Delta[i, j]:
                if x[i, j, k] > 0.9:
                    machine2 = k
                    break

            start_time1 = a[i, j] - t_time_matrix[machine1][machine2]
            start_time2 = start_time1 - t_time_matrix[machine][machine1]

            if a[i, j] > start_time1:
                result_agv.append(
                    {"agv": r,
                     "text": "$O_{" + str(i) + "," + str(j) + "}$",
                     "color_category": i,
                     "start_m": machine1,
                     "end_m": machine2,
                     "start": start_time1,
                     "finish": a[i, j],
                     })
            if start_time1 > start_time2:
                result_agv.append(
                    {"agv": r,
                     "text": "",
                     "color_category": 0,
                     "start_m": machine,
                     "end_m": machine1,
                     "start": start_time2,
                     "finish": start_time1,
                     })

            ti = i
            tj = j
            while not w[ti, tj, 0, 1] >= 0.9:
                for i, j in job_operation_array:
                    if w[ti, tj, i, j] >= 0.9:
                        machine = machine2
                        if j == 1:
                            machine1 = 0
                        else:
                            for k1 in Delta[i, j - 1]:
                                if x[i, j - 1, k1] > 0.9:
                                    machine1 = k1
                                    break

                        for k in Delta[i, j]:
                            if x[i, j, k] > 0.9:
                                machine2 = k
                                break

                        start_time1 = a[i, j] - t_time_matrix[machine1][machine2]
                        start_time2 = start_time1 - t_time_matrix[machine][machine1]

                        if a[i, j] > start_time1:
                            result_agv.append(
                                {"agv": r,
                                 "text": "$O_{" + str(i) + "," + str(j) + "}$",
                                 "color_category": i,
                                 "start_m": machine1,
                                 "end_m": machine2,
                                 "start": start_time1,
                                 "finish": a[i, j],
                                 })
                        if start_time1 > start_time2:
                            result_agv.append(
                                {"agv": r,
                                 "text": "",
                                 "color_category": 0,
                                 "start_m": machine,
                                 "end_m": machine1,
                                 "start": start_time2,
                                 "finish": start_time1,
                                 })
                        ti = i
                        tj = j

            r = r + 1

    df_job = pd.DataFrame(result_job)
    df_agv = pd.DataFrame(result_agv)
    df_agv.sort_values(by=['agv', 'start', 'finish'], ascending=True, inplace=True, ignore_index=True)

    for r in vehicle_set:
        df_of_one_agv = df_agv[df_agv.agv == r]
        row_index = df_of_one_agv.index
        for i in range(len(row_index) - 1):
            if df_of_one_agv.start[row_index[i + 1]] > df_of_one_agv.finish[row_index[i]]:
                new_row = pd.DataFrame({"agv": [r],
                                        "text": [""],
                                        "color_category": [0],
                                        "start_m": [df_of_one_agv.end_m[row_index[i]]],
                                        "end_m": [df_of_one_agv.start_m[row_index[i + 1]]],
                                        "start": [df_of_one_agv.finish[row_index[i]]],
                                        "finish": [df_of_one_agv.start[row_index[i + 1]]],
                                        })
                df_agv = pd.concat([df_agv, new_row], ignore_index=True)

    max_finish = df_job.finish.max()
    # print(df)

    set_gantt_color(df_job, palette="Pastel1")
    set_gantt_color(df_agv, palette="Pastel1")

    fig, axes = plt.subplots(1, 1, figsize=(7.5, 2.3))
    gantt(data_job=df_job, data_agv=df_agv, max_finish=max_finish, show_title=False, show_y_lable=True, show_legend=False)
    plt.tight_layout()
    plt.show()
    # plt.savefig(gantt_image_filename, format='svg', dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    num_jobs, num_machines, p, operation_set, Delta, t_time_matrix = read_fjsp_data("toy_instance.fjs", "layout_4m.txt")
    num_vehicles = 2
    model = solve_fjsp_agv(num_jobs, num_machines, num_vehicles, p, operation_set, Delta, t_time_matrix)
    show_solution(num_jobs, num_machines, num_vehicles, p, operation_set, Delta, t_time_matrix, model, "gantt_image")
