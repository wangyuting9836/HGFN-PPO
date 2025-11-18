import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB
import pandas as pd
from pygantt_agv import *
from read_data import read_fjsp_data

try:

    h = 0x0000ffff

    num_jobs, num_machines, p, operation_set, Delta, t_time_matrix = read_fjsp_data("jobset1.fjs", "layout.dat")
    job_set = np.arange(0, num_jobs + 1)
    operation_set[0] = np.array([0, 1])
    num_vehicles = 1

    # Create a new model
    model = gp.Model("FJSP_AGV")
    # model.setParam(GRB.Param.IntFeasTol, 1e-9)
    # model.setParam(GRB.Param.TimeLimit, 60)

    # Create variables
    x = {}
    for i in job_set[1:]:
        for j in operation_set[i]:
            x[(i, j)] = model.addVars([i], [j], Delta[i, j], vtype=GRB.BINARY, name="x")
    y = {}
    for i in job_set[1:]:
        for i1 in job_set[1:]:
            for j in operation_set[i]:
                for j1 in operation_set[i1]:
                    y[(i, j, i1, j1)] = model.addVars([i], [j], [i1], [j1],
                                                      np.intersect1d(Delta[i, j], Delta[i1, j1]),
                                                      vtype=GRB.BINARY,
                                                      name="y")
    w = {}
    for i in job_set:
        for i1 in job_set:
            w[(i, i1)] = model.addVars([i], operation_set[i], [i1], operation_set[i1], vtype=GRB.BINARY, name="w")
    c = {}
    for i in job_set[1:]:
        c[i] = model.addVars([i], operation_set[i], vtype=GRB.INTEGER, name="c")
    a = {}
    for i in job_set[1:]:
        a[i] = model.addVars([i], operation_set[i], vtype=GRB.INTEGER, name="a")
    c_max = model.addVar(vtype=GRB.INTEGER, name="c_max")

    # Set objective
    model.setObjective(c_max, GRB.MINIMIZE)

    # model.setObjectiveN(c_max, 0, priority=2, abstol=0, reltol=0, name="makespan")
    # model.setObjectiveN(gp.quicksum(c[i][i, j] for i in job_set[1:] for j in operation_set[i]), 1, priority=1, abstol=0, reltol=0,
    #                     name="TT")
    # model.setObjectiveN(gp.quicksum(a[i][i, j] for i in job_set[1:] for j in operation_set[i]), 2, priority=0, abstol=0, reltol=0,
    #                     name="AA")

    # 1
    model.addConstrs(gp.quicksum(x[(i, j)][i, j, k] for k in Delta[i, j]) == 1
                     for i in job_set[1:]
                     for j in operation_set[i])

    # 2
    model.addConstrs(y[(i, j, i1, j1)][i, j, i1, j1, k] + y[(i1, j1, i, j)][i1, j1, i, j, k] <= x[(i, j)][i, j, k]
                     for i in job_set[1:]
                     for i1 in job_set[1:]
                     for j in operation_set[i]
                     for j1 in operation_set[i1]
                     if i < i1 or (i == i1 and j < j1)
                     for k in np.intersect1d(Delta[i, j], Delta[i1, j1]))

    # 3
    model.addConstrs(y[(i, j, i1, j1)][i, j, i1, j1, k] + y[(i1, j1, i, j)][i1, j1, i, j, k] <= x[(i1, j1)][i1, j1, k]
                     for i in job_set[1:]
                     for i1 in job_set[1:]
                     for j in operation_set[i]
                     for j1 in operation_set[i1]
                     if i < i1 or (i == i1 and j < j1)
                     for k in np.intersect1d(Delta[i, j], Delta[i1, j1]))

    # 4
    model.addConstrs(y[(i, j, i1, j1)][i, j, i1, j1, k] + y[(i1, j1, i, j)][i1, j1, i, j, k]
                     >= x[(i, j)][i, j, k] + x[(i1, j1)][i1, j1, k] - 1
                     for i in job_set[1:]
                     for i1 in job_set[1:]
                     for j in operation_set[i]
                     for j1 in operation_set[i1]
                     if i < i1 or (i == i1 and j < j1)
                     for k in np.intersect1d(Delta[i, j], Delta[i1, j1]))

    # 5
    model.addConstrs(c[i][i, j] >= c[i][i, j - 1] + gp.quicksum(x[(i, j)][i, j, k] * p[i][j][k] for k in Delta[i, j])
                     for i in job_set[1:]
                     for j in operation_set[i][1:])

    # 6
    model.addConstrs(c[i1][i1, j1] >= c[i][i, j] + p[i1][j1][k] + (y[(i, j, i1, j1)][i, j, i1, j1, k] - 1) * h
                     for i in job_set[1:]
                     for i1 in job_set[1:]
                     for j in operation_set[i]
                     for j1 in operation_set[i1]
                     if i != i1 or (i == i1 and j != j1)
                     for k in np.intersect1d(Delta[i, j], Delta[i1, j1]))

    # 7
    model.addConstrs(w[0, i][0, 0, i, j] ==
                     gp.quicksum(y[(i, j - 1, i, j)][i, j - 1, i, j, k]
                                 for k in np.intersect1d(Delta[i, j - 1], Delta[i, j]))
                     for i in job_set[1:]
                     for j in operation_set[i][1:])

    # 8
    model.addConstrs(w[i, 0][i, j, 0, 0] ==
                     gp.quicksum(y[(i, j - 1, i, j)][i, j - 1, i, j, k]
                                 for k in np.intersect1d(Delta[i, j - 1], Delta[i, j]))
                     for i in job_set[1:]
                     for j in operation_set[i][1:])

    # 9
    model.addConstrs(w[0, i][0, 0, i, 1] == 0
                     for i in job_set[1:])

    # 10
    model.addConstrs(w[i, 0][i, 1, 0, 0] == 0
                     for i in job_set[1:])

    # 11
    model.addConstrs(gp.quicksum(gp.quicksum(w[i, i1][i, j, i1, j1] for j in operation_set[i]) for i in job_set if i != i1)
                     + gp.quicksum(w[i1, i1][i1, j, i1, j1] for j in operation_set[i1] if j < j1) == 1
                     for i1 in job_set[1:]
                     for j1 in operation_set[i1])

    # 12
    model.addConstrs(gp.quicksum(gp.quicksum(w[i, i1][i, j, i1, j1] for j1 in operation_set[i1]) for i1 in job_set if i1 != i)
                     + gp.quicksum(w[i, i][i, j, i, j1] for j1 in operation_set[i] if j1 > j) == 1
                     for i in job_set[1:]
                     for j in operation_set[i])
    # 13
    model.addConstr(gp.quicksum(gp.quicksum(w[i, 0][i, j, 0, 1] for j in operation_set[i]) for i in job_set[1:]) <= num_vehicles)

    # 14
    model.addConstr(gp.quicksum(gp.quicksum(w[0, i1][0, 1, i1, j1] for j1 in operation_set[i1]) for i1 in job_set[1:]) <= num_vehicles)

    # 15
    model.addConstr(gp.quicksum(gp.quicksum(w[i, 0][i, j, 0, 1] for j in operation_set[i]) for i in job_set[1:]) ==
                    gp.quicksum(gp.quicksum(w[0, i1][0, 1, i1, j1] for j1 in operation_set[i1]) for i1 in job_set[1:]))

    # 16
    model.addConstrs(c[i][i, j] >= a[i][i, j] + gp.quicksum(x[(i, j)][i, j, k] * p[i][j][k] for k in Delta[i, j])
                     for i in job_set[1:]
                     for j in operation_set[i])

    # 17
    model.addConstrs(a[i][i, j] >= c[i][i, j - 1] + t_time_matrix[k1][k]
                     + (x[(i, j)][i, j, k] + x[(i, j - 1)][i, j - 1, k1] - w[0, i][0, 0, i, j] - 2) * h
                     for i in job_set[1:]
                     for j in operation_set[i][1:]
                     for k in Delta[i, j]
                     for k1 in Delta[i, j - 1]
                     if k != k1)

    # 18
    model.addConstrs(a[i1][i1, j1] >= a[i][i, j] + t_time_matrix[k][k2] + t_time_matrix[k2][k1]
                     + (x[(i, j)][i, j, k] + x[(i1, j1)][i1, j1, k1]
                        + x[(i1, j1 - 1)][i1, j1 - 1, k2] + w[(i, i1)][i, j, i1, j1] - 4) * h
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
    model.addConstrs(a[i1][i1, 1] >= a[i][i, j] + t_time_matrix[k][0] + t_time_matrix[0][k1]
                     + (x[(i, j)][i, j, k] + x[(i1, 1)][i1, 1, k1] + w[(i, i1)][i, j, i1, 1] - 3) * h
                     for i in job_set[1:]
                     for i1 in job_set[1:]
                     if i != i1
                     for j in operation_set[i]
                     for k in Delta[i, j]
                     for k1 in Delta[i1, 1])

    # 20
    model.addConstrs(a[i1][i1, j1] >= t_time_matrix[0][k2] + t_time_matrix[k2][k1]
                     + (x[(i1, j1)][i1, j1, k1] + x[(i1, j1 - 1)][i1, j1 - 1, k2] + w[(0, i1)][0, 1, i1, j1] - 3) * h
                     for i1 in job_set[1:]
                     for j1 in operation_set[i1][1:]
                     for k1 in Delta[i1, j1]
                     for k2 in Delta[i1, j1 - 1]
                     if k1 != k2)

    # 21
    model.addConstrs(a[i1][i1, 1] >= t_time_matrix[0][k1]
                     + (x[(i1, 1)][i1, 1, k1] + w[(0, i1)][0, 1, i1, 1] - 2) * h
                     for i1 in job_set[1:]
                     for k1 in Delta[i1, 1])

    # 22
    model.addConstrs(c_max >= c[i][i, operation_set[i][len(operation_set[i]) - 1]]
                     for i in job_set[1:])

    # Optimize model
    model.optimize()

    result_job = []
    result_agv = []
    for k in np.arange(1, num_machines + 1):
        result_job.append(
            {"bar_type": "PlaceholderBar",
             "factory": 1,
             "machine": k,
             "group": -1,
             "label": "",
             "text": "",
             "color_category": 0,
             "start": 0,
             "finish": 0,
             "departure": 0
             })

    for i in job_set[1:]:
        for j in operation_set[i]:
            for k in Delta[i, j]:
                if x[(i, j)][i, j, k].X >= 0.9:
                    result_job.append(
                        {"bar_type": "NormalBar",
                         "factory": 1,
                         "machine": k,
                         "group": -1,
                         "label": "Job" + str(i),
                         "text": "$O_{" + str(i) + "," + str(j) + "}$",
                         "color_category": i,
                         "start": c[i][i, j].X - p[i][j][k],
                         "finish": c[i][i, j].X,
                         "departure": c[i][i, j].X
                         })

    m = 1
    for i in job_set[1:]:
        for j in operation_set[i]:
            if w[(0, i)][0, 1, i, j].X >= 0.9:
                if j == 1:
                    for k in Delta[i, j]:
                        if x[(i, j)][i, j, k].X > 0.9:
                            mach = k
                            break
                    if a[i][i, j].X > a[i][i, j].X - sum(t_time_matrix[0][k] * x[(i, j)][i, j, k].X for k in Delta[i, j]):
                        result_agv.append(
                            {"factory": 1,
                             "agv": m,
                             "group": -1,
                             "text": "$O_{" + str(i) + "," + str(j) + "}$",
                             "color_category": i,
                             "start_m": 0,
                             "end_m": mach,
                             "start": a[i][i, j].X - sum(t_time_matrix[0][k] * x[(i, j)][i, j, k].X for k in Delta[i, j]),
                             "finish": a[i][i, j].X,
                             })

                else:
                    starttime1 = a[i][i, j].X - \
                                 sum(t_time_matrix[k1][k] * x[(i, j - 1)][i, j - 1, k1].X * x[(i, j)][i, j, k].X
                                     for k1 in Delta[i, j - 1]
                                     for k in Delta[i, j])
                    starttime2 = starttime1 - \
                                 sum(t_time_matrix[0][k1] * x[(i, j - 1)][i, j - 1, k1].X for k1 in Delta[i, j - 1])
                    for k1 in Delta[i, j - 1]:
                        if x[(i, j - 1)][i, j - 1, k1].X > 0.9:
                            mach1 = k1
                            break
                    for k in Delta[i, j]:
                        if x[(i, j)][i, j, k].X > 0.9:
                            mach = k
                            break
                    if a[i][i, j].X > starttime1:
                        result_agv.append(
                            {"factory": 1,
                             "agv": m,
                             "group": -1,
                             "text": "$O_{" + str(i) + "," + str(j) + "}$",
                             "color_category": i,
                             "start_m": mach1,
                             "end_m": mach,
                             "start": starttime1,
                             "finish": a[i][i, j].X,
                             })
                    if starttime1 > starttime2:
                        result_agv.append(
                            {"factory": 1,
                             "agv": m,
                             "group": -1,
                             "text": '',
                             "color_category": 0,
                             "start_m": 0,
                             "end_m": mach1,
                             "start": starttime2,
                             "finish": starttime1,
                             })

                ti1 = i
                tj1 = j
                isEnd = False
                while True:
                    if isEnd is True:
                        break
                    for i2 in job_set:
                        if isEnd is True:
                            break
                        for j2 in operation_set[i2]:
                            if not (i2 != ti1 or (i2 == ti1 and j2 != tj1)):
                                continue
                            if w[(ti1, i2)][ti1, tj1, i2, j2].X >= 0.9:
                                if i2 == 0 and j2 == 1:
                                    isEnd = True
                                    break
                                else:
                                    if j2 == 1:
                                        starttime1 = a[i2][i2, j2].X - \
                                                     sum(
                                                         t_time_matrix[0][k2] * x[(i2, j2)][i2, j2, k2].X
                                                         for k2 in Delta[i2, j2]
                                                     )
                                        starttime2 = starttime1 - \
                                                     sum(
                                                         t_time_matrix[k][0] * x[(ti1, tj1)][ti1, tj1, k].X
                                                         for k in Delta[ti1, tj1]
                                                     )

                                        mach1 = 0
                                        for k2 in Delta[i2, j2]:
                                            if x[(i2, j2)][i2, j2, k2].X > 0.9:
                                                mach2 = k2
                                                break
                                        for k in Delta[ti1, tj1]:
                                            if x[(ti1, tj1)][ti1, tj1, k].X > 0.9:
                                                mach = k
                                                break
                                    else:
                                        starttime1 = a[i2][i2, j2].X - \
                                                     sum(
                                                         t_time_matrix[k1][k2] * x[(i2, j2 - 1)][i2, j2 - 1, k1].X *
                                                         x[(i2, j2)][
                                                             i2, j2, k2].X
                                                         for k1 in Delta[i2, j2 - 1]
                                                         for k2 in Delta[i2, j2]
                                                     )
                                        starttime2 = starttime1 - \
                                                     sum(
                                                         t_time_matrix[k][k1] * x[(ti1, tj1)][ti1, tj1, k].X * x[(i2, j2 - 1)][
                                                             i2, j2 - 1, k1].X
                                                         for k in Delta[ti1, tj1]
                                                         for k1 in Delta[i2, j2 - 1]
                                                     )
                                        for k1 in Delta[i2, j2 - 1]:
                                            if x[(i2, j2 - 1)][i2, j2 - 1, k1].X > 0.9:
                                                mach1 = k1
                                                break
                                        for k2 in Delta[i2, j2]:
                                            if x[(i2, j2)][i2, j2, k2].X > 0.9:
                                                mach2 = k2
                                                break
                                        for k in Delta[ti1, tj1]:
                                            if x[(ti1, tj1)][ti1, tj1, k].X > 0.9:
                                                mach = k
                                                break
                                    if a[i2][i2, j2].X > starttime1:
                                        result_agv.append(
                                            {"factory": 1,
                                             "agv": m,
                                             "group": -1,
                                             "text": "$O_{" + str(i2) + "," + str(j2) + "}$",
                                             "color_category": i2,
                                             "start_m": mach1,
                                             "end_m": mach2,
                                             "start": starttime1,
                                             "finish": a[i2][i2, j2].X,
                                             })
                                    if starttime1 > starttime2:
                                        result_agv.append(
                                            {"factory": 1,
                                             "agv": m,
                                             "group": -1,
                                             "text": "",
                                             "color_category": 0,
                                             "start_m": mach,
                                             "end_m": mach1,
                                             "start": starttime2,
                                             "finish": starttime1,
                                             })
                                ti1 = i2
                                tj1 = j2

                m = m + 1

    df_job = pd.DataFrame(result_job)
    df_agv = pd.DataFrame(result_agv)
    df_agv.sort_values(by=['agv', 'start', 'finish'], ascending=True, inplace=True, ignore_index=True)

    for m in np.arange(1, num_vehicles + 1):
        df_of_one_agv = df_agv[df_agv.agv == m]
        row_index = df_of_one_agv.index
        for i in range(len(row_index) - 1):
            if df_of_one_agv.start[row_index[i + 1]] > df_of_one_agv.finish[row_index[i]]:
                new_row = pd.DataFrame({
                    "factory": [1],
                    "agv": [m],
                    "group": [-1],
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
    gantt(data_job=df_job, data_agv=df_agv, max_finish=max_finish, show_title=False, show_y_lable=True)
    plt.tight_layout()
    plt.show()
    # plt.savefig("agv_gantt_1.png", format='png', dpi=600)
    for v in model.getVars():
        print('%s %g' % (v.VarName, v.X))

    print('Obj: %g' % model.ObjVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError as e:
    print('Encountered an attribute error')
