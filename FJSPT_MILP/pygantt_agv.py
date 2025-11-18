# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import numpy as np

from typing import List, Tuple, Dict


def table_find_col(table: object, value: str) -> List[str]:
    value_col_list = []
    for col in table.columns:
        if value in table[col].values.astype(str):
            value_col_list.append(col)
    return value_col_list


def table_find_row(table: object, col_list: List[str], value: str) -> List[Tuple[str]]:
    col_row_list = []
    for col in col_list:
        row_list = table.query(f'{col} =="{value}"').index
        for row in row_list:
            col_row_list.append((col, row))
    return col_row_list


def table_find_pos(table: object, value_list: List[str]) -> Dict[List[str], List[Tuple[str]]]:
    position_list = []
    for value in value_list:
        # col_list 一个值可能被很多列包含
        col_list = table_find_col(table, value)
        # row_list 在每一个包含value的列中搜索对应的row
        col_row_list = table_find_row(table, col_list, value)
        position_list.append(col_row_list)
    position_dict = dict(zip(value_list, position_list))
    return position_dict


def set_gantt_color(data, palette=None, **kwargs):
    # data.insert(0, "color", None)
    color_category = data[data.color_category > 0].drop_duplicates(subset=['color_category'])[
        'color_category'].sort_values()
    color_count = color_category.count()
    current_palette = sns.color_palette(palette, n_colors=color_count)

    # current_palette = plt.cm.get_cmap('Pastel1', len(job_array))

    color_dict = dict((c, p) for c, p in zip(color_category, current_palette))

    # color_dict = {1: (0.6313725490196078, 0.788235294117647, 0.9568627450980393),
    #               2: (0.5529411764705883, 0.8980392156862745, 0.6313725490196078),
    #               3: (1.0, 0.6235294117647059, 0.6078431372549019),
    #               4: (0.8156862745098039, 0.7333333333333333, 1.0), 5: (1.0, 0.996078431372549, 0.6392156862745098)}

    colors = []
    for i in data.index:
        if data.color_category[i] > 0:
            # data.loc[i, "color"] = [frozenset(color_dict[data.color_category[i]])]
            colors.append(color_dict[data.color_category[i]])
        if data.color_category[i] <= 0:
            colors.append(None)
    data.insert(0, "color", colors)
    return data


def gantt(data_job, data_agv, max_finish=None, show_title=False, show_y_label=True, show_legend=True, text_font_size=8, time_font_size=6, **kwargs):
    """ Plot a gantt chart.
    """

    # gantt_labels = {"AGV1": "AGV1", "AGV2": "AGV2", "Job1": "Job1", "Job2": "Job2", "Job3": "Job3", "Job4": "Job4", "Job5": "Job5"}

    # 1. 从 data_job.label 里取所有唯一的 “JobX”
    job_labels = sorted(data_job['label'].unique())
    # 2. 从 data_agv.agv 里取所有唯一的 AGV id（假设值为 1, 2, …），并加上前缀
    agv_ids = sorted(data_agv['agv'].unique())
    agv_labels = [f"AGV{int(i)}" for i in agv_ids]
    # 3. 合并成一个总列表，并生成映射字典
    all_labels = agv_labels + job_labels
    gantt_labels = {lbl: lbl for lbl in all_labels}

    ax = plt.gca()
    data_job.insert(0, "y_label", None)
    for i in data_job.index:
        data_job.loc[i, "y_label"] = "$m_{" + str(data_job.machine[i]) + "}$"

    data_job.insert(0, "y_offset", data_job["y_label"].rank(method='dense', ascending=True))
    # data_job.y_offset = data_job.y_offset - 1

    bar_height = 0.65
    for i in data_job.index:
        if data_job.bar_type[i] == "PlaceholderBar":
            ax.broken_barh([(data_job.start[i], data_job.finish[i] - data_job.start[i])], (data_job.y_offset[i] - bar_height / 2, bar_height),
                           facecolor=None, edgecolor=None)

        elif data_job.bar_type[i] == "NormalBar":
            ax.broken_barh([(data_job.start[i], data_job.finish[i] - data_job.start[i])], (data_job.y_offset[i] - bar_height / 2, bar_height),
                           facecolor=data_job.color[i], edgecolor="gray", label=gantt_labels[data_job.label[i]])
            gantt_labels[data_job.label[i]] = "_nolegend_"

            ax.text((data_job.finish[i] + data_job.start[i]) / 2, data_job.y_offset[i] + 0.1,
                    data_job.text[i], verticalalignment="center", horizontalalignment="center", color="black", fontsize=text_font_size)

            ax.text(data_job.start[i] + 0.6, data_job.y_offset[i] - bar_height / 2,
                    "{0}".format(int(data_job.start[i])), verticalalignment="bottom", horizontalalignment="center", color="black", fontsize=time_font_size)
            ax.text(data_job.finish[i] - 0.6, data_job.y_offset[i] - bar_height / 2,
                    "{0}".format(int(data_job.finish[i])), verticalalignment="bottom", horizontalalignment="center", color="black", fontsize=time_font_size)

    for i in data_agv.index:
        if data_agv.finish[i] - data_agv.start[i] > 0:
            x = [data_agv.start[i], data_agv.finish[i]]
            y = [data_agv.start_m[i], data_agv.end_m[i]]
            if data_agv.color_category[i] > 0:
                if data_agv.agv[i] == 1:
                    ax.plot(x, y, linestyle='-', color=data_agv.color[i], linewidth=1, marker='o', markersize=1.5)
                else:
                    ax.plot(x, y, linestyle='--', color=data_agv.color[i], linewidth=1, marker='o', markersize=1.5)
            else:
                if data_agv.agv[i] == 1:
                    ax.plot(x, y, linestyle='-', color='black', linewidth=1, marker='o', markersize=1.5, label=gantt_labels["AGV1"])
                    gantt_labels["AGV1"] = "_nolegend_"
                else:
                    ax.plot(x, y, linestyle='--', color='black', linewidth=1, marker='o', markersize=1.5, label=gantt_labels["AGV2"])
                    gantt_labels["AGV2"] = "_nolegend_"


    plt.tick_params(axis='both', which='major', labelsize=time_font_size)

    # Set xticks
    if max_finish is not None:
        ax.set_xlim(0, max_finish)

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    # ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))

    ax.set_xlabel("Time", fontsize=text_font_size)

    # Set yticks
    if show_y_label is True:
        labels = data_job.drop_duplicates(subset=['y_label'])['y_label'].sort_values(ascending=True).to_list()

        labels.insert(0, "Depot")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
    else:
        ax.set_yticks([])

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Set title
    if show_title is True:
        ax.set_title('Factory' + str(data_job.factory.iloc[0]), {'fontsize': 10})
    if show_legend is True:
        ax.legend(loc="upper left", fontsize=time_font_size)


