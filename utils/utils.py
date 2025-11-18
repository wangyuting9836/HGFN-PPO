# -*-coding:gbk-*-
import torch
import logging
import os
import time


def diagonal_stack(tensors):
    """
    对多个张量按对角线方式拼接。

    :param tensors: 一个包含多个张量的列表，每个张量可以是矩形或方形。
    :return: 拼接后的张量，其大小由输入张量的尺寸决定。
    """
    # 计算最终张量的大小
    total_rows = sum(tensor.size(0) for tensor in tensors)  # 总行数
    total_cols = sum(tensor.size(1) for tensor in tensors)  # 总列数

    # 创建结果全零张量
    result = torch.zeros((total_rows, total_cols), dtype=tensors[0].dtype)

    # 逐一填充到对角线位置
    row_offset = 0
    col_offset = 0
    for tensor in tensors:
        rows, cols = tensor.size()  # 当前张量的尺寸
        result[row_offset:row_offset + rows, col_offset:col_offset + cols] = tensor
        row_offset += rows  # 更新行偏移
        col_offset += cols  # 更新列偏移

    return result


def create_logger(logger_file_path, filename):
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    log_name = '{}_{}.log'.format(filename, time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
