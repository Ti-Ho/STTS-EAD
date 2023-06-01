#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 19:42
# @Author  : Tiho
# @File    : data_util.py
# @Software: PyCharm
import torch
from args import get_parser
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
from torch_geometric.utils import remove_self_loops, add_self_loops
import warnings
from tqdm import tqdm
import os
import statsmodels.api as sm
import random

# import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# load data
def load_data(dataset=None, val_split=0.0):
    """
    get data
    :param val_split:
    :param dataset: choose dataset from ["NON10", "NON12"]
    :return: data shape (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size] or None))
    """
    prefix = "datasets"
    print(f"-- loading {dataset} --")
    parser = get_parser()
    args = parser.parse_args()

    del_epidemic = args.del_epidemic

    train_df = pd.read_csv(f"{prefix}/{dataset}/train.csv", index_col=0)
    test_df = pd.read_csv(f"{prefix}/{dataset}/test.csv", index_col=0)
    train_df.index = pd.to_datetime(train_df.index)
    test_df.index = pd.to_datetime(test_df.index)

    if del_epidemic and "NON" in dataset:
        epidemic_time_period = pd.date_range(start="2020-01-05", end="2020-06-01")
        train_df = train_df[~(train_df.index.isin(epidemic_time_period))]

    # rename the columns name
    train_df.columns = np.arange(len(train_df.columns))
    test_df.columns = np.arange(len(test_df.columns))

    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    train_df.fillna(0.0, inplace=True)
    test_df.fillna(0.0, inplace=True)

    val_df = None
    if val_split != 0.0:
        split = int(np.floor(len(train_df.index) * (1.0 - val_split)))
        val_df = train_df[split:]
        val_df.reset_index(inplace=True, drop=True)
        train_df = train_df[:split]
        train_df.reset_index(inplace=True, drop=True)

    return train_df, val_df, test_df


# Dataset
class SlidingWindowDataset(Dataset):
    def __init__(self, data, window):
        self.data = torch.tensor(data=data.values.T, dtype=torch.float32)
        self.window = window
        self.node_num, self.time_len = self.data.shape
        self.node_list = list(range(len(data.columns)))
        self.st, self.target_node = self.process()  # start_point, target_node

    def process(self):
        st_arr = np.array(list(range(0, self.time_len - self.window)) * self.node_num)  # start point
        node_arr = np.concatenate(
            ([[node] * (self.time_len - self.window) for node in self.node_list]))  # correspond target node
        return st_arr, node_arr

    def __len__(self):
        return len(self.st)

    def __getitem__(self, item):
        start_point = self.st[item]
        target_node = self.target_node[item]

        # target_data = self.data[target_node, start_point:start_point+self.window].reshape(1, -1)
        # ref_data = self.data[np.arange(self.node_num) != target_node, start_point:start_point+self.window]
        # X = torch.cat((target_data, ref_data), dim=0)
        X = self.data[:, start_point:start_point + self.window]
        # y = self.data[target_node, start_point + self.window]
        y = self.data[:, start_point + self.window]

        # return X, y, start_point, target_node
        return X, y, target_node, start_point


def build_graph(num_node):
    print("-- building graph --")
    # load node msg
    node_msg_df = pd.read_csv('datasets/node_msg.csv')

    # construct the graph
    relation_cols = ['city', 'territory_name', 'trade_area_type_fix']
    struct_map = {}
    for col in relation_cols:
        for group_df in node_msg_df.groupby(col):
            nodes = group_df[1].index
            for node_i in nodes:
                if node_i not in struct_map:
                    struct_map[node_i] = set()
                for node_j in nodes:
                    if node_j != node_i:
                        struct_map[node_i].add(node_j)

    # transform to edge list
    edge_indexes = [
        [],
        []
    ]

    for node_i, node_list in struct_map.items():
        for node_j in node_list:
            edge_indexes[0].append(node_i)
            edge_indexes[1].append(node_j)

    edge_indexes = torch.tensor(edge_indexes, dtype=torch.int32)
    edge_indexes, _ = remove_self_loops(edge_indexes)
    edge_indexes, _ = add_self_loops(edge_indexes, num_nodes=num_node)

    return edge_indexes


def process_fill_data(df_fill, fill_type):
    col_name = df_fill.columns[0]
    # process fill data
    if fill_type == "season_mean_4":
        lag_list = [7, 14, 21, 28]
        for lag in lag_list:
            df_fill[f'lag{lag}'] = df_fill[col_name].shift(lag)
        df_fill[fill_type] = df_fill[[f'lag{lag}' for lag in lag_list]].mean(axis=1)
        # fill nan
        null_index = df_fill[df_fill[fill_type].isnull()].index
        fill_values = df_fill.iloc[null_index][col_name]
        df_fill.loc[null_index.values.tolist(), fill_type] = fill_values
    elif fill_type == "mean":
        lag_list = range(-5, 5)
        for lag in lag_list:
            df_fill[f"lag{lag}"] = df_fill[col_name].shift(lag)
        df_fill[fill_type] = df_fill[[f'lag{lag}' for lag in lag_list]].mean(axis=1)
    else:  # lowess
        lowess = sm.nonparametric.lowess
        lowess_result = lowess(df_fill[col_name], df_fill.index, frac=0.1, it=3, delta=0.0, return_sorted=False)
        df_fill[fill_type] = lowess_result
        # plt.plot(df_fill.index, df_fill[fill_type], label='lowess')
        # plt.plot(df_fill.index, df_fill[col_name], label='real_data')
        # plt.legend()
        # plt.grid()
        # plt.show()

    return df_fill[fill_type]


def get_fill_data(train_data, fill_type, save_path):
    print("-- getting fill data --")
    fill_df = pd.DataFrame(columns=train_data.columns)
    for col_i in tqdm(train_data.columns, desc="preprocessing fill data"):
        fill_df[col_i] = process_fill_data(train_data[[col_i]], fill_type)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fill_df.to_csv(save_path + "/fill_data.csv")

    return fill_df


def replace_anomalies(train_data, fill_data, indices, targets):
    for index, target in zip(indices, targets):
        train_data.iloc[index, target] = fill_data.iloc[index, target]

    return train_data


SEED = 42


def seed_everything(seed=SEED):
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)


def stable(dataloader, seed=SEED):
    seed_everything(seed)
    return dataloader
