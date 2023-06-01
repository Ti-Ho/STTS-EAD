#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 19:27
# @Author  : Tiho
# @File    : args.py
# @Software: PyCharm
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # -- Data Params --
    parser.add_argument("--dataset", type=str.upper, default="NON1388")
    parser.add_argument("--del_epidemic", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--shuffle_dataset", type=lambda x: (str(x).lower() == 'true'), default=True)

    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)

    # -- Model Params --
    parser.add_argument("--temporal_embedding_dim", type=int, default=32)
    parser.add_argument("--spatial_embedding_dim", type=int, default=32)
    parser.add_argument("--TopK", type=int, default=64)
    parser.add_argument("--GATv2", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--out_dim", type=int, default=1)

    # -- Train Params --
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--use_cuda", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--print_per_epoch", type=int, default=1)
    parser.add_argument("--process_anomalies", type=lambda x: (str(x).lower() == 'true'), default=True)
    # parser.add_argument("--recons_loss_gate", type=lambda x: (str(x).lower() == 'true'), default=True)
    # parser.add_argument("--detect_n", type=int, default=10000)
    parser.add_argument("--recons_decay", type=float, default=1)  # recons_decay == 1 -> not decay; recons_decay < 1 -> decay

    # -- Anomaly detection Params --
    parser.add_argument("--detect_per_epoch", type=int, default=15)
    # ['Nonparametric', 'SPOT', 'Th_Decay']
    parser.add_argument("--threshold_type", type=str, default="Nonparametric")
    parser.add_argument("--init_threshold", type=float, default=4.0)
    parser.add_argument("--threshold_decay", type=float, default=0.8)
    # ['season_mean_4', 'mean', 'lowess']
    parser.add_argument("--fill_data_type", type=str.lower, default="mean")
    parser.add_argument("--score_ratio", type=float, default=0.5)
    parser.add_argument("--score_scale", type=lambda x: (str(x).lower() == 'true'), default=True)

    return parser
