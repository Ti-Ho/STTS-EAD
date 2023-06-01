#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 20:54
# @Author  : Tiho
# @File    : detection.py
# @Software: PyCharm
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import os
# import more_itertools as mit


class Detector:
    """
    Anomalies detector
    """

    def __init__(self,
                 model, data_loader, device, forecast_loss,
                 reconstruction_loss, n_features, time_len,
                 window_size, score_ratio, score_scale,
                 threshold_type, init_threshold, threshold_decay, logger):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.forecast_loss = forecast_loss
        self.reconstruction_loss = reconstruction_loss
        self.data_n_features = n_features
        self.data_time_len = time_len
        self.window_size = window_size
        self.score_ratio = score_ratio
        self.score_scale = score_scale
        self.threshold_type = threshold_type
        self.init_threshold = init_threshold
        self.threshold_decay = threshold_decay
        self.logger = logger

    def get_scores(self):
        self.logger.info("-- Predicting and calculating anomaly scores --")
        self.model.eval()

        # construct score dataframe
        fields = ['', '_forecast', '_recons', '_score']
        nodes_id = [x for x in np.arange(self.data_n_features)]
        df_cols = []
        for field in fields:
            tmp = [f"{x}{field}" for x in nodes_id]
            df_cols = df_cols + tmp

        init_values = np.array([np.nan] * len(df_cols) * self.data_time_len). \
            reshape(self.data_time_len, len(df_cols))
        score_df = pd.DataFrame(init_values, columns=df_cols)

        # get prediction and reconstruction result
        with torch.no_grad():
            for x, y, target_node, start_pt in tqdm(self.data_loader, desc="Getting Prediction/Reconstruction Result"):
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat, _ = self.model(x, target_node)
                preds = y_hat.squeeze(-1).detach().cpu().numpy()

                # Shift input to include the observed value (y) when reconstruct the x
                recon_x = torch.cat((x[:, :, 1:], y.reshape(y.shape[0], y.shape[1], 1)), dim=-1)
                _, window_recon = self.model(recon_x, target_node)
                recons = window_recon.squeeze(-1)[:, -1].detach().cpu().numpy()

                target_y = y[np.arange(len(target_node)), target_node.tolist()].detach().cpu().numpy()
                target_node = target_node.detach().cpu().numpy()
                start_pt = start_pt.detach().cpu().numpy()

                for node_i, pt_i, real_y, pred_res, recons_res in zip(target_node, start_pt, target_y, preds, recons):
                    score_df.loc[pt_i + self.window_size, str(node_i)] = real_y
                    score_df.loc[pt_i + self.window_size, str(node_i) + '_forecast'] = pred_res
                    score_df.loc[pt_i + self.window_size, str(node_i) + '_recons'] = recons_res

        # calculate anomaly scores
        for node in nodes_id:
            node_real = score_df.loc[:, str(node)]
            node_pred = score_df.loc[:, f"{str(node)}_forecast"]
            node_recon = score_df.loc[:, f"{str(node)}_recons"]
            ano_score = self.score_ratio * np.sqrt((node_pred - node_real) ** 2) \
                        + (1 - self.score_ratio) * np.sqrt((node_recon - node_real) ** 2)

            if self.score_scale:
                # q75, q25 = np.percentile(ano_score, [75, 25])
                # iqr = q75 - q25
                # median = np.median(ano_score)
                # ano_score = (ano_score - median) / (1 + iqra)
                ano_score = (ano_score - ano_score.mean()) / (ano_score.max() - ano_score.min())

            score_df[f"{node}_score"] = ano_score

        return score_df

    def get_threshold(self, scores, reg_level=1, n_decay=0):
        if self.threshold_type == "Nonparametric":
            """
            Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
            Code from TelemAnom (https://github.com/khundman/telemanom)
            """
            e_s = scores.values
            best_epsilon = None
            max_score = -10000000
            mean_e_s = np.mean(e_s[self.window_size:])
            sd_e_s = np.std(e_s[self.window_size:])

            for z in np.arange(2.5, 12, 0.5):
                epsilon = mean_e_s + sd_e_s * z
                pruned_e_s = e_s[e_s < epsilon]

                i_anom = np.argwhere(e_s >= epsilon).reshape(-1, )
                buffer = np.arange(1, 50)  # ToDo 这里的50也是一个可以调整的超参数
                i_anom = np.sort(
                    np.concatenate(
                        (
                            i_anom,
                            np.array([i + buffer for i in i_anom]).flatten(),
                            np.array([i - buffer for i in i_anom]).flatten(),
                        )
                    )
                )  # 前后50也加入i_anom中
                i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
                i_anom = np.sort(np.unique(i_anom))

                if len(i_anom) > 0:
                    # groups = [list(group) for group in mit.consecutive_groups(i_anom)]
                    # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

                    mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s  # mean降低了多少 百分比
                    sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s  # std降低了多少 百分比
                    if reg_level == 0:
                        denom = 1
                    elif reg_level == 1:
                        denom = len(i_anom)
                    elif reg_level == 2:
                        denom = len(i_anom) ** 2

                    score = (mean_perc_decrease + sd_perc_decrease) / denom

                    if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                        max_score = score
                        best_epsilon = epsilon

            if best_epsilon is None:
                best_epsilon = np.max(e_s[self.window_size:])
            return best_epsilon
        elif self.threshold_type == "Th_Decay":
            return self.init_threshold * (self.init_threshold * (self.threshold_decay ** n_decay))
        else:
            self.logger.info("get threshold error")
            return None

    def detect_anomalies_and_fill(self, epoch, save_path, n_decay):
        score_df = self.get_scores()  # 这里的scores可以是DataFramed
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        score_df.to_csv(f"{save_path}/score_epoch{epoch}.csv")
        self.logger.info(f"########## Score_df_Std: {score_df.std().mean()} ##########")

        anomaly_indices = np.array([], dtype=np.int64)
        anomaly_targets = np.array([], dtype=np.int64)

        score_threshold_list = []
        for i in range(self.data_n_features):
            # get threshold
            score_threshold = self.get_threshold(score_df[f'{i}_score'], n_decay=n_decay)
            # determine anomalies
            indices = score_df[score_df[f"{i}_score"] >= score_threshold][f"{i}_score"].index.values
            targets = np.array([i] * len(indices), dtype=np.int64)
            anomaly_indices = np.concatenate([anomaly_indices, indices])
            anomaly_targets = np.concatenate([anomaly_targets, targets])
            score_threshold_list.append(score_threshold)
        self.logger.info(f"########## Threshold: {np.std(score_threshold_list)} ##########")

        res_dict = {'ano_indices': anomaly_indices, 'ano_targets': anomaly_targets}
        anomaly_res = pd.DataFrame(res_dict, columns=['ano_indices', 'ano_targets'])
        anomaly_res.to_csv(f"{save_path}/anomalies_epoch{epoch}.csv")

        self.logger.info(f"-- save anomalies result to {save_path}/anomalies_epoch{epoch}.csv --")
        return anomaly_indices, anomaly_targets
