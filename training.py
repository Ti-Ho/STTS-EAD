#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 23:54
# @Author  : Tiho
# @File    : training.py
# @Software: PyCharm
import os.path
import time
import torch.cuda
from math import sqrt
from Utils.data_util import *
from detection import Detector
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            forecast_loss,
            reconstruction_loss,
            save_path="",
            x_train=None,
            val_loader=None,
            test_loader=None,
            logger=None
    ):
        parser = get_parser()
        args = parser.parse_args()

        # -- Data Params --
        self.val_split = args.val_split
        self.shuffle = args.shuffle_dataset
        self.window_size = args.window_size
        self.batch_size = args.batch_size
        self.train_data = x_train
        self.val_loader = val_loader

        # -- Train Params --
        self.model = model
        self.optimizer = optimizer
        self.forecast_loss = forecast_loss
        self.reconstruction_loss = reconstruction_loss
        self.device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
        self.save_path = save_path
        self.n_epochs = args.epochs
        self.print_per_epoch = args.print_per_epoch
        self.process_anomalies = args.process_anomalies
        self.lr = args.lr
        self.recons_decay = args.recons_decay
        self.recons_loss_ratio = 1.0

        # -- Anomaly detection Params --
        self.detect_per_epoch = args.detect_per_epoch
        self.threshold_type = args.threshold_type
        self.init_threshold = args.init_threshold
        self.threshold_decay = args.threshold_decay
        self.fill_data_type = args.fill_data_type
        self.score_ratio = args.score_ratio
        self.score_scale = args.score_scale

        self.fill_data = get_fill_data(self.train_data, self.fill_data_type, self.save_path)
        self.test_loader = test_loader

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
            "val_rmse": [],
            "val_mae": []
        }

        self.logger = logger

        if self.device == "cuda":
            self.model.cuda()

    def fit(self):
        # 1. Prepare initial Dataset and DataLoader
        seed_everything()
        train_dataset = SlidingWindowDataset(self.train_data, self.window_size)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = self.val_loader

        eva_start = time.time()
        # init_train_loader = self.train_loader
        init_train_loss = self.evaluate(train_loader)
        eva_end = time.time()
        self.logger.info(f"Init total train loss: {init_train_loss[2]}, evaluating done in {eva_end - eva_start}s")

        # init_val_loader = self.val_loader
        if val_loader is not None:
            eva_start = time.time()
            init_val_loss = self.evaluate(val_loader)
            eva_end = time.time()
            self.logger.info(f"Init total val loss: {init_val_loss[2]}, evaluating done in {eva_end - eva_start}s")

        # 2. Train model
        self.logger.info(f"-- Starting Training model for {self.n_epochs} epochs --")
        train_start = time.time()
        optimal_val_rmse = None
        n_decay = 1
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_train_losses = []
            recon_train_losses = []
            # 1). Train for one epoch
            for x, y, target_node, _ in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                preds, recons = self.model(x, target_node)
                preds = preds.squeeze(1)
                target_y = y[np.arange(len(target_node)), target_node.tolist()]
                forecast_loss = torch.sqrt(self.forecast_loss(target_y, preds))
                recon_loss = None

                recons = recons.squeeze(2)
                target_x = x[np.arange(len(target_node)), target_node.tolist(), :]
                recon_loss = torch.sqrt(self.reconstruction_loss(target_x, recons))
                loss = forecast_loss + self.recons_loss_ratio * recon_loss

                loss.backward()
                self.optimizer.step()
                recon_train_losses.append(recon_loss.item())
                forecast_train_losses.append(forecast_loss.item())

            forecast_train_losses = np.array(forecast_train_losses)
            forecast_epoch_loss = np.sqrt((forecast_train_losses ** 2).mean())
            recon_train_losses = np.array(recon_train_losses)
            recon_epoch_loss = np.sqrt((recon_train_losses ** 2).mean())
            total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)
            self.losses["train_forecast"].append(forecast_epoch_loss)

            # 2). Evaluate on validation set
            forecast_val_loss, recon_val_loss, total_val_loss = None, None, None
            if val_loader is not None:
                forecast_val_loss, recon_val_loss, total_val_loss, rmse, mae = self.evaluate(val_loader)
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_total"].append(total_val_loss)
                self.losses["val_rmse"].append(rmse)
                self.losses["val_mae"].append(mae)
                if total_val_loss is None:
                    total_val_loss = forecast_val_loss
                if optimal_val_rmse is None or rmse <= optimal_val_rmse:
                    optimal_val_rmse = rmse
                    self.save(f"model.pt")

            epoch_time = time.time() - epoch_start

            # print train and validation msg
            if epoch % self.print_per_epoch == 0:
                train_msg = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    train_msg += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_rmse = {rmse}, val_mae = {mae}, "
                        f"recon_loss = {recon_epoch_loss:.5f}, "
                        f"total_loss = {total_epoch_loss:.5f}"
                    )

                train_msg += f" [{epoch_time:.1f}s]"
                self.logger.info(train_msg)
                _, _, _, test_rmse, test_mae = self.evaluate(self.test_loader)
                self.logger.info(f"test_rmse: {test_rmse}, test_mae: {test_mae}")

            # 3). Anomaly detection
            if epoch % self.detect_per_epoch == 0 and self.process_anomalies:
                # recons_loss_decay
                self.recons_loss_ratio = self.recons_loss_ratio * self.recons_decay

                # detector init
                n_features = len(self.train_data.columns)
                time_len = len(self.train_data)

                detector = Detector(self.model, train_loader, self.device, self.forecast_loss,
                                    self.reconstruction_loss, n_features, time_len,
                                    self.window_size, self.score_ratio, self.score_scale,
                                    self.threshold_type, self.init_threshold, self.threshold_decay, self.logger)
                # detect anomalies
                anomalies_indices, anomalies_targets = detector.detect_anomalies_and_fill(epoch=epoch,
                                                                                          save_path=self.save_path,
                                                                                          n_decay=n_decay)
                n_decay += 1
                # replace anomalies with fill data
                self.train_data = replace_anomalies(self.train_data, self.fill_data, anomalies_indices, anomalies_targets)
                self.logger.info(f"########## Data Std: {self.train_data.std().mean()} ##########")

                seed_everything()
                # reconstruct dataset and dataloader
                train_dataset = SlidingWindowDataset(self.train_data, self.window_size)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
                self.logger.info(f"Detect and Replace {len(anomalies_indices)} anomalies, accounts for "
                      f"{len(anomalies_indices) * 100 / len(train_dataset):.2f}%, Dataloader has been updated")

        if val_loader is None:
            self.save(f"model.pt")

        # save the train record
        with open(f'{self.save_path}/train_loss_record.pkl', 'wb') as f:
            pickle.dump(self.losses, f)

        train_time = time.time() - train_start
        self.logger.info(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader):
        self.model.eval()

        forecast_losses = []
        recon_losses = []
        y_list = np.array([])
        y_hat_list = np.array([])
        with torch.no_grad():
            for x, y, target_node, _ in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                preds, recons = self.model(x, target_node)
                preds = preds.squeeze(1)
                target_y = y[np.arange(len(target_node)), target_node.tolist()]
                forecast_loss = torch.sqrt(self.forecast_loss(target_y, preds))

                recons = recons.squeeze(2)
                target_x = x[np.arange(len(target_node)), target_node.tolist(), :]
                recon_loss = torch.sqrt(self.reconstruction_loss(target_x, recons))
                recon_losses.append(recon_loss.item())

                forecast_losses.append(forecast_loss.item())
                y_list = np.concatenate([y_list, target_y.detach().cpu().numpy()])
                y_hat_list = np.concatenate([y_hat_list, preds.detach().cpu().numpy()])

        forecast_losses = np.array(forecast_losses)
        y_list = np.array(y_list)
        y_hat_list = np.array(y_hat_list)
        forecast_loss = np.sqrt((forecast_losses ** 2).mean())
        rmse = sqrt(mean_squared_error(y_list, y_hat_list))
        mae = mean_absolute_error(y_list, y_hat_list)

        recon_losses = np.array(recon_losses)
        recon_loss = np.sqrt((recon_losses ** 2).mean())
        total_loss = forecast_loss + recon_loss * self.recons_loss_ratio

        return forecast_loss, recon_loss, total_loss, rmse, mae

    def save(self, file_name):
        PATH = f"{self.save_path}/{file_name}"
        self.logger.info(f"-- save model to {PATH} --")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))
