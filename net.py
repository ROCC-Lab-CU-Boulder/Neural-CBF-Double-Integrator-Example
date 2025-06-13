import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import constants

from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchsummary import summary
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Metrics:
    def __init__(self, mode="train", metadata_loss: str = "", metadata_activation: str = ""):
        self.mode = mode  # "train" or "test"
        activations_disp = {
            "gelu":  "GELU",
            "tanh":  "tanh",
            "elu":   "ELU "
        }
        losses_disp = {
            "mae":      "Mean-Absolute Error",
            "huber":    "Huber Loss         ",
            "logcosh":  "log-cosh           "
        }
        self.metadata_loss = losses_disp[metadata_loss]
        self.metadata_activation = activations_disp[metadata_activation]
        self.epochs = []
        self.losses = []
        self.y_true_all = []
        self.y_pred_all = []
        # parse previous run directories
        self.savedir = f"./metrics/run-{constants.identifier}/{self.metadata_loss}+{self.metadata_activation}"
        os.makedirs(self.savedir, exist_ok=True)

    def log_epoch(self, epoch, loss):
        self.epochs.append(epoch)
        self.losses.append(loss)

    def log_batch_predictions(self, y_true, y_pred):
        self.y_true_all.append(y_true.detach().cpu().numpy())
        self.y_pred_all.append(y_pred.detach().cpu().numpy())

    def export_to_csv(self):
        if self.mode == "train":
            epochs = np.array(self.epochs)
            losses = np.array(self.losses)
            np.savetxt(f"{self.savedir}/loss.csv", np.column_stack((epochs, losses)))
        elif self.mode == "test":
            y_true, y_pred = self.get_all_predictions()
            errors = y_pred - y_true
            np.savetxt(f"{self.savedir}/errors-{self.mode}.csv", errors.flatten())

    def get_all_predictions(self):
        y_true = np.concatenate(self.y_true_all, axis=0)
        y_pred = np.concatenate(self.y_pred_all, axis=0)
        return y_true, y_pred

    def compute_metrics(self):
        y_true, y_pred = self.get_all_predictions()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"mse": mse, "mae": mae, "r2": r2}

    def plot_loss(self):
        plt.figure(figsize=(6, 6))
        plt.plot(self.epochs, self.losses, label=f"{self.mode} Loss ({self.metadata_loss} + {self.metadata_activation})", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{self.mode.capitalize()} Loss Over Epochs")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/loss.png", transparent=True)

    def plot_prediction_scatter(self):
        y_true, y_pred = self.get_all_predictions()
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.4)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{self.mode.capitalize()} Predictions ({self.metadata_loss} + {self.metadata_activation})")
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/predictions-{self.mode}.png", transparent=True)

    def plot_error_distribution(self):
        y_true, y_pred = self.get_all_predictions()
        errors = y_pred - y_true
        plt.figure(figsize=(6, 6))
        plt.hist(errors.flatten(), bins=50, alpha=0.7)
        plt.title(f"{self.mode.capitalize()} Error Distribution ({self.metadata_loss} + {self.metadata_activation})")
        plt.xlabel(f"{self.mode.capitalize()} Prediction Error")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/error-{self.mode}.png", transparent=True)


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, activation="gelu", dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.activation_type = activation
        self.activation = self._get_activation(activation)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, self.output_dim)
        )

    def _get_activation(self, name):
        if name.lower() == "gelu":
            return nn.GELU()
        elif name.lower() == "tanh":
            return nn.Tanh()
        elif name.lower() == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        return self.net(x)

    def get_input_gradient(self, x):
        x = x.clone().detach().requires_grad_(True)
        output = self.forward(x)
        if output.ndim == 1:
            output = output.unsqueeze(0)
        grads = []
        for i in range(output.shape[1]):
            grad = torch.autograd.grad(outputs=output[:, i], inputs=x,
                                       grad_outputs=torch.ones_like(output[:, i]),
                                       retain_graph=True, create_graph=True)[0]
            grads.append(grad)
        return torch.stack(grads, dim=1)  # Shape: [batch_size, output_dim, input_dim]


class ModelTrainer:
    def __init__(
        self,
        model,
        loss_type="mae",
        lr=1e-3,
        batch_size=32,
        epochs=10,
        device=None,
        plots=False,
    ):
        self.model = model
        self.loss_type = loss_type
        self.loss_fn = self._get_loss_function(loss_type)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.plots = plots

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _get_loss_function(self, loss_type):
        if loss_type == "mae":
            return nn.L1Loss()
        elif loss_type == "huber":
            return nn.HuberLoss()
        elif loss_type == "logcosh":
            def logcosh_loss(x, y):
                return torch.mean(torch.log(torch.cosh(x - y + 1e-12)))
            return logcosh_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def train(self, x_train, y_train):
        self.model.train()
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # save metrics keeping in mind which loss function and activation we're using
        metrics = Metrics(mode="train", metadata_loss=self.loss_type, metadata_activation=self.model.activation_type)

        for epoch in range(self.epochs):
            num_samples = 0
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for x_batch, y_batch in pbar:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))
                if self.plots:
                    metrics.log_batch_predictions(y_batch, y_pred)
            total_loss += loss.item() * x_batch.size(0)  # total loss weighted by batch size
            num_samples += x_batch.size(0)
            avg_loss = total_loss / num_samples
            pbar.set_postfix(loss=avg_loss)

            if self.plots:
                metrics.log_epoch(epoch, avg_loss)

        if self.plots:
            # metrics.plot_loss()
            # metrics.plot_prediction_scatter()
            # metrics.plot_error_distribution()
            metrics.export_to_csv()
            print(metrics.compute_metrics())

    def test(self, x_test, y_test):
        self.model.eval()
        dataset = TensorDataset(x_test, y_test)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        metrics = Metrics(mode="test", metadata_loss=self.loss_type, metadata_activation=self.model.activation_type)

        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                total_loss += loss.item() * x_batch.size(0)
            if self.plots:
                metrics.log_batch_predictions(y_batch, y_pred)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Test Loss: {avg_loss:.4f}")
        if self.plots:
            # metrics.plot_prediction_scatter()
            # metrics.plot_error_distribution()
            metrics.export_to_csv()

    def predict(self, inpt):
        self.model.eval()
        with torch.no_grad():
            inpt = torch.tensor(inpt, dtype=torch.float32)
            return self.model(inpt).cpu().numpy()

    def compute_input_gradients(self, inpt):
        inpt = torch.tensor(inpt, dtype=torch.float32)
        return self.model.get_input_gradient(inpt).detach().cpu().numpy()

def prompt_save_model(model_trainer: ModelTrainer):
    """Ask the user if they want to save the model """
    inpt = input("Do you want to save this model? (y/n): ")
    if inpt.lower() == "y":
        print("*You selected yes*")
        inpt_path = input("Type in the filename: ")
        torch.save(model_trainer.model.state_dict(), inpt_path)
    else:
        print("*You selected no*")
