import argparse
from functools import partial
import json
import matplotlib.pyplot as plt
import torch

from typing import Tuple

import model as modellib
import tasks
from train import train

class Experiment:
    
    def __init__(self,
                 task: str,
                 timesteps: int = 1000,
                 n_examples: int = 5000,
                 model: str = 'DeepLRUModel',
                 d_hidden: int = 128,
                 d_encoder: int = 64,
                 depth: int = 1,
                 skip_connection: bool = True,
                 lr: float = 1e-3,
                 epochs: int = 10,
                 model_name: str = "model",
                 task_kwargs: dict = {},
                 model_kwargs: dict = {},
                 ) -> None:
        # Task parameters
        assert task in tasks.tasks.keys()
        self.task = task
        self.task_fn = partial(getattr(tasks, task), **task_kwargs) if task_kwargs is not None \
                        else getattr(tasks, task)
        self.timesteps = timesteps
        self.N = n_examples
        self.d_in, self.d_out = tasks.tasks[task]
        
        # Model parameters
        # TODO: add more flexibility to define the architecture?
        # TODO: encoder width
        self.model_class = model
        assert hasattr(modellib, self.model_class)
        if self.model_class == "DeepLRUModel":
            self.model = modellib.DeepLRUModel(self.d_in, d_hidden, depth, [d_hidden, d_hidden, d_hidden], 
                                            [self.d_out], skip_connection=skip_connection, **model_kwargs)
        elif self.model_class == "DSModel":
            self.model = modellib.DSModel(self.d_in, 1, d_hidden, depth, [d_hidden, d_hidden], 
                                       [d_encoder, d_hidden * depth], [self.d_out],
                                       lru_kwargs={"skip_connection": skip_connection}, **model_kwargs)
        
        # Training parameters
        self.lr = lr
        self.n_epochs = epochs
        self.batch_size = 32
        self.model_name = model_name
        
        
    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, \
                                torch.Tensor, torch.Tensor]:
        data = self.task_fn(n=self.N, timesteps=self.timesteps)
        if len(data) == 2:
            x, y = data
            x0_train, x0_test = None, None
        elif len(data) == 3:
            x, y, x0 = data
        p_train = 0.8
        N = self.N
        x_train, x_test = x[:int(p_train*N)], x[int(p_train*N):]
        y_train, y_test = y[:int(p_train*N)], y[int(p_train*N):]
        if len(data) == 3:
            x0_train, x0_test = x0[:int(p_train*N)], x0[int(p_train*N):]
        return x_train, y_train, x0_train, x_test, y_test, x0_test
    
    def train(self) -> None:
        x_train, y_train, x0_train, x_test, y_test, x0_test = self.generate_data()
        self.hist = train(self.model, x_train, y_train, x_test, y_test, 
                          n_epochs=self.n_epochs, lr=self.lr, batch_size=32,
                          train_x0=x0_train, test_x0=x0_test)
        # TODO: history append

    def save(self) -> None:
        torch.save(self.model.state_dict(), f"saved_models/{self.model_name}.pt")
        exp_dict = {"task": self.task,
                    "timesteps": self.timesteps,
                    "N": self.N,
                    "model": self.model_class,
                    "model_architecture": self.model.get_metadata(),
                    "lr": self.lr,
                    "n_epochs": self.n_epochs,
                    "batch_size": self.batch_size,
                    "history": self.hist
                    }
        with open(f"saved_models/{self.model_name}.json", "w") as f:
            json.dump(exp_dict, f)
       
       
    def load(self, name: str) -> None:
        with open(f"saved_models/{name}.json", "r") as f:
            exp_dict = json.load(f)
        self.task = exp_dict["task"]
        self.task_fn = getattr(tasks, self.task)
        self.d_in, self.d_out = tasks.tasks[self.task]
        self.timesteps = exp_dict["timesteps"]
        self.N = exp_dict["N"]
        self.model_class = exp_dict["model"]
        model_builder = getattr(self.model, self.model_class)
        self.model = model_builder(**exp_dict["model_architecture"])
        self.lr = exp_dict["lr"]
        self.n_epochs = exp_dict["n_epochs"]
        self.batch_size = exp_dict["batch_size"]
        self.hist = exp_dict["history"]
        self.model_name = name
        
    def plot_loss(self) -> None:
        plt.plot(self.hist["train_loss"], label="train")
        plt.plot(self.hist["test_loss"], label="test")
        plt.legend()
        plt.show()
        