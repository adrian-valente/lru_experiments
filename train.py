from typing import Callable, Union

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

def train(model: nn.Module, 
          train_x: torch.Tensor, 
          train_y: torch.Tensor, 
          test_x: torch.Tensor, 
          test_y: torch.Tensor, 
          n_epochs: int, 
          lr: float,
          train_x0: torch.Tensor = None,
          test_x0: torch.Tensor = None,
          batch_size: int = 32,
          compute_accuracy: bool = False,
          acc_fn: Callable[[torch.Tensor, torch.Tensor], int] = None,
          cuda: Union[int, bool] = None,
          ) -> dict:
    if compute_accuracy:
        assert acc_fn is not None
        
    if cuda is None or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        if cuda is True:
            cuda = 0
        device = torch.device(f'cuda:{cuda}')
    model = model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    if train_x0 is not None:
        train_x0 = train_x0.to(device)
    if test_x0 is not None:
        test_x0 = test_x0.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    history = {"train_loss": [], 
               "test_loss": [],
               "train_acc": [],
               "test_acc": []
               }
    
    print("Training...")
    for epoch in range(n_epochs):
        model.train()
        batch_train_losses = []
        batch_train_accs = []
        print(f"Epoch {epoch+1}/{n_epochs}")
        for batch in tqdm(range(train_x.shape[0] // batch_size)):
            optim.zero_grad()
            rand_idxes = torch.randint(train_x.shape[0], (batch_size,))
            x = train_x[rand_idxes]
            y = train_y[rand_idxes]
            if train_x0 is not None:
                x0 = train_x0[rand_idxes]
                preds = model(x, x0)
            else:
                preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optim.step()
            loss.detach_()
            batch_train_losses.append(loss.item())
            if compute_accuracy:
                batch_train_accs.append(acc_fn(preds, y))
        history["train_loss"].append(np.mean(batch_train_losses))
        history["train_acc"].append(np.mean(batch_train_accs))
        
        # test
        model.eval()
        with torch.no_grad():
            if test_x0 is not None:
                preds = model(test_x, test_x0)
            else:
                preds = model(test_x)
            loss = loss_fn(preds, test_y)
            history["test_loss"].append(loss.item())
            if compute_accuracy:
                history["test_acc"].append(acc_fn(preds, test_y))
        print(f"Train loss: {history['train_loss'][-1]}, Test loss: {history['test_loss'][-1]}")
        if compute_accuracy:
            print(f"Train acc: {history['train_acc'][-1]}, Test acc: {history['test_acc'][-1]}")
     
    print('Done.')           
    return history