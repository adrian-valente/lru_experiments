import matplotlib.pyplot as plt
import torch

from model import LRU, SequenceLayer, DeepLRUModel
from tasks import flip_flop
from train import train


if __name__ == '__main__':
    # Tasks parameters
    dimensions = 1
    timesteps = 1000
    p = 0.2
    N = 5000
    p_train = 0.8
    
    # get task tensors
    x, y = flip_flop(dimensions, timesteps, N, p)
    x_train, x_test = x[:int(p_train*N)], x[int(p_train*N):]
    y_train, y_test = y[:int(p_train*N)], y[int(p_train*N):]

    # define model
    d_hidden = 128
    model = DeepLRUModel(dimensions, d_hidden, 1, [d_hidden, d_hidden, dimensions], skip_connection=True)
    
    # train
    hist = train(model, x_train, y_train, x_test, y_test, n_epochs=10,  lr=1e-3, batch_size=32)
    
    # some plots
    plt.plot(hist["train_loss"], label="train loss")
    plt.plot(hist["test_loss"], label="test loss")
    plt.show()
    
    # save model
    torch.save(model.state_dict(), "saved_models/model.pt")
    