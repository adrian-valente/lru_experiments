import argparse
from experiment import Experiment


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", type=str, default="flip_flop1")
    argparser.add_argument("--model", type=str, default="DeepLRUModel")
    argparser.add_argument("--lr", type=float, default=1e-3)
    argparser.add_argument("--epochs", type=int, default=10)
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--d_hidden", type=int, default=128)
    argparser.add_argument("--skip_connection", action="store_true")
    argparser.add_argument("--model_name", type=str, default="model")
    argparser.add_argument("--timesteps", type=int, default=1000)
    argparser.add_argument("--n_examples", type=int, default=5000)
    args = argparser.parse_args()
    
    exp = Experiment(**vars(args))
    exp.train()
    exp.save()
    