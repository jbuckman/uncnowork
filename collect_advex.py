import torch, argparse, inspect
import numpy as np
from collections import defaultdict
from ipdb import launch_ipdb_on_exception
import matplotlib.pyplot as plt

def all_object_names(module): return {key for key, value in inspect.getmembers(module) if inspect.isfunction(object) or inspect.isclass(object)}
def load_to_dict(s): return eval(f"dict({s})")
import tasks, algorithms, advsearch

parser = argparse.ArgumentParser()
parser.add_argument("--name")
parser.add_argument("--task", choices=all_object_names(tasks), default="mnist")
parser.add_argument("--algo", choices=all_object_names(algorithms), default="vanilla")
parser.add_argument("--task_params", type=load_to_dict, default="")
parser.add_argument("--algo_params", type=load_to_dict, default="")
parser.add_argument("--minibatch_size", default=100, type=int)
parser.add_argument("--total", default=1000, type=int)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--search_params", type=load_to_dict, default="")
args = parser.parse_args()

with launch_ipdb_on_exception():
    task = getattr(tasks, args.task)(**args.task_params)
    learner = getattr(algorithms, args.algo)(task, **args.algo_params)

    with open(f"models/{args.name}.pt", "rb") as f:
        learner.nn = torch.load(f)

    xs = []
    for batch_i in range(args.total // args.minibatch_size):
        x = advsearch.find_highest(task, learner, n=args.minibatch_size, return_all=True, **args.search_params)
        xs.append(x)
        print(f"{(batch_i+1)*args.minibatch_size/args.total:.2%}\tcollected={(batch_i+1)*args.minibatch_size}/{args.total}")

    with open(f"advex/{args.name}.dat", "wb") as f:
        torch.save(torch.cat(xs, 0), f)
