import torch, argparse, inspect
import numpy as np
from collections import defaultdict
from ipdb import launch_ipdb_on_exception
import matplotlib.pyplot as plt

def all_object_names(module): return {key for key, value in inspect.getmembers(module) if inspect.isfunction(object) or inspect.isclass(object)}
def load_to_dict(s): return eval(f"dict({s})")

parser = argparse.ArgumentParser()
parser.add_argument("name")
args = parser.parse_args()

with launch_ipdb_on_exception():
    with open(f"advex/{args.name}.dat", "rb") as f:
        data = torch.load(f)[:,0].detach().numpy()

    for z in range(10):
        fig=plt.figure(figsize=(8, 8))
        columns = 10
        rows = 10
        for i in range(columns*rows):
            img = data[100*z + i]
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(img, cmap='gray', vmin=0., vmax=1.)
        plt.show()

