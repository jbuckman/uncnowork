import torch, argparse, inspect
import numpy as np
from collections import defaultdict
from ipdb import launch_ipdb_on_exception
import matplotlib.pyplot as plt

def all_object_names(module): return {key for key, value in inspect.getmembers(module) if inspect.isfunction(object) or inspect.isclass(object)}
def load_to_dict(s): return eval(f"dict({s})")
import tasks, algorithms, advsearch

parser = argparse.ArgumentParser()
parser.add_argument("--steps", default=60000, type=int)
parser.add_argument("--minibatch_size", default=128, type=int)
parser.add_argument("--dataset_size", default=None, type=int)
parser.add_argument("--test_runs", default=100, type=int)
parser.add_argument("--name", default=None)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--task", choices=all_object_names(tasks), default="mnist")
parser.add_argument("--algo", choices=all_object_names(algorithms), default="vanilla")
parser.add_argument("--task_params", type=load_to_dict, default="")
parser.add_argument("--algo_params", type=load_to_dict, default="")
args = parser.parse_args()

with launch_ipdb_on_exception():
    task = getattr(tasks, args.task)(**args.task_params)
    learner = getattr(algorithms, args.algo)(task, total_steps=args.steps, dataset_size=args.dataset_size, **args.algo_params)

    if args.seed is not None: torch.manual_seed(args.seed)

    losses = defaultdict(list)
    for step in range(args.steps):
        x_minibatch, y_minibatch = task.train_sample(args.minibatch_size)
        loss = learner.learn(step, x_minibatch, y_minibatch)
        for k,v in loss.items(): losses[k].append(v)
        if (step + 1) % 100 == 0:
            print(f"{(step+1)/args.steps:.2%}\tstep={step+1}/{args.steps} | " + ' '.join([f"{k}={sum(v[-25:])/25:.3}" for k,v in losses.items()]))
        if (step + 1) % 1000 == 0:
            print("Evaluating...")
            test_losses = defaultdict(list)
            test_gen = task.test_sample(args.minibatch_size)
            for test_step in range(args.test_runs):
                try:
                    x_minibatch, y_minibatch = next(test_gen)
                    test_loss = learner.learn(step, x_minibatch, y_minibatch)
                    for k, v in test_loss.items(): test_losses[k].append(v)
                    # if (test_step + 1) % 10 == 0: print(f"\t{(test_step+1)/args.test_runs:.2%}\tstep={test_step+1}/{args.test_runs}")
                except StopIteration:
                    pass
            for name, val in test_losses.items(): print(f"test {name}: {np.mean(val):.3}")

    print("Training complete.")
    namestr = f"-{args.name}" if args.name is not None else ""
    with open(f"models/{args.task}-{args.algo}{namestr}.pt", "wb") as f:
        torch.save(learner.nn, f)

    with open(f"models/{args.task}-{args.algo}.pt", "rb") as f: learner.nn = torch.load(f)

    print("\n\nFinding highest-output image...")
    x = advsearch.find_highest(task, learner)
    if hasattr(task, "score_fn"):
        print(f"Model evaluation: {learner.predict(x).item()}, true score: {task.score_fn(x).item()}")
    else:
        print(f"Model evaluation: {learner.predict(x).item()}")
    npimg = x[0,0].detach().numpy()
    plt.imshow(npimg, cmap='gray', vmin=0., vmax=1.)
    plt.show()

    if hasattr(task, "score_fn"):
        print("\n\nFinding highest-error image...")
        x = advsearch.find_highest(task, learner, search_for_error=True)
        print(f"Model evaluation: {learner.predict(x).item()}, true score: {task.score_fn(x).item()}")
        npimg = x[0, 0].detach().numpy()
        plt.imshow(npimg, cmap='gray', vmin=0., vmax=1.)
        plt.show()

    if hasattr(task, "restriction_fn"):
        print("\n\nFinding highest-output restricted image...")
        x = advsearch.find_highest(task, learner, restriction_fn=task.restriction_fn)
        print(f"Model evaluation on submanifold: {learner.predict(x).item()}, true score: {task.score_fn(x).item()}")
        npimg = x[0,0].detach().numpy()
        plt.imshow(npimg, cmap='gray', vmin=0., vmax=1.)
        plt.show()

        if hasattr(task, "score_fn"):
            print("\n\nFinding highest-error restricted image...")
            x = advsearch.find_highest(task, learner, search_for_error=True, restriction_fn=task.restriction_fn)
            print(f"Model evaluation: {learner.predict(x).item()}, true score: {task.score_fn(x).item()}")
            npimg = x[0, 0].detach().numpy()
            plt.imshow(npimg, cmap='gray', vmin=0., vmax=1.)
            plt.show()
