import argparse
import os
import pickle
from multiprocessing import Pool

import matplotlib.pyplot as plt

from qual import eval_dir

# import sys

def func(x):
    m, q = eval_dir(x)
    return (x, m, q)


def order(x):
    y = x[0].split("/")
    n = y[len(y) - 1]
    return int(n)


def val(x):
    return x[1]


# def progbar(i, n, size=16):
#     done = (i * size) // n
#     bar = ""
#     for i in range(size):
#         bar += "█" if i <= done else "░"
#     return bar

# def stream(message):
#     sys.stdout.write(f"\r{message}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation on test samples and return mean, quals per epoch"
    )

    parser.add_argument(
        "-n",
        "--workers",
        dest="workers",
        type=int,
        default=4,
        help="Number of workers to spawn.",
    )
    parser.add_argument("path", type=str, help="Directory that contains the epochs")
    parser.add_argument(
        "-o", "--out", dest="dump", type=str, help="Filename to put pickle dump"
    )
    parser.add_argument(
        "-s", "--savefig", dest="save_fig", type=str, help="Filename to save the figure"
    )

    args = parser.parse_args()

    paths = []
    for f in os.listdir(args.path):
        if os.path.isdir(args.path + f):
            paths.append(args.path + f)
    paths.sort()

    with Pool(args.workers) as p:
        result = p.map(func, paths, chunksize=2)

    result.sort(key=order)

    with open(args.dump, "wb") as f:
        pickle.dump(result, f)

    if args.save_fig is not None:
        plt.plot(range(0, len(paths)), list(map(val, result)))
        plt.savefig(args.save_fig)

    quit()
