import os
import os.path as osp
import re
import argparse
import numpy as np
from pdb import set_trace

def parse_args():
    parser = argparse.ArgumentParser(description='Experiment summary parser')
    parser.add_argument('--save_dir', default='/mnt/models/Tuning_all/checkpoints_tune', type=str)
    parser.add_argument('--exp_format', type=str)
    return parser.parse_args()


def read_num(saveDir, exp):
    path = osp.join(saveDir, exp, 'log.txt')
    if not osp.isfile(path):
        return -1
    with open(path, 'r') as file:
        lines = file.read().splitlines()
    bestAcc = -1
    for line in lines[-20:]:
        # set_trace()
        groups = re.match("^test: Average loss: [0-9]+\.[0-9]+, Accuracy: top1 \(([0-9]+\.[0-9]+)%\) top5 \([0-9]+\.[0-9]+%\)$", line)
        if groups:
            bestAcc = float(groups[1])
    return bestAcc

def main():
    args = parse_args()
    exps_all = os.listdir(args.save_dir)
    exps_select = []
    # set_trace()
    for exp in exps_all:
        if re.match(args.exp_format, exp) is not None:
            exps_select.append(exp)

    numbers = []
    for exp in exps_select:
        acc = read_num(args.save_dir, exp)
        if acc > 0:
            print("{}: {}".format(exp, acc))
            numbers.append(acc)
        else:
            print("read fail for {}".format(exp))

    if len(numbers) > 0:
        print("mean is {}, std is {} for {}".format(np.mean(np.array(numbers)), np.std(np.array(numbers)), numbers))


if __name__ == "__main__":
    main()
