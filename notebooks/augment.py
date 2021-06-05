import os
import sys
import argparse
sys.path.append('quality')

from librosa.effects import trim
from librosa.util import normalize
from soundfile import write
import matplotlib.pyplot as plt
import numpy as np

from quality.utils import butter_lp, butter_hp
from quality.qual import kick_qual
from quality.decay import env_qual, rms_qual
from quality.tonality import frq_qual, ihd_qual
from quality.response import res_qual

import torch
from torchaudio import load

datapath = "../sampleset/full"
target = "../sampleset/augm"

Fs = 44100

def load_sample(path):
    sample = load(datapath + "/" + path)[0][0].numpy()
    return normalize(butter_hp(trim(sample, top_db=50)[0],10))

def save_sample(sample, path):
    write(target + "/" + path, sample, Fs)

def layer_lowhigh(sample1, sample2, freq=400):
    lp = butter_lp(sample1, freq, order=3)
    hp = butter_hp(sample2, freq, order=4)
    n = max(len(lp), len(hp))
    out = np.empty(n)
    for i in range(0,min(len(lp), len(hp))):
        out[i] = lp[i] + hp[i]
    for i in range(min(len(lp), len(hp)), n):
        if i >= len(lp):
            out[i] = hp[i]
        else:
            out[i] = lp[i]
    return normalize(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate augmented dataset with given options')

    parser.add_argument('--path', type=str, dest='datapath', help='Directory that contains the original samples')
    parser.add_argument('-o', '--out', dest='target', type=str, help='Directory to put the new samples')
    parser.add_argument('--dropout', type=int, help="How many samples to skip each low layer iteration", default=8)
    parser.add_argument('--sort', action="store_true", help='Whether to use env_qual and res_qual to prune high and low layers')
    parser.add_argument('--copy', action="store_true", help="Whether to copy over the original dataset")
    parser.add_argument('--min_qual', type=float, help='Minimum quality for output samples', default=0)
    parser.add_argument('--max_qual', type=float, help='Maximum quality for output samples', default=1.5)

    args = parser.parse_args()

    datapath = args.datapath
    target = args.target
    
    dataset = []
    for f in os.listdir(datapath):
        dataset.append(load_sample(f))
    print(f"Found {len(dataset)} input samples")

    if(args.sort):
        def sort_low(sample):
            return 5 * env_qual(sample) + rms_qual(sample) + frq_qual(sample)

        def sort_high(sample):
            return res_qual(sample) + ihd_qual(sample)

        dataset.sort(key=sort_low)
        top_low = dataset[0:len(dataset)//2]
        dataset.sort(key=sort_high)
        top_high = dataset[0:len(dataset)//2]
    else:
        top_low = dataset[0:len(dataset),2]
        top_high = dataset[1:len(dataset),2]

    out_samples = 0

    for i in range(len(top_low)):
        for j in range(i % args.dropout, len(top_high), args.dropout):
            sample = layer_lowhigh(top_low[i], top_high[j], freq=375+5*(i % 7 + j % 5))
            q = kick_qual(sample)
            if args.min_qual <= q <= args.max_qual:
                out_samples += 1
                save_sample(sample, f"augm_{out_samples}.wav")

    print(f"Generated {out_samples} layered samples")
    
    if(args.copy):
        for i in range(len(dataset)):
            save_sample(dataset[i], f"orig_{i+1}.wav")
