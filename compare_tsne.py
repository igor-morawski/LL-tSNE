# python compare_tsne.py features_auto_no_overlap_w_sim/COCO_R50_epoch_1-1.pkl features_coco/COCO_R50_epoch_1-1.pkl --output_dir=comparison

import argparse
import tqdm
from PIL import Image
import glob
import os
import os.path as op
import pickle
import tqdm

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from matplotlib.lines import Line2D

import numpy as np
import sklearn
from sklearn.manifold import TSNE
import seaborn as sns

from tsne import scatter, read_pickle, TSNE_DEFAULT
if __name__ == "__main__":
                
    # https://github.com/oreillymedia/t-SNE-tutorial
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('pickled_features_list', nargs="+")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--lr', default=200, type=int)
    parser.add_argument('--perplexity', default=None, type=int)
    parser.add_argument('--cat', default=None, type=str)
    parser.add_argument('--no_cat', action="store_true")
    args = parser.parse_args()

    assert 1.0 <= args.lr <= 1000.0
    if not op.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert op.exists(args.output_dir)
    filenames = set()
    for pickled_features in args.pickled_features_list:
        args.pickled_features = pickled_features
        plot_dir = op.join(args.output_dir, op.split(args.pickled_features)[-1].split(".")[0])
        filenames.add(op.split(args.pickled_features)[-1].split(".")[0])
    assert len(filenames) == 1
    if not op.exists(plot_dir):
        os.mkdir(plot_dir)
    assert op.exists(plot_dir)

    outputs=[]
    offset=0
    for idx, pickled_features in enumerate(args.pickled_features_list):
        args.pickled_features = pickled_features
        x, y, cats = read_pickle(args)
        y = np.hstack([cat_idx+offset for cat_idx in y])
        if args.no_cat:
            cats = tuple([str(idx)])
            y = np.hstack([idx for cat_idx in y])
        outputs.append([x,y,cats])
        offset+=len(cats)
    x = np.vstack([_x for _x, _y, _cats in outputs])
    y = np.hstack([_y for _x, _y, _cats in outputs])
    cats = tuple(np.hstack([_cats for _x, _y, _cats in outputs]))

    cat_prefix = "all" if not args.cat else args.cat 
    perplexities = [args.perplexity] if args.perplexity else [10, 30, 50]
    for perplexity in perplexities:
        print(f"Perplexity {perplexity}...")
        tsne = TSNE(learning_rate=args.lr, perplexity=perplexity, **TSNE_DEFAULT).fit_transform(x)
        scatter(tsne, y, cats, alpha=0.5)
        plt.savefig(op.join(plot_dir, f"{cat_prefix}_p{perplexity}_lr{int(args.lr)}.png"))
        plt.savefig(op.join(plot_dir, f"{cat_prefix}_p{perplexity}_lr{int(args.lr)}.svg"))
        