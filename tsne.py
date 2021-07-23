# python tsne.py /tmp2/igor/LL-tSNE/features/S_R50_epoch_121-1.pkl --output_dir=tsne 
# python tsne.py /tmp2/igor/LL-tSNE/features_manual/COCO_R50_epoch_1-1.pkl --output_dir=manual --lr=10 --perplexity=5
# https://towardsdatascience.com/how-to-tune-hyperparameters-of-tsne-7c0596a18868
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

TSNE_DEFAULT = {"n_iter" : 3*5000, "random_state" : 3, "init":"pca"}

def scatter(x, colors, cats, alpha=1):
    # https://github.com/oreillymedia/t-SNE-tutorial
    # We choose a color palette with seaborn.
    rgb_palette = np.array(sns.color_palette("hls", len(cats)))
    palette = np.c_[rgb_palette, alpha*np.ones(rgb_palette.shape[0])]

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each class.
    txts = []
    for i in range(len(cats)):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    # custom legend
    # rgba_palette = x=np.c_[palette, np.ones(palette.shape[0])]
    legend_elements = [Line2D([0], [0], marker='o', color=palette[i], label="{}, ({})".format(cats[i], i),
                          markerfacecolor=palette[i], markersize=15) for i in range(len(cats))]

    ax.legend(handles=legend_elements, loc='lower right')
    return f, ax, sc, txts

def read_pickle(args):
    x = []
    y = []
    cats = set()
    with open(args.pickled_features, "rb") as handle:
        data = pickle.load(handle)
        for idx, sample in tqdm.tqdm(enumerate(data)):
            features = sample["features"]
            img_metas = sample["img_metas"]
            cat_name = "_".join(img_metas["ori_filename"].split("_")[:2])
            if args.cat and (args.cat not in cat_name):
                continue
            cats.add(cat_name)
            x.append(features)
            y.append(cat_name)
    cats = tuple(sorted(list(cats)))
    for cat in cats:
        cat_n = y.count(cat)
        print(f"{cat} - {cat_n} samples")
    cats_name2int = cats.index
    x = np.array(x)
    B, C, H, W = x.shape
    std=np.moveaxis(x, 0, 1).std(axis=(1,2,3))
    for c in range(C):
        x[:, c, :, :]/=std[c]
    x = x.reshape(B, -1)
    y = np.hstack([cats_name2int(cat_name) for cat_name in y])
    return x, y, cats


if __name__ == "__main__":
                
    # https://github.com/oreillymedia/t-SNE-tutorial
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('pickled_features')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--lr', default=200, type=int)
    parser.add_argument('--perplexity', default=None, type=int)
    parser.add_argument('--cat', default=None, type=str)
    args = parser.parse_args()

    assert 1.0 <= args.lr <= 1000.0
    if not op.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert op.exists(args.output_dir)
    plot_dir = op.join(args.output_dir, op.split(args.pickled_features)[-1].split(".")[0])
    if not op.exists(plot_dir):
        os.mkdir(plot_dir)
    assert op.exists(plot_dir)

    x, y, cats = read_pickle(args)
    cat_prefix = "all" if not args.cat else args.cat 
    perplexities = [args.perplexity] if args.perplexity else [10, 30, 50]
    for perplexity in perplexities:
        tsne = TSNE(learning_rate=args.lr, perplexity=perplexity, **TSNE_DEFAULT).fit_transform(x)
        scatter(tsne, y, cats, alpha=0.5)
        print(op.join(plot_dir, f"{cat_prefix}_p{perplexity}_lr{int(args.lr)}.png"))
        plt.savefig(op.join(plot_dir, f"{cat_prefix}_p{perplexity}_lr{int(args.lr)}.png"))
        plt.savefig(op.join(plot_dir, f"{cat_prefix}_p{perplexity}_lr{int(args.lr)}.svg"))
        
    