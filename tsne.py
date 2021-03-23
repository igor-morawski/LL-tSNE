# python tsne.py /tmp2/igor/LL-tSNE/features/PUR50_epoch_107-1.pkl --output_dir=tsne 
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

TSNE_DEFAULT = {"n_iter" : 2*5000, "random_state" : 1}

def scatter(x, colors, cats):
    # https://github.com/oreillymedia/t-SNE-tutorial
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", len(cats)))

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

if __name__ == "__main__":
                
    # https://github.com/oreillymedia/t-SNE-tutorial
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('pickled_features')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--lr', default=200.0, type=float)
    parser.add_argument('--cat', default=None, type=str)
    args = parser.parse_args()

    assert 10.0 <= args.lr <= 1000.0
    if not op.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert op.exists(args.output_dir)
    plot_dir = op.join(args.output_dir, op.split(args.pickled_features)[-1].split(".")[0])
    if not op.exists(plot_dir):
        os.mkdir(plot_dir)
    assert op.exists(plot_dir)

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
            x.append(features.flatten())
            y.append(cat_name)
            # if len(cats) >= 4:
            #     break
            # if idx > 5:
            #     break
    cats = tuple(sorted(list(cats)))
    for cat in cats:
        cat_n = y.count(cat)
        print(f"{cat} - {cat_n} samples")
    cats_name2int = cats.index
    x = np.vstack(x)
    y = np.hstack([cats_name2int(cat_name) for cat_name in y])

    cat_prefix = "all" if not args.cat else args.cat 
    for perplexity in [10, 30, 50]:
        tsne = TSNE(learning_rate=args.lr, perplexity=perplexity, **TSNE_DEFAULT).fit_transform(x)
        scatter(tsne, y, cats)
        plt.savefig(op.join(plot_dir, f"{cat_prefix}_p{perplexity}_lr{int(args.lr)}.png"))
        plt.savefig(op.join(plot_dir, f"{cat_prefix}_p{perplexity}_lr{int(args.lr)}.svg"))
        
    