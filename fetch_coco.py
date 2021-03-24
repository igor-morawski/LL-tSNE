# python fetch_coco.py manual_labeled/0 manual_labeled/1 --coco_dir=/tmp2/igor/train2017 --output_dir=coco --ann_file=/tmp2/igor/annotations/instances_train2017.json --extension=png
import argparse
import tqdm
from PIL import Image
import glob
import os
import os.path as op
from pycocotools.coco import COCO
import requests
import numpy as np
import cv2
CATS = ['person', 'bicycle', 'car']
int2cat = {1:'person', 2:'bicycle', 3:'car'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('reference_dir', nargs="+")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--coco_dir', type=str)
    parser.add_argument('--ann_file', type=str)
    parser.add_argument('--extension', help='File extension, e.g. "png"')
    args = parser.parse_args()
    assert args.extension
    paths = []
    for reference_dir in args.reference_dir:
        paths.extend(glob.glob(op.join(reference_dir, "*.{}".format(args.extension))))
    counts = {cat:0 for cat in CATS}
    for path in paths:
        for cat in CATS:
            if cat in path:
                counts[cat]+=1
                continue

    if not op.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert op.exists(args.output_dir)

    coco = COCO(args.ann_file)
    cat_id  = coco.getCatIds(catNms=CATS)
    ann_ids = coco.getAnnIds(catIds=cat_id, iscrowd=False)
    all_ann = coco.loadAnns(ann_ids)
    
    countdown = {k:v for (k, v) in counts.items()}
    print("Copying...")
    idx=0
    for cur_ann in all_ann:
        cat = int2cat[int(cur_ann["category_id"])]
        if not countdown[cat]:
            continue
        output_fp = op.join(args.output_dir, f"{cat}_coco_{idx}.png")
        if not op.exists(output_fp):
            cbbox      = cur_ann["bbox"]
            cimg_info  = coco.loadImgs(cur_ann["image_id"])
            filename   = cimg_info[0]["file_name"]
            coco_fp = op.join(args.coco_dir, filename)
            img_data = requests.get(cimg_info[0]['coco_url']).content
            nparr = np.frombuffer(img_data, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            width    = cimg_info[0]["width"]
            height   = cimg_info[0]["height"]
            xmin     = int(cbbox[0])
            ymin     = int(cbbox[1])
            xmax     = min(int(xmin + cbbox[2]), width-1)
            ymax     = min(int(ymin + cbbox[3]), height-1)
            instance = img_np[ymin:ymax, xmin:xmax, :]
            cv2.imwrite(output_fp, instance)
        idx+=1
        countdown[cat]-=1
        if not any([countdown[c] for c in CATS]):
            break
