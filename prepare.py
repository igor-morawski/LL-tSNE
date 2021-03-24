"""
Resize while keeping aspect ratio 
# https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black/44231784
"""
# python prepare.py auto_labeler/0 auto_labeler/1 --size=224 --extension=png --output_dir=resized
# python prepare.py coco --size=224 --extension=png --output_dir=coco_resized
# python prepare.py manual_labeled/0 manual_labeled/1 --size=224 --extension=png --output_dir=manual_resized
import argparse
import tqdm
from PIL import Image
import glob
import os
import os.path as op

def make_square(im, min_size, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_dir', nargs="+")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--extension', help='File extension, e.g. "png"')
    parser.add_argument('--size', type=int)
    args = parser.parse_args()
    paths = []
    for input_dir in args.input_dir:
        paths.extend(glob.glob(op.join(input_dir, "*.{}".format(args.extension))))

    if not op.exists(args.output_dir):
        os.mkdir(args.output_dir)
    assert op.exists(args.output_dir)
    
    lines = []
    for path in tqdm.tqdm(paths):
        file_name = op.split(path)[-1]
        out_file_name = op.join(args.output_dir, file_name)
        if not op.exists(out_file_name):
            img = Image.open(path)
            img = make_square(img, min_size=args.size).resize((args.size,args.size))
            img.save(out_file_name)
            assert img.size == (args.size, args.size)

        if "_low_" in file_name: 
            is_lowlight = 1
        else:
            is_lowlight = 0 
        cat_name = file_name.split("_")[0]
        lines.append(f"{file_name},1,2,3,4,{cat_name},is_lowlight={is_lowlight}")

    with open(op.join(args.output_dir, "tsne_dummy_anno.csv"), 'w') as f:
        text = "\n".join(lines)+"\n"
        f.write(text)