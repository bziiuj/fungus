import argparse
from pathlib import Path

import numpy as np
from skimage import exposure
from skimage import io
from tqdm import tqdm
import imageio


def process(dataset_dir, out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)
    for image_path in tqdm(dataset_dir.glob('**/*.tif')):
        # use freeimage plugin to obtain 16 bits per sample
        img = imageio.imread(image_path).astype(np.float)
        img = img / (2 ** 16 - 1)
        p5, p95 = np.percentile(img, (5, 95))
        img = exposure.rescale_intensity(
            img, in_range=(p5, p95), out_range=(0, 1))
        kind, = image_path.parent.parts[-1:]
        save_path = out_dir / kind
        save_path.mkdir(exist_ok=True, parents=True)
        np.save(save_path / image_path.stem, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_dir', type=Path, help='absolute path to directory with dataset')
    parser.add_argument(
        '--out_dir', type=Path, help='absolute path to output directory', default=None)
    args = parser.parse_args()
    if not args.out_dir:
        args.out_dir = (args.dataset_dir / '../preprocessed').resolve()
    process(args.dataset_dir, args.out_dir)
