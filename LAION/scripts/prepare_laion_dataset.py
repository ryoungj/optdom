import os
import numpy as np
from tqdm import tqdm
import glob
import argparse

parser = argparse.ArgumentParser(description='Prepare LAION-400M dataset for use.')
parser.add_argument('--data_download_dir', type=str, help='base directory for downloading original dataset')
parser.add_argument('--data_processed_dir', type=str, help='base directory for saving processed dataset')
args = parser.parse_args()


def load_processed_dataset(path):
    processed_dataset = np.load(path)
    # print("Loaded CLIP featurized dataset from {}.".format(path))
    return processed_dataset


download_dataset_base_dir = os.path.join(args.data_download_dir,
                                         'the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/')
download_dataset_images_dir = os.path.join(download_dataset_base_dir, 'images')
download_dataset_texts_dir = os.path.join(download_dataset_base_dir, 'texts')
processed_dataset_base_dir = args.data_processed_dir
processed_dataset_domain_dir = os.path.join(processed_dataset_base_dir, 'original')  # a dummy domain for compatibility

os.makedirs(processed_dataset_domain_dir, exist_ok=True)

for txt_path in tqdm(glob.glob(os.path.join(download_dataset_texts_dir, '*'))):
    txt_base_name = os.path.basename(txt_path)
    img_base_name = txt_base_name.replace('text', 'img')
    img_path = os.path.join(download_dataset_images_dir, img_base_name)
    new_img_path = os.path.join(processed_dataset_domain_dir, img_base_name)
    new_txt_path = os.path.join(processed_dataset_domain_dir, txt_base_name)

    # only create symbol link to original numpy files
    os.symlink(img_path, new_img_path)
    os.symlink(txt_path, new_txt_path)
