import os
import glob
import numpy
import torch
import argparse
import pandas as pd
import shutil

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('--src_dir', '-sd', metavar='SRC DIR',
                    help='path to test data directory (default ./test_data)')
parser.add_argument('--dest_dir', '-dd', metavar='DEST DIR', default='',
                    help='path to dest dir (default: ./ILSVRC if empty)')
parser.add_argument('--samples', metavar='NAME', default='1000',
                    help='number of samples for each class from pseudo class to be added into the dataset train split (default: 1000)')


def get_labels(dataset):
    labels = []
    for data in dataset:
        labels.append(data[1])
    return labels

def get_files(dataset):
    labels = []
    for data in dataset:
        labels.append(data[0])
    return labels

def sampler(dataset, 
            samples_per_class:int=1000):
    indices = list(range(len(dataset))) 
    df = pd.DataFrame()
    df["label"] = get_labels(dataset)
    df["files"] = get_files(dataset)
    df.index = indices
    df = df.sort_index()
    label_to_count = df["label"].value_counts()
    df_test = pd.DataFrame()
    for label, value in label_to_count.items():
        df_class = df[df['label'] == label]
        if df_class.shape[0] > samples_per_class:
            df_class = df_class.sample(samples_per_class,
                                    random_state=0)
        elif df_class.shape[0] < samples_per_class:
            df_class = df_class.sample(samples_per_class,
                            random_state=0, replace=True)
        elif df_class.shape[0] == 0:
            continue
        df_test = pd.concat([df_test, df_class], axis=0)
    return df_test

def copy_pseudolabelled_images(src_dir,
                                dest_dir,
                                df):
    with open('./imagenet_classes.txt', 'r') as f:
            wnetids = f.read().splitlines()
    for i in range(df.shape[0]):
        wnet_id = wnetids[df.iloc[i]['label']]
        src_file = os.path.join(src_dir, df.iloc[i]['files'])
        dest_file = os.path.join(dest_dir, 'train', wnet_id, df.iloc[i]['files'])
        # shutil.copy(src_file, dest_file)
        os.symlink(src_file, dest_file)

def remove_pseudo_images(dest_dir):
    with open('./imagenet_classes.txt', 'r') as f:
        wnetids = f.read().splitlines()
    for wnet_id in wnetids:
        files = glob.glob(os.path.join(dest_dir, 'train', wnet_id))
        for obj in files:
            if os.path.exists(obj):
                if os.path.islink(obj):
                    os.unlink(obj)

def main():
    args = parser.parse_args()
    filename = './test_example_submission/adapt_pred.txt'
    with open(filename, 'r') as f:
            lines = f.read().splitlines()
    dataset = [(x.split(' ')[0].split('/')[-1], x.split(' ')[1]) \
                        for x in lines]
    # df = sampler(dataset, args.samples)
    # copy_pseudolabelled_images(args.src_dir, args.dest_dir, df)

if __name__=='__main__':
    main() 