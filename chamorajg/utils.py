from typing import Optional, Tuple, List, Dict, Callable, cast

import os
import ast
import shutil
import pandas as pd
import os.path as osp

from PIL import Image
from tqdm import tqdm
from pathlib import Path

PARENT_PATH = Path(__file__).parent.resolve()
PROJECT_PATH = PARENT_PATH.parent.parent.absolute().resolve()
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', 
                    '.tiff', '.webp')

def resolve_path(path):
    return osp.expanduser(path)

def validate_path(path):
    if osp.exists(path):
        return path
    elif osp.exists(osp.join(PARENT_PATH, path)):
        return osp.join(PARENT_PATH, path)
    elif osp.exists(osp.join(PROJECT_PATH, path)):
        return osp.join(PROJECT_PATH, path)
    else:
        raise FileNotFoundError


def parse_val(foldername):
    foldername = resolve_path(foldername)
    df = pd.read_csv(osp.join(foldername, 'LOC_val_solution.csv'))
    for i in tqdm(range(df.shape[0])):
        src_loc = osp.join(foldername, 'ILSVRC', 'val',
                df.iloc[i]['ImageId']+'.JPEG')
        dest_loc = osp.join(foldername, 'ILSVRC', 'val',
                    df.iloc[i]['PredictionString'].split(' ')[0],
                    df.iloc[i]['ImageId']+'.JPEG')
        if osp.exists(dest_loc):
            continue
        os.makedirs(osp.join(foldername, 'ILSVRC', 'val', 
                        df.iloc[i]['PredictionString'].split(' ')[0]),
                        exist_ok=True)
        shutil.move(src_loc, dest_loc)

def class_to_idx(filename):
    with open(str(filename), 'r') as f:
        _labeldict = f.read()
    _labeldict = ast.literal_eval(_labeldict)
    # print(_labeldict[999]['id'][:-2])
    _labeldict = {v['id'][-1] + v['id'][:-2]:int(k) 
                        for k, v in _labeldict.items()}
    return _labeldict

def label_to_idx(filename):
    with open(str(filename), 'r') as f:
        _labeldict = f.read()
    _labeldict = ast.literal_eval(_labeldict)
    _labeldict = {v['label']:int(k) 
                        for k, v in _labeldict.items()}
    return _labeldict

def cifar_label_to_idx():
    _labeldict= {
        "airplane": 0,	"automobile": 1,
        "bird": 2,  "cat": 3,										
        "deer": 4,	"dog": 5,									
        "frog": 6,	"horse": 7,									
        "ship": 8,	"truck": 9,
        "random": 10,}
    return _labeldict

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    split: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = osp.expanduser(osp.join(directory, split))

    if class_to_idx is None:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

if __name__ == "__main__":
    class_to_idx('./imagenet_label.txt')