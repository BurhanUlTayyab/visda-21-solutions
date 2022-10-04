from typing import Optional, Callable, Tuple, Any, List
import os
import cv2
import ast
import sys
import timm
import torch
import argparse
import warnings
import numpy as np
import torch.nn.functional as F

sys.path.append('../')
warnings.filterwarnings('ignore')

from PIL import Image
from utils import utils
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from timm.models.helpers import load_checkpoint, load_state_dict

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('--test_dir', '-sd', metavar='SRC DIR',
                    help='path to test data directory (default ./test_data)')
parser.add_argument('--result_dir', '-dd', metavar='DEST DIR', default='./',
                    help='path where result should be stored ')
parser.add_argument('--result_fname', metavar='NAME', default='1000',
                    help='file where results would be stored.')
parser.add_argument('--checkpoint', required=True, metavar='CHECKPOINT',
                    help='model checkpoint file.')


class TestDataset():
    def __init__(self, 
                    testdir: str='./test_data/'):
        super(TestDataset, self).__init__()
        self.base_folder = testdir # Test Data folder
        filename = './test_example_submission/adapt_pred.txt' # Sample test submission folder
        filename = utils.validate_path(filename)

        with open(filename, 'r') as f:
            val_files = f.read().splitlines()
        self.val_files = [x.split(' ')[0] for x in val_files]
        self.tfms = transforms.Compose([
                                transforms.Resize((608, 608)), 
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225]),
                                ])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        image_file = self.val_files[index]
        img = utils.pil_loader(os.path.join(self.base_folder,
                        image_file))
        img = self.tfms(img)
        return img, image_file


    def __len__(self) -> int:
        return len(self.val_files)
    
    def __getlables__(self) -> List[int]:
        return np.array([x[1] for x in self.val_files])

def collate_fn(batch):
    img_file = []
    imgs = []
    labels = []
    for _batch in batch:
        imgs.append(_batch[0])
        img_file.append(_batch[-1])
    imgs = torch.stack(imgs)
    return imgs, img_file

def test(checkpoint_path, 
            outputfname,
            testdir):
    td = TestDataset(testdir)
    if os.path.exists(outputfname):
        os.remove(outputfname)
    output_file = Path(outputfname)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    model = timm.create_model('tf_efficientnet_b7_ns',
                    pretrained=False, num_classes=1000)
    load_checkpoint(model, checkpoint_path)
    model = model.to('cuda')
    dl = torch.utils.data.DataLoader(td, 
                                        batch_size=1, 
                                        shuffle=False, 
                                        pin_memory=False, 
                                        num_workers=4,
                                        collate_fn=collate_fn,
                                        drop_last=False)
    
    model.eval()
    with open(outputfname, 'w') as f:
        f.close()
    with torch.no_grad() and open(outputfname, 'a') as f:
        # labels = []
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), 
                    desc = "Testing the model"):
            torch.cuda.empty_cache()
            img, image_file = batch
            img = img.to('cuda')
            outputs = model(img)
            outputs = outputs.detach().cpu()
            pred = outputs.data.max(1)[1]
            logit_t = outputs
            outputs = F.softmax(outputs)
            pred_unk = -torch.max(logit_t, dim=-1)[0]
            anomaly_score = pred_unk.data.numpy()
            for i in range(len(image_file)):
                f.write(f'{image_file[i]} {pred[i].item()} {anomaly_score[i]}\n')
        f.close()

def main():
    args = parser.parse_args()
    outputfname = os.path.join(args.result_dir, args.result_fname)
    test(outputfname, args.test_dir)

if __name__ == '__main__':
    main() 