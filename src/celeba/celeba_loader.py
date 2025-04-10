# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/

import imageio
import torch
import numpy as np
import re
import glob

from torch.utils import data
from PIL import Image


class CELEBA(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=(32, 32), transform=None,
                 labels = [], send_image_name=False):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.transform = transform
        self.n_classes =  40
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
       
        self.files = {}
        self.labels = {}
        self.send_image_name = send_image_name

        self.label_file = self.root+"/list_attr_celeba.txt"
        label_map = {}
        with open(self.label_file, 'r') as l_file:
            labels = l_file.read().split('\n')[2:-1]
        for label_line in labels:
            f_name = re.sub('jpg', 'jpg', label_line.split(' ')[0])
            label_txt = list(map(lambda x:int(x), re.sub('-1','0',label_line).split()[1:]))
            label_map[f_name]=label_txt

        self.all_files = glob.glob(self.root+'/img_align_celeba/*.jpg')
        with open(root+'/list_eval_partition.txt', 'r') as f:
            fl = f.read().split('\n')
            fl.pop()
            if 'train' in self.split:
                selected_files = list(filter(lambda x:x.split(' ')[1]=='0', fl))
            elif 'val' in self.split:
                selected_files =  list(filter(lambda x:x.split(' ')[1]=='1', fl))
            elif 'test' in self.split:
                selected_files =  list(filter(lambda x:x.split(' ')[1]=='2', fl))
            selected_file_names = list(map(lambda x:re.sub('jpg', 'jpg', x.split(' ')[0]), selected_files))
        
        base_path = '/'.join(self.all_files[0].split('/')[:-1])
        self.files[self.split] = list(map(lambda x: '/'.join([base_path, x]), set(map(lambda x:x.split('/')[-1], self.all_files)).intersection(set(selected_file_names))))
        self.labels[self.split] = list(map(lambda x: label_map[x], set(map(lambda x:x.split('/')[-1], self.all_files)).intersection(set(selected_file_names))))
        self.class_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                                'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',      
                                'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',       
                                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 
                                'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 
                                'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        if len(self.files[self.split]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))

        print("Found %d %s images" % (len(self.files[self.split]), self.split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        label = self.labels[self.split][index]
        # img = np.asarray(imageio.imread(img_path))
        img = Image.open(img_path).convert('RGB')

        if self.is_transform:
            img = self.transform(img)

        if self.send_image_name == True:            
            img_path = img_path.split("/")
            return img_path[-1], img, torch.Tensor(label)
           
        return img, torch.Tensor(label)

    