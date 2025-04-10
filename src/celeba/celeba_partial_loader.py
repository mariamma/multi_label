# Code adapted from https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/

import imageio
import torch
import numpy as np
import re
import glob

from torch.utils import data
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os


class CELEBA_PARTIAL(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=(32, 32), transform=None,
                 tasks = [], send_image_name=False, evaluate_subset=False):
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
        self.base_path = self.root+'/img_align_celeba/'
        self.send_image_name = send_image_name
        self.evaluate_subset = evaluate_subset

        self.class_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                                'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',      
                                'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',       
                                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 
                                'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 
                                'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
       
        self.file_list = []
        self.labels = []

        df = pd.read_csv(self.root+"/list_attr_celeba.csv",delimiter="\s+", index_col=False)
        print(df.head())
        df = df.replace(-1, 0)
        print(df.head())
        selected_classes  = [self.class_names[i] for i in tasks]
        print("Selected classes : {}".format(selected_classes))
        selected_classes.append("Image")
        df = df[selected_classes]
        print(df.head())

        if self.evaluate_subset == True:
            # fileDescriptor = open(self.root+'/integrated_grad_list_eval_partition.txt', "r")
            fileDescriptor = open(self.root+'/celeba_test10_percent.txt', "r")

            line = True
            # for i in tqdm(range(1032)):
            for i in tqdm(range(2070)):
                line = fileDescriptor.readline()
                rows = df.loc[df['Image'] == line.strip()] 
                rows = rows.drop('Image', axis=1)
                labels = rows.to_numpy()[0]   
                if sum(labels) > 0:        
                    self.file_list.append(line)      
                    self.labels.append(labels)
            print("Found %d %s images" % (len(self.file_list), self.split))
            return        
        
        fileDescriptor = open(self.root+'/list_eval_partition.txt', "r")
        #---- get into the loop
        line = True
        for i in tqdm(range(202599)):
            line = fileDescriptor.readline()
            line = line.strip()
            line = line.split(" ")      
            
            if 'train' in self.split and int(line[1]) == 0:
                rows = df.loc[df['Image'] == line[0]] 
                rows = rows.drop('Image', axis=1)
                labels = rows.to_numpy()[0]   
                if sum(labels) > 0:
                    self.file_list.append(line[0])           
                    self.labels.append(labels)

            elif 'val' in self.split and int(line[1]) == 1:
                rows = df.loc[df['Image'] == line[0]] 
                rows = rows.drop('Image', axis=1)
                labels = rows.to_numpy()[0]   
                if sum(labels) > 0:    
                    self.file_list.append(line[0]) 
                    self.labels.append(labels)  
                    
            elif 'test' in self.split and int(line[1]) == 2:    
                rows = df.loc[df['Image'] == line[0]] 
                rows = rows.drop('Image', axis=1)
                labels = rows.to_numpy()[0]   
                if sum(labels) > 0:        
                    self.file_list.append(line[0])      
                    self.labels.append(labels)

        print("Found %d %s images" % (len(self.file_list), self.split))


    def __len__(self):
        """__len__"""
        return len(self.file_list)


    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.file_list[index].strip()
        label = self.labels[index]
        # img = np.asarray(imageio.imread(img_path))
        img = Image.open(os.path.join(self.base_path,img_path)).convert('RGB')

        if self.is_transform:
            img = self.transform(img)

        if self.send_image_name == True:            
            img_path = img_path.split("/")
            return img_path[-1], img, torch.Tensor(label)    
           
        return img, torch.Tensor(label)

    