import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class NIH(Dataset):
       
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform, send_imagename=False):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        self.send_imagename = send_imagename
    
        fileDescriptor = open(pathDatasetFile, "r")
        line = True

        while line:        
            line = fileDescriptor.readline()
            
            if line:
                lineItems = line.split()
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                if sum(imageLabel) > 0:
                    imageLabel.append(0)
                else:
                    imageLabel.append(1)    
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel) 
            
        fileDescriptor.close()
    

    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
                
        if self.transform != None: imageData = self.transform(imageData)

        if self.send_imagename ==True:
            image_name = imagePath.split("/")
            return image_name[-1], imageData, imageLabel
        return imageData, imageLabel


    def __len__(self):
        
        return len(self.listImagePaths)
    
    