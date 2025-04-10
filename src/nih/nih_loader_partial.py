import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class NIH_PARTIAL(Dataset):
       
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform, tasks):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        self.tasks = tasks
    
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

                imageLabel_final = [imageLabel[i] for i in tasks]
                if sum(imageLabel_final) > 0:
                    self.listImagePaths.append(imagePath)
                    self.listImageLabels.append(imageLabel_final) 

        fileDescriptor.close()
        print("Added data : {}".format(len(self.listImagePaths)))    
    

    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
                
        if self.transform != None: imageData = self.transform(imageData)
        return imageData, imageLabel


    def __len__(self):
        
        return len(self.listImagePaths)
    
    