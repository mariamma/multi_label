import torch
import torch.nn as nn
import logging


def create_logger(name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class DenseNet121_Multi_Class(nn.Module):
    
    def __init__(self, classCount, isTrained=False):
        super(DenseNet121_Multi_Class, self).__init__()
    
        self.densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', weights = 'DenseNet121_Weights.IMAGENET1K_V1')
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, classCount)


    def forward(self, x):
        x = self.densenet(x)
        return x


class Inception_Multi_Class(nn.Module):
    """Model for training densenet baseline"""
    def __init__(self, classCount, isTrained=False):
        super(Inception_Multi_Class, self).__init__()
        self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.IMAGENET1K_V1')
        self.kernelCount = self.inception.fc.in_features
        self.inception.fc = nn.Linear(self.kernelCount, classCount)

    def forward(self, x):
        x = self.inception(x)
        
        if hasattr(x, 'logits'):
            return x.logits
            # x = self.sigmoid(x.logits)
        else:
            # x = self.sigmoid(x)    
            return x


class ResNet_Multi_Class(nn.Module):
    
    def __init__(self, classCount, isTrained=False):
        super(ResNet_Multi_Class, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V1')
        self.kernelCount = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.kernelCount, classCount)

    def forward(self, x):
        x = self.resnet(x)
        return x     



class ResNeXt_Multi_Class(nn.Module):
    
    def __init__(self, classCount, isTrained=False):
        super(ResNeXt_Multi_Class, self).__init__()
        self.resnext = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2')
        self.kernelCount = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(self.kernelCount, classCount)
        

    def forward(self, x):
        x = self.resnext(x)
        return x 


