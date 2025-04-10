import os
import torch
import torchvision

import torch.nn as nn
from tqdm import tqdm
from utils import DenseNet121_Multi_Class, Inception_Multi_Class, ResNet_Multi_Class, ResNeXt_Multi_Class
from torchvision import transforms
from torch.utils.data import DataLoader
import metrics as metrics
import argparse
import timm

import numpy as np
import torch.nn.functional as F

from nih_loader import NIH
import pandas as pd

threshold = 0.5
sigmoid = nn.Sigmoid()

def load_saved_model(model, net_basename, folder="saved_models/", name="best"):
    state = torch.load(f"{folder}{net_basename}_{name}_model.pkl")
    print("Epoch : ", state['epoch'])
    model.load_state_dict(state["model_rep"])
    return model

    
def topk_attack(model, image, label, k=50, epsilon=0.1, device='cuda',
        num_iter=100, eps = 10):
    image.requires_grad = True
    original_logit = model(image)
    original_pred = (sigmoid(original_logit) >= threshold).to(torch.float32)
    perturbed_image = image.clone().detach()
    
    for i in range(num_iter):
        image = perturbed_image.clone().detach()
        image.requires_grad = True
        output = model(image)
        
        gradient_orig = torch.autograd.grad(torch.sum(output), image, retain_graph=True)
        grad = torch.mean(gradient_orig[0], dim=(0,1)) #grad.mean(dim=1, keepdim=True)  # Average over RGB channels

        # Flatten and get top-k indices
        flat_grad = grad.view(-1)
        topk_vals, topk_indices = torch.topk(flat_grad, k)
        
        perturbed_image = image.clone().detach()
        perturbed_image = perturbed_image.view(-1)
        perturbed_image[topk_indices] += epsilon * torch.sign(topk_vals)
        perturbed_image = perturbed_image.view(image.shape)

        # if torch.norm(image-perturbed_image) > eps:
        #     print("Iter  norm:", i)
        #     return image
        
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        pert_pred = (sigmoid(output) >= threshold).to(torch.float32)
        if not torch.equal(original_pred, pert_pred):
            return image, i
    return perturbed_image, num_iter    


def manipulate_attack(model, images, labels, criterion, max_iter=100, max_epsilon=2, device='cuda'):
    eps = max_epsilon / 255.0
    images.requires_grad = True
    original_logit = model(images)
    original_pred = (sigmoid(original_logit) >= threshold).to(torch.float32)

    target_map = torch.autograd.grad(torch.sum(original_logit), images, create_graph=True)[0] * images
    target_map = target_map.squeeze().detach()
   
    # combine color channel; normalized into (0,1) and scale by image size; flatten saliency map into 1D 
    normalized_target_map = torch.sum(torch.abs(target_map),0)
    # normalized_target_map = 224*224*normalized_target_map/torch.sum(normalized_target_map)
    best_adv_image = images.clone()
    best_adv_norm = 0.0

    adv_image = images.clone()
    for i in range(max_iter):
        adv_image = adv_image.clone().detach().requires_grad_(True)
        adv_image.to(device)
        # get the saliency map of current adversarial image
        logit = model(adv_image)
        
        saliency = torch.autograd.grad(torch.sum(logit), adv_image, create_graph=True)[0] * adv_image
        saliency = saliency.squeeze()

        # normalize the saliency map
        saliency = torch.sum(torch.abs(saliency),0)
        # print("Saliency : ", saliency.shape)
        # saliency = 224*224*saliency/torch.sum(saliency)

        # print(f"Labels: {labels[0][label_index]}, Logit: {logit[0][label_index]}")
        loss_expl = F.mse_loss(saliency, normalized_target_map)
        loss_output = criterion(labels, logit)
        total_loss = loss_expl
        # print("Loss expl:{}, loss output:{}".format(loss_expl, loss_output))
        # print("Total loss :", total_loss)
        grad = torch.autograd.grad(total_loss, adv_image)[0]
        grad_sign = - torch.sign(grad)
        # adv_image = adv_image + torch.clamp(adv_image+ 2*grad_sign - images, -eps, eps)
   
        adv_image = adv_image + .0005*grad_sign + (adv_image - images)
        
        output = model(adv_image)
        pert_pred = (sigmoid(output) >= threshold).to(torch.float32)
        # print("Original image norm :{}, adv images norm:{}".format( torch.norm(images), torch.norm(adv_image)))
        if torch.equal(original_pred, pert_pred):
            norm_diff = torch.norm(images-adv_image)
            if  norm_diff <= max_epsilon:
                if norm_diff >= best_adv_norm:
                    best_adv_image = adv_image.clone().detach()
                    best_adv_norm = norm_diff
                    print("Iter", i)
                   
    return best_adv_image


def fgsm_attack(model, images, labels, eps=8 / 255, device = 'cuda') :
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.BCEWithLogitsLoss() 

    images.requires_grad = True
    outputs = model(images)
    model.zero_grad()
           
    cost = loss(outputs, labels).to(device)
    # cost.backward()
    grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

    adv_images = images + eps * grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()  
    return adv_images


def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, iters=40, device = 'cuda') :
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.BCEWithLogitsLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        cost = loss(outputs[0], labels[0])
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    return images


def train(args, debug = False):
    """Training Function"""
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    nih_classes = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'Normal']
    CLASS_CNT = len(nih_classes)
    IMG_SIZE = 256
    tasks = [int(x) for x in range(CLASS_CNT)]
    print("Tasks : ", tasks)
    
    if args.model== "densenet":
        model = DenseNet121_Multi_Class(classCount=CLASS_CNT)
    elif args.model== "inception":
        model = Inception_Multi_Class(classCount=CLASS_CNT)
    elif args.model== "resnet":
        model = ResNet_Multi_Class(classCount=CLASS_CNT)    
    elif args.model== "resnext":
        model = ResNeXt_Multi_Class(classCount=CLASS_CNT)            
    elif args.model == "xception":
        model = timm.create_model('xception', pretrained=True, num_classes=CLASS_CNT)
    elif args.model== "vgg":
        model = timm.create_model('vgg19', pretrained=True, num_classes = CLASS_CNT)

    criterion = nn.BCELoss()
    model = load_saved_model(model, args.net_basename, folder=args.model_folder, name=args.model_type)
    model.to(device)

   
    nih_pathFileVal = "/scratch/mariamma/xraysetu/dataset/val_1.txt"

    nih_val_data = NIH(args.root, nih_pathFileVal,
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                    send_imagename=True )

    dataLoaderVal = DataLoader(dataset = nih_val_data,
                                    batch_size = args.val_batch_size,
                                    shuffle = False,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=True)
    
    df = pd.DataFrame(columns=["imagename", "iter"])
    # model.eval()
    for batch_no, (img_name, images, labels) in enumerate(tqdm(dataLoaderVal)):
        images = images.to(device)
        labels = labels.to(device)
        
        # Perform Top-K attack
        if args.attack == "topk":
            row = {}
            perturbed_image, index = topk_attack(model, images, labels, k=100, epsilon=2, device=device)
            perturbed_image = perturbed_image.cpu().detach().numpy()
            np.save(os.path.join(args.image_save_dir, img_name[0].replace(".png","")), perturbed_image) 
            row["imagename"] = img_name[0].replace(".png","")
            row["iter"] = index
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        elif args.attack == "manipulate":
            perturbed_image = manipulate_attack(model, images, labels, criterion, max_iter=30, max_epsilon=6, device=device)
            image = perturbed_image.cpu().detach().numpy()
            image_name_new = img_name[0].replace(".png","")
            np.save(os.path.join(args.image_save_dir, image_name_new), image) 
        
        elif args.attack == "pgd":
            images.requires_grad = True
            perturbed_image = pgd_attack(model, images, labels, device = device)
            perturbed_image = perturbed_image.cpu().detach().numpy()
            np.save(os.path.join(args.image_save_dir, img_name[0].replace(".png","")), perturbed_image) 

        elif args.attack == "fgsm":
            images.requires_grad = True
            perturbed_image = fgsm_attack(model, images, labels, device = device)
            perturbed_image = perturbed_image.cpu().detach().numpy()
            np.save(os.path.join(args.image_save_dir, img_name[0].replace(".png","")), perturbed_image)     
            
    if args.attack == "topk":
        dirname = "/data/mariammaa/nih_multi_label/"
        df.to_csv(os.path.join(dirname, "topk_image_iter.csv"))        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/scratch/mariamma/xraysetu/dataset/images/', help='root dataset dir')
    parser.add_argument('--model', type=str, default='resnext', help='model to train')
    # parser.add_argument('--label', type=str, default='full', help='wandb group')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--debug', action='store_true', help='Debug mode: disables wandb.')
   
    parser.add_argument('--model_folder', type=str, default="/data/mariammaa/nih_multi_label/checkpoints/", help='model folder to save the model')
    parser.add_argument('--model_type', type=str, default='last', help='best or last model', choices=['best', 'last'])
    parser.add_argument('--net_basename', type=str, default='full_full_resnext-lr:0.01-wd:0.0_generous-sea-6', help='basename of network (excludes _x_model.pkl)')                        

    # parser.add_argument('--output_dir', type=str, default="/data/mariammaa/celeba/results/", help=' folder to save the results')
    parser.add_argument('--image_save_dir', type=str, default="/data/mariammaa/nih_multi_label/perturbed_nih_topk/", help=' folder to save the results')
    parser.add_argument('--attack', type=str, default="topk", help='attack type')
    parser.add_argument('--evaluate_subset', default=False, action="store_true", help='whether to evaluate subset')    
    args = parser.parse_args()
    

    train(args, args.debug)
