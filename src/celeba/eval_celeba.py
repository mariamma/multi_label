import os
import torch
import torchvision

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import DenseNet121_Multi_Class, Inception_Multi_Class, ResNet_Multi_Class, ResNeXt_Multi_Class
from torchvision import transforms
from torch.utils.data import DataLoader
import metrics as metrics
from utils import create_logger
from timeit import default_timer as timer
import argparse
import timm
from celeba_partial_loader import CELEBA_PARTIAL
import numpy as np

threshold = 0.5


def load_saved_model(model, net_basename, folder="saved_models/", name="best"):
    state = torch.load(f"{folder}{net_basename}_{name}_model.pkl")
    print("Epoch : ", state['epoch'])
    model.load_state_dict(state["model_rep"])
    return model

    
def topk_attack(model, output, image, label, k=50, epsilon=0.1, device='cuda'):
    
    gradient_orig = torch.autograd.grad(torch.max(output), image, retain_graph=True)
    # print("gradient_orig : ", gradient_orig[0].shape)
    grad = torch.mean(gradient_orig[0], dim=(0,1)) #grad.mean(dim=1, keepdim=True)  # Average over RGB channels
    
    # Flatten and get top-k indices
    flat_grad = grad.view(-1)
    topk_vals, topk_indices = torch.topk(flat_grad, k)
    # print("topk_vals, topk_indices :", topk_vals, topk_indices)
    
    perturbed_image = image.clone().detach()
    perturbed_image = perturbed_image.view(-1)
    perturbed_image[topk_indices] += epsilon * torch.sign(topk_vals)
    perturbed_image = perturbed_image.view(image.shape)
    return perturbed_image    


def train(args, debug = False):
    """Training Function"""
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)
    
    IMG_SIZE = 128  

    tasks = args.tasks.split('_')
    tasks = [int(x) for x in tasks]
    CLASS_CNT = len(tasks)

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


    model = load_saved_model(model, args.net_basename, folder=args.model_folder, name=args.model_type)
    model.to(device)

    celeba_val_data = CELEBA_PARTIAL(root = args.root, split = "test",
                            is_transform=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                    tasks = tasks, send_image_name = True
    )
    
    
    dataLoaderVal = DataLoader(dataset = celeba_val_data,
                                    batch_size = args.val_batch_size,
                                    shuffle = True,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=True)
     
    AV_ij = torch.zeros(CLASS_CNT, CLASS_CNT).to(device=device)
    AV = torch.zeros(CLASS_CNT).to(device=device)

    vulnerability = torch.zeros(CLASS_CNT).to(device=device)
    rho_score = torch.zeros(CLASS_CNT, CLASS_CNT)
    train_batch_size = 0
    
    
    # model.eval()
    for batch_no, (img_name, images, labels) in enumerate(tqdm(dataLoaderVal)):
        images = images.to(device)
        labels = labels.to(device)
        images.requires_grad = True            

        outputs = model(images)
        
        gradient_orig = []
        for i in range(CLASS_CNT):
            gradient_orig_i = torch.autograd.grad(outputs[0][i], images, retain_graph=True)
            gradient_orig.append(gradient_orig_i[0])

        # Perform Top-K attack
        # perturbed_image = topk_attack(model, outputs, images, labels, k=100, epsilon=0.05, device=device)
        perturbed_image = np.load(os.path.join(args.image_save_dir,img_name[0]+".npy"))
        perturbed_image = torch.from_numpy(perturbed_image)
        perturbed_image = perturbed_image.to(device)
        perturbed_image.requires_grad = True   
        outputs_pert = model(perturbed_image)

        gradient_pert = []
        for i in range(CLASS_CNT):
            gradient_pert_i = torch.autograd.grad(outputs_pert[0][i], perturbed_image, retain_graph=True)
            gradient_pert.append(gradient_pert_i[0])
   
        for i in range(CLASS_CNT):
            u_i = gradient_orig[i] - gradient_pert[i]
            for j in range(i, CLASS_CNT):
                u_j = gradient_orig[j] - gradient_pert[j]
                AV_ij[i][j] += torch.dot(torch.flatten(u_i), torch.flatten(u_j))
                AV_ij[j][i] += torch.dot(torch.flatten(u_i), torch.flatten(u_j))
            AV[i] += torch.norm(u_i)
            vulnerability[i] += torch.norm(u_i)/torch.norm(gradient_orig[i])

        train_batch_size += 1    
            
    for i in range(CLASS_CNT):
        for j in range(i, CLASS_CNT):        
            rho_score[i][j] = AV_ij[i][j]/(AV[i] * AV[j])
            rho_score[j][i] = AV_ij[j][i]/(AV[i] * AV[j])
        vulnerability[i] = vulnerability[i]/train_batch_size    

    rho_score = rho_score.cpu().detach().numpy()
    filename = "rho_score_" + args.label + args.net_basename + ".csv"
    np.savetxt(os.path.join(args.output_dir, filename), rho_score, delimiter=",")

    vulnerability = vulnerability.cpu().detach().numpy()
    filename = "vulnerability_" + args.label + args.net_basename + ".csv"
    np.savetxt(os.path.join(args.output_dir, filename), vulnerability, delimiter=",")

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/scratch/mariamma/celeba/data/celeba/', help='root dataset dir')
    parser.add_argument('--model', type=str, default='resnext', help='model to train')
    parser.add_argument('--label', type=str, default='', help='wandb group')
    
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--debug', action='store_true', help='Debug mode: disables wandb.')
   
    parser.add_argument('--model_folder', type=str, default="/data/mariammaa/celeba/checkpoints/", help='model folder to save the model')
    parser.add_argument('--model_type', type=str, default='last', help='best or last model', choices=['best', 'last'])
    parser.add_argument('--net_basename', type=str, default='full_resnext-lr:0.01-wd:0.0_fanciful-smoke-3', help='basename of network (excludes _x_model.pkl)')                        

    parser.add_argument('--output_dir', type=str, default="/data/mariammaa/celeba/results/", help=' folder to save the results')
    parser.add_argument('--image_save_dir', type=str, default="/data/mariammaa/celeba/perturbed_testset/", help=' folder to save the results')
    parser.add_argument('--tasks', type=str, default="", help=' tasks')
    args = parser.parse_args()
    

    train(args, args.debug)
