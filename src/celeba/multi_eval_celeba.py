import os
import torch

from tqdm import tqdm
from utils import DenseNet121_Multi_Class, Inception_Multi_Class, ResNet_Multi_Class, ResNeXt_Multi_Class
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
# import timm
from celeba_partial_loader import CELEBA_PARTIAL
import numpy as np
from captum.attr import IntegratedGradients, LayerGradCam, DeepLift, GuidedBackprop, InputXGradient, GuidedGradCam

threshold = 0.5


def load_saved_model(model, net_basename, folder="saved_models/", name="best"):
    state = torch.load(f"{folder}{net_basename}_{name}_model.pkl")
    print("Epoch : ", state['epoch'])
    model.load_state_dict(state["model_rep"])
    return model 


def get_gradcam(layer_gradcam, input_img, pred_label_idx):
    attributions_lgc = layer_gradcam.attribute(input_img, target=pred_label_idx)
    return attributions_lgc


def get_integrated_gradients(integrated_gradients, input_img, pred_label_idx):
    # Ask the algorithm to attribute our output target to
    attributions_ig = integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)
    return attributions_ig


def get_deeplift(deeplift, input_img, pred_label_idx):
    # Ask the algorithm to attribute our output target to
    attributions_dl = deeplift.attribute(input_img, target=pred_label_idx)
    return attributions_dl


def get_guidedbackprop(guided_backprop, input_img, pred_label_idx):
    # Ask the algorithm to attribute our output target to
    attributions_gb = guided_backprop.attribute(input_img, target=pred_label_idx)
    return attributions_gb


def get_inputxgradient(input_x_gradient, input_img, pred_label_idx):
    # Ask the algorithm to attribute our output target to
    attributions_ixg = input_x_gradient.attribute(input_img, target=pred_label_idx)
    return attributions_ixg    

def get_guidedgradcam(guided_gradcam, input_img, pred_label_idx):
    # Ask the algorithm to attribute our output target to
    attributions_gg = guided_gradcam.attribute(input_img, target=pred_label_idx)
    return attributions_gg   
   

def get_results_dir(output_dir, sal_method, attack):
    dirname = "results_" + attack + "_" + sal_method.replace("_","")
    return os.path.join(output_dir, dirname)
    

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
    # elif args.model == "xception":
    #     model = timm.create_model('xception', pretrained=True, num_classes=CLASS_CNT)
    # elif args.model== "vgg":
    #     model = timm.create_model('vgg19', pretrained=True, num_classes = CLASS_CNT)


    model = load_saved_model(model, args.net_basename, folder=args.model_folder, name=args.model_type)
    model.to(device)

    celeba_val_data = CELEBA_PARTIAL(root = args.root, split = "test",
                            is_transform=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                    tasks = tasks, send_image_name = True,
                    evaluate_subset = args.evaluate_subset)
    
    dataLoaderVal = DataLoader(dataset = celeba_val_data,
                                    batch_size = args.val_batch_size,
                                    shuffle = False,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=False)
    
    if args.sal_method == "integrated_gradients":
        integrated_gradients = IntegratedGradients(model)
    elif args.sal_method == "gradcam":    
        layer_gradcam = LayerGradCam(model, model.resnext.layer4)
    elif args.sal_method == "deeplift":
        deeplift = DeepLift(model)
    elif args.sal_method == "guidedBackprop":
        guided_backprop = GuidedBackprop(model=model)    
    elif args.sal_method == "input_x_gradient":    
        input_x_gradient = InputXGradient(model)
    elif args.sal_method == "guided_gradcam":
        guided_gradcam = GuidedGradCam(model, model.resnext.layer4) 

    AV_ij = torch.zeros(CLASS_CNT, CLASS_CNT).to(device=device)
    AV = torch.zeros(CLASS_CNT).to(device=device)

    vulnerability = torch.zeros(CLASS_CNT).to(device=device)
    rho_score = torch.zeros(CLASS_CNT, CLASS_CNT)
    train_batch_size = 0
    results_dir = args.results_dir
    print("Results dir :", results_dir)
    
    # model.eval()
    for batch_no, (img_name, images, labels) in enumerate(tqdm(dataLoaderVal)):
        images = images.to(device)
        labels = labels.to(device)
        images.requires_grad = True            
        
        gradient_orig = []
        if args.sal_method == "saliency_map":
            for i in range(CLASS_CNT):
                outputs = model(images)
                gradient_orig_i = torch.autograd.grad(outputs[0][i], images, retain_graph=True)
                gradient_orig.append(gradient_orig_i[0].cpu().detach())
        elif args.sal_method == "integrated_gradients":
            for i in range(CLASS_CNT):
                gradient_orig_i = get_integrated_gradients(integrated_gradients, images, [i])
                gradient_orig.append(gradient_orig_i[0].cpu().detach())
        elif args.sal_method == "gradcam":    
            for i in range(CLASS_CNT):
                gradient_orig_i = get_gradcam(layer_gradcam, images, [i])
                gradient_orig.append(gradient_orig_i[0].cpu().detach())
        elif args.sal_method == "deeplift":
            for i in range(CLASS_CNT):
                gradient_orig_i = get_deeplift(deeplift, images, [i])
                gradient_orig.append(gradient_orig_i[0].cpu().detach())
        elif args.sal_method == "guidedBackprop":        
            for i in range(CLASS_CNT):
                gradient_orig_i = get_guidedbackprop(guided_backprop, images, [i])
                gradient_orig.append(gradient_orig_i[0].cpu().detach())
        elif args.sal_method == "input_x_gradient":          
            for i in range(CLASS_CNT):
                gradient_orig_i = get_inputxgradient(input_x_gradient, images, [i])
                gradient_orig.append(gradient_orig_i[0].cpu().detach())
        elif args.sal_method == "guided_gradcam":        
            for i in range(CLASS_CNT):
                gradient_orig_i = get_guidedgradcam(guided_gradcam, images, [i])
                gradient_orig.append(gradient_orig_i[0].cpu().detach())

        # Perform Top-K attack
        # perturbed_image = topk_attack(model, outputs, images, labels, k=100, epsilon=0.05, device=device)
        perturbed_image = np.load(os.path.join(args.image_save_dir,img_name[0].replace(".jpg","") + ".npy"))
        perturbed_image = torch.from_numpy(perturbed_image)
        perturbed_image = perturbed_image.to(device)
        perturbed_image.requires_grad = True   

        gradient_pert = []
        if args.sal_method == "saliency_map":
            for i in range(CLASS_CNT):
                outputs_pert = model(perturbed_image)
                gradient_pert_i = torch.autograd.grad(outputs_pert[0][i], perturbed_image, retain_graph=True)
                gradient_pert.append(gradient_pert_i[0].cpu().detach())
        elif args.sal_method == "integrated_gradients":
            for i in range(CLASS_CNT):
                gradient_pert_i = get_integrated_gradients(integrated_gradients, perturbed_image, [i])
                gradient_pert.append(gradient_pert_i[0].cpu().detach())
                
        elif args.sal_method == "gradcam":    
            for i in range(CLASS_CNT):
                gradient_pert_i = get_gradcam(layer_gradcam, perturbed_image, [i])
                gradient_pert.append(gradient_pert_i[0].cpu().detach())
        elif args.sal_method == "deeplift":
            for i in range(CLASS_CNT):
                gradient_pert_i = get_deeplift(deeplift, perturbed_image, [i])
                gradient_pert.append(gradient_pert_i[0].cpu().detach())        
        elif args.sal_method == "guidedBackprop":        
            for i in range(CLASS_CNT):
                gradient_pert_i = get_guidedbackprop(guided_backprop, perturbed_image, [i])
                gradient_pert.append(gradient_pert_i[0].cpu().detach())
        elif args.sal_method == "input_x_gradient":          
            for i in range(CLASS_CNT):        
                gradient_pert_i = get_inputxgradient(input_x_gradient, perturbed_image, [i])
                gradient_pert.append(gradient_pert_i[0].cpu().detach())
        elif args.sal_method == "guided_gradcam":        
            for i in range(CLASS_CNT):
                gradient_pert_i = get_guidedgradcam(guided_gradcam, perturbed_image, [i])
                gradient_pert.append(gradient_pert_i[0].cpu().detach())
        
                    
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
    np.savetxt(os.path.join(results_dir, filename), rho_score, delimiter=",")

    vulnerability = vulnerability.cpu().detach().numpy()
    filename = "vulnerability_" + args.label + args.net_basename + ".csv"
    np.savetxt(os.path.join(results_dir, filename), vulnerability, delimiter=",")

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/scratch/mariamma/celeba/data/celeba/', help='root dataset dir')
    parser.add_argument('--model', type=str, default='resnext', help='model to train')
    parser.add_argument('--label', type=str, default='', help='wandb group')
    
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--debug', action='store_true', help='Debug mode: disables wandb.')
   
    # parser.add_argument('--model_folder', type=str, default="/data/mariammaa/celeba/checkpoints_saliency/", help='model folder to save the model')
    parser.add_argument('--model_folder', type=str, default="/data/mariammaa/celeba/checkpoints_manipulate/", help='model folder to save the model')
    parser.add_argument('--model_type', type=str, default='last', help='best or last model', choices=['best', 'last'])
    parser.add_argument('--net_basename', type=str, default='full_resnext-lr:0.01-wd:0.0_fanciful-smoke-3', help='basename of network (excludes _x_model.pkl)')                        

    parser.add_argument('--output_dir', type=str, default="/data/mariammaa/celeba/", help=' folder to save the results')
    # parser.add_argument('--image_save_dir', type=str, default="/data/mariammaa/celeba/perturbed_testset/", help=' folder to save the results')
    parser.add_argument('--image_save_dir', type=str, default="/data/mariammaa/celeba/perturbed_test_manipulate_set2//", help=' folder to save the results')
    parser.add_argument('--tasks', type=str, default="", help=' tasks')

    parser.add_argument('--sal_method', type=str, default="integrated_gradients", help=' tasks',
                        choices=['integrated_gradients','saliency_map','gradcam','deeplift', 'guidedBackprop', 'input_x_gradient', 'guided_gradcam'])
    parser.add_argument('--evaluate_subset', default=False, action="store_true", help='whether to evaluate subset')    
    parser.add_argument('--results_dir', type=str, default="topk", help=' attack_name')
    args = parser.parse_args()
    

    train(args, args.debug)
