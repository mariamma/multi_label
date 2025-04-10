import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from tqdm import tqdm
from utils import ResNet_Multi_Class, DenseNet121_Multi_Class, Inception_Multi_Class, ResNeXt_Multi_Class
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device to use')
    parser.add_argument('--image_path', type=str, default='/scratch/mariamma/celeba/data/celeba/img_align_celeba/', help='model folder')
    

    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='/data/mariammaa/celeba/heatmaps/',
                        help='Output directory to save the images')
    
    parser.add_argument('--net_basename', type=str, default='full_resnext-lr:0.01-wd:0.0_fanciful-smoke-3', help='basename of network (excludes _x_model.pkl)')                        
    parser.add_argument('--model_type', type=str, default='last', help='best or last model', choices=['best', 'last'])
    parser.add_argument('--model_folder', type=str, default='/data/mariammaa/celeba/checkpoints/', help='model folder')
    parser.add_argument('--model_name', type=str, default='resnext', help='model folder')
    parser.add_argument('--label', type=str, default='', help='Label')
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args



def preprocess_image(
    img: np.ndarray, img_size:int, mean=[
        0.5, 0.5, 0.5], std=[
            0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Resize((img_size, img_size)),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def load_saved_model(models, net_basename, folder="saved_models/", name="best"):
    state = torch.load(f"{folder}{net_basename}_{name}_model.pkl")
    print("Epoch : ", state['epoch'])
    models.load_state_dict(state["model_rep"])



if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }

    
    foldername = args.net_basename + args.method + args.label
    print("Foldername : {}".format(foldername))
    os.makedirs(os.path.join(args.output_dir, foldername), exist_ok=True)
    heatmap_dir = os.path.join(args.output_dir, foldername)
    os.makedirs(heatmap_dir, exist_ok=True)

    heatmap_bw_folder           = os.path.join(heatmap_dir, "heatmap_bw/")
    heatmap_rgb_on_xray_folder  = os.path.join(heatmap_dir, "heatmap_rgb_on_xray/")
    if os.path.exists(heatmap_bw_folder):
        shutil.rmtree(heatmap_bw_folder) 
    if os.path.exists(heatmap_rgb_on_xray_folder):
        shutil.rmtree(heatmap_rgb_on_xray_folder)     
    os.makedirs(heatmap_bw_folder)
    os.makedirs(heatmap_rgb_on_xray_folder)

    CLASS_CNT = 40
    IMG_SIZE = 128

    if args.model_name == "resnext":
        model = ResNeXt_Multi_Class(classCount=CLASS_CNT).to(torch.device(args.device)).eval()                                    

    load_saved_model(model, args.net_basename, folder=args.model_folder, name=args.model_type)
    
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
    if args.model_name == "resnet":
        target_layers = [model.resnet.layer4]
    elif args.model_name == "densenet":  
        target_layers = [model.densenet.features[-1]]    
    elif args.model_name == "inception":  
        target_layers = [model.inception.Mixed_7c]         
    elif args.model_name == "resnext":
        target_layers = [model.resnext.layer4]
    # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [ClassifierOutputTarget(281)]
        # targets = [ClassifierOutputTarget(281)]
    targets = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]

    list_images = []

    fileDescriptor = open("/scratch/mariamma/celeba/data/celeba/list_eval_partition.txt", "r")
        
    line = True
    while line:
        line = fileDescriptor.readline()
        line = line.strip()
        if len(line) <= 0:
            break
        line_split = line.split()
        if int(line_split[1]) == 2:
            list_images.append(line_split[0])


    with cam_algorithm(model=model,
                        target_layers=target_layers) as cam:

        for i in tqdm(range(len(list_images))):
            image_path = os.path.join(args.image_path, list_images[i])
            rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, IMG_SIZE,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]).to(args.device)

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            input_tensor = input_tensor.to(args.device)
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)                             

            grayscale_cam_output_path = os.path.join(heatmap_bw_folder, list_images[i])
            cam_output_path = os.path.join(heatmap_rgb_on_xray_folder, list_images[i])                                

            grayscale_cam = grayscale_cam[0, :]
            cam_img = np.uint8(255*grayscale_cam)
            cv2.imwrite(grayscale_cam_output_path, cam_img)
            
            rgb_img = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(cam_output_path, cam_image)
