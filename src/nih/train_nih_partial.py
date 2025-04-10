import os
import torch
import torchvision

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import DenseNet121_Multi_Class, Inception_Multi_Class, ResNet_Multi_Class, ResNeXt_Multi_Class
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
import metrics as metrics
from utils import create_logger
from timeit import default_timer as timer
import argparse
import timm
from nih_loader_partial import NIH_PARTIAL

#Hyperparameters
# lr = 2e-4

threshold = 0.5

def save_model(models, optimizer, scheduler, epoch, args, folder="saved_models/", name="best"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    state = {'epoch': epoch + 1,
             'model_rep': models.state_dict(),
             'optimizer_state': optimizer.state_dict(),
             'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
             'args': vars(args)}

    run_name = "debug" if args.debug else wandb.run.name
    torch.save(state, f"{folder}{args.label}_{run_name}_{name}_model.pkl")


def train(args, debug = False):
    """Training Function"""
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    nih_classes = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'Normal']
    CLASS_CNT = len(nih_classes)
    IMG_SIZE = 256
    tasks = args.tasks
    CLASS_CNT = len(tasks)
    tasks = [int(x) for x in tasks]
    print(tasks)

    logger = create_logger('Main')  

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


    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)
    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.7, patience=5)
    scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True, factor=0.2)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    criterion = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    if not debug:
        wandb.init(project="multi_label_nih", group=args.label, config=args, reinit=True)
        dropout_str = "" if not args.dropout else "-dropout"
        task_str = "_".join(args.tasks)
        wandb.run.name = f"{args.cluster_label}-{task_str}-{args.model}{dropout_str}-lr:{args.lr}-wd:{args.weight_decay}_" + wandb.run.name

    metric, aggregators, model_saver = metrics.get_metrics(args.metric_arg, tasks)

    model_storage = args.model_storage 

    nih_pathFileTrain = "/scratch/mariamma/xraysetu/dataset/train_1.txt"
    nih_pathFileVal = "/scratch/mariamma/xraysetu/dataset/val_1.txt"


    nih_train_data = NIH_PARTIAL(args.root, nih_pathFileTrain,
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                    tasks = tasks
    )

    nih_val_data = NIH_PARTIAL(args.root, nih_pathFileVal,
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                    tasks = tasks
    )
    
    dataLoaderTrain = DataLoader(dataset = nih_train_data,
                                    batch_size = args.train_batch_size,
                                    shuffle = True,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=True)
    
    dataLoaderVal = DataLoader(dataset = nih_val_data,
                                    batch_size = args.val_batch_size,
                                    shuffle = True,
                                    num_workers = 12,
                                    pin_memory = True,
                                    drop_last=True)
         
    best_result = {k: -float("inf") for k in model_saver}  # Something to maximize.

    iter_epochs = tqdm(range(args.num_epochs))
    for epoch_num in iter_epochs:
        train_losses = 0
        valid_losses = 0
        train_correct = 0
        valid_correct = 0
        len_train_batchloader = 0
        len_val_batchloader = 0
        # training mode
        model.train()
        for batch_no, (images, labels) in enumerate(tqdm(dataLoaderTrain)):
            
            start = timer()
            images = images.to(device)
            labels = labels.to(device)
            # zeroing the optimizer
            optimizer.zero_grad()
            
            outputs = model(images)
            
            outputs = sigmoid(outputs)

            prediction = (outputs >= threshold).to(torch.float32)
        
            loss = criterion(outputs, labels)
            train_losses += loss.item()
            # calculating the gradients
            loss.backward()
            optimizer.step()

            # Correct predictions
            train_correct += (prediction == labels).sum()
            len_train_batchloader += 1

        train_accuracy = train_correct.item() / len_train_batchloader
        train_loss = train_losses / len_train_batchloader

        model.eval()
        for batch_no, (images, labels) in enumerate(tqdm(dataLoaderVal)):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            outputs = sigmoid(outputs)
            prediction = (outputs >= threshold).to(torch.float32)
            loss = criterion(outputs, labels)
            
            valid_losses += loss.item()
            # Correct predictions 
            valid_correct += (prediction == labels).sum()
            
            for idx, t in enumerate(tasks):
                metric[t].update(outputs[:,idx], labels[:,idx])
            len_val_batchloader += 1    

        valid_accuracy = valid_correct.item() / len_val_batchloader
        valid_loss = valid_losses / len_val_batchloader

        iter_epochs.set_description(desc = 'Train Loss {} Validation : Loss {}, Accuracy {}'.format(train_loss, valid_loss, valid_accuracy))
        
        scheduler.step(valid_loss)
       
        epoch_stats = {}
        epoch_stats['train_loss'] = train_loss
        epoch_stats['validation_loss'] = valid_loss
        # Print the stored (averaged across batches) validation losses and metrics, per task.
        clog = "epochs {}/{}:".format(epoch_num, args.num_epochs)
        metric_results = {}
        for t in tasks:
            metric_results[t] = metric[t].get_result()
            metric[t].reset()
            for metric_key in metric_results[t]:
                clog += ' val metric-{} {} = {:5.4f}'.format(metric_key, t, metric_results[t][metric_key])
            clog += " |||"

        # Store aggregator metrics (e.g., avg) as well
        for agg_key in aggregators:
            clog += ' val metric-{} = {:5.4f}'.format(agg_key, aggregators["avg"](metric_results))

        logger.info(clog)
        for i, t in enumerate(tasks):
            for metric_key in metric_results[t]:
                epoch_stats[f"val_metric_{metric_key}_{t}"] = metric_results[t][metric_key]

        # Store aggregator metrics (e.g., avg) as well
        for agg_key in aggregators:
            epoch_stats[f"val_metric_{agg_key}"] = aggregators[agg_key](metric_results)
        if not args.debug:
            wandb.log(epoch_stats, step=epoch_num)

        # Any time one of the model_saver metrics is improved upon, store a corresponding model.
        c_saver_metric = {k: model_saver[k](metric_results) for k in model_saver}
        for k in c_saver_metric:
            if c_saver_metric[k] >= best_result[k]:
                best_result[k] = c_saver_metric[k]
                # Evaluate the model on the test set and store relative results.
                # test_evaluator(args, test_loader, tasks, DEVICE, model, loss_fn, metric, aggregators, logger, k, epoch)
                if args.store_models:
                    # Save (overwriting) any model that improves the average metric
                    save_model(model, optimizer, scheduler, epoch_num, args,
                               folder=model_storage, name=k)

        end = timer()
        print('Epoch ended in {}s'.format(end - start))

    # Save training/validation results.
    if args.store_models and (not args.time_measurement_exp):
        # Save last model.
        save_model(model, optimizer, scheduler, epoch_num, args, folder=model_storage,
                   name="last")    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/scratch/mariamma/xraysetu/dataset/images/', help='root dataset dir')
    parser.add_argument('--model', type=str, default='resnext', help='model to train')
    parser.add_argument('--label', type=str, default='full', help='wandb group')
    parser.add_argument('--metric_arg', type=str, default='nih', help='which dataset to use',
                        choices=['nih'])
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--p', type=float, default=0.1, help='Task dropout probability')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--val_batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=40, help='Epochs to train for.')
    
    parser.add_argument('--debug', action='store_true', help='Debug mode: disables wandb.')
    parser.add_argument('--store_models', action='store_true', help='Whether to store  models at fixed frequency.')
    parser.add_argument('--decay_lr', action='store_true', help='Whether to decay the lr with the epochs.')
    parser.add_argument('--dropout', action='store_true', help='Whether to use additional dropout in training.')
    parser.add_argument('--no_dropout', action='store_true', help='Whether to not use dropout at all.')
    parser.add_argument('--store_convergence_stats', action='store_true',
                        help='Whether to store the squared norm of the unitary scalarization gradient at that point')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization.')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of experiment repetitions.')
    parser.add_argument('--random_seed', type=int, default=1, help='Start random seed to employ for the run.')
    
    parser.add_argument('--baseline_losses_weights', type=int, nargs='+',
                        help='Weights to use for losses. Be sure that the ordering is correct! (ordering defined as in config for losses.')
    parser.add_argument('--time_measurement_exp', action='store_true',
                        help="whether to only measure time (does not log training/validation losses/metrics)")

    parser.add_argument('--model_storage', type=str, default="/data/mariammaa/nih_multi_label/checkpoint_nih_topk/", help='model folder to save the model')
    parser.add_argument('--tasks', nargs = '+', help='tasks to be trained')
    parser.add_argument('--cluster_label', type=str, default="", help='label for label group')

    args = parser.parse_args()
    

    train(args, args.debug)
