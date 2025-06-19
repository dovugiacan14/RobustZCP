import os
import sys
import logging
import time
import gc
from logging.handlers import RotatingFileHandler

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
file_handler = RotatingFileHandler('logging.txt', maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Add current directory to Python path for measures
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.append(os.path.dirname(__file__))

import json 
import torch
import random
import pickle
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict 
from torchvision import transforms
from torchvision import datasets as dset
from nas_201_api import NASBench201API as API
from models import get_cell_based_tiny_net
from functions import procedure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

AVAILABLE_OPERATIONS = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_operation(op_string):
    return op_string.split("~")[0]

def encode_architecture(architecture_str):
    nodes = architecture_str.split("+")
    node_ops = [list(map(extract_operation, n.strip()[1:-1].split("|"))) for n in nodes]
    encoded_architecture = [AVAILABLE_OPERATIONS.index(op) for ops in node_ops for op in ops]
    return "".join(map(str, encoded_architecture))

def data_transform_cifar10_train():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    return train_transform, valid_transform

def data_transform_cifar100_train():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            #transforms.Normalize((0.5071, 0.4867, 0.4304), (0.2675, 0.2565, 0.2761)),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5071, 0.4867, 0.4304), (0.2675, 0.2565, 0.2761)),
        ]
    )
    return train_transform, valid_transform

def data_transform_imagenet_train():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, valid_transform

def main(api):
    set_seed(2024)
    summary_stats = defaultdict(list)
    # datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    datasets = ["ImageNet16-120"]
    n_archs = 15625
    batch_size = 200
    output_folder = "/content/drive/MyDrive/RosNasBenchmark/RobustZCP"
    # output_folder = "Claim_Data/"

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for dataset in datasets:
        results_dt = {}
        if dataset == "cifar10": 
            train_transform, _ = data_transform_cifar10_train()
            train_data = dset.CIFAR10(
                root="./data", train=True, download=True, transform=train_transform
            )
            train_loader_1 = torch.utils.data.DataLoader(
                train_data, batch_size=1, shuffle=True, num_workers=2
            )
            train_loader_2 = torch.utils.data.DataLoader(
                train_data, batch_size=8, shuffle=True, num_workers=2
            )
            # with open("/content/drive/MyDrive/RosNasBenchmark/NASBench201/[CIFAR-10]_data.p", "rb") as file: 
            with open("NasBench201/data/NB201_CIFAR-10_data.p", "rb") as file: 
                data = pickle.load(file) 
            data_info = data["200"]

            # load robust scores  
            # with open("/content/drive/MyDrive/RosNasBenchmark/Robustness_Score_Data/cifar10.json", "r") as file: 
            with open("Robustness_Score/cifar10.json") as file: 
                robust_scores_pre = json.load(file) 
            
        elif dataset == "cifar100":
            train_transform, _ = data_transform_cifar100_train()
            train_data = dset.CIFAR100(
                root="./data", train=True, download=True, transform=train_transform
            )
            train_loader_1 = torch.utils.data.DataLoader(
                train_data, batch_size=1, shuffle=True, num_workers=0
            )
            train_loader_2 = torch.utils.data.DataLoader(
                train_data, batch_size=8, shuffle=True, num_workers=0
            )
            # load data info  
            with open("/content/drive/MyDrive/RosNasBenchmark/NASBench201/[CIFAR-100]_data.p", "rb") as file: 
                data = pickle.load(file) 
            data_info = data["200"]

            with open("/content/drive/MyDrive/RosNasBenchmark/Robustness_Score_Data/cifar100.json", "r") as file: 
                    robust_scores_pre = json.load(file) 

        elif dataset == "ImageNet16-120":
            train_transform, _ = data_transform_imagenet_train()
            train_data = dset.ImageFolder(
                root= "data_imagenet",
                transform= train_transform
            )
            train_loader_1 = torch.utils.data.DataLoader(
                train_data, batch_size=1, shuffle=True, num_workers=2
            )
            train_loader_2 = torch.utils.data.DataLoader(
                train_data, batch_size=8, shuffle=True, num_workers=2
            )
            with open("NasBench201/data/NB201_ImageNet16-120_data.p", "rb") as file: 
                data = pickle.load(file) 
            data_info = data["200"]

            with open("Robustness_Score/imagenet.json", "r") as file: 
                    robust_scores_pre = json.load(file) 
        else:
            raise ValueError(f"Dataset {dataset} not supported") 
        
        criterion = torch.nn.CrossEntropyLoss().to(device)

        robust_scores = [] 
        acc_scores = [] 
        fgsm_scores = []  
        pgd_scores = [] 

        # Process architectures in batches
        for batch_start in range(0, n_archs, batch_size):
            batch_end = min(batch_start + batch_size, n_archs)
            results_dt = {}  # Reset results_dt for each batch

            for i in range(batch_start, batch_end):
                config = api.get_net_config(i, dataset)
                arch_str = config['arch_str']
                fgsm_score = robust_scores_pre[arch_str]["val_fgsm_3.0_acc"]["threeseed"]
                pgd_score = robust_scores_pre[arch_str]["val_pgd_3.0_acc"]["threeseed"]
                arch_id = encode_architecture(arch_str)
                val_acc = data_info[arch_id]['val_acc'][-1]
                acc_scores.append(val_acc) 
                fgsm_scores.append(fgsm_score)
                pgd_scores.append(pgd_score)
                net = get_cell_based_tiny_net(config)
                net.to(device)

                score_robust, _, _, _ = procedure(
                    train_loader_1,
                    train_loader_2,
                    net,
                    criterion,
                    None,
                    None,   
                    "train",
                    grad=False,
                    h=50,
                    device=device
                )
                robust_scores.append(score_robust)
                results_dt[arch_str] = score_robust
                logging.info(f"Architecture {i} completed")
                logging.info(f"Robust score: {score_robust}")

            # Save batch results to pickle file in output folder
            batch_file = os.path.join(output_folder, f"summary_stats_{dataset}_batch_{batch_start}_{batch_size}.pkl")
            with open(batch_file, "wb") as f:
                pickle.dump({dataset: results_dt}, f)
            logging.info(f">>> Saved {dataset} batch {batch_start//batch_size} to {batch_file}")

            # Clear memory
            results_dt.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Sleep for 5 seconds
            time.sleep(5)

        # Compute correlations after processing all architectures
        tau, p_value = spearmanr(acc_scores, robust_scores)
        logging.info(f"Spearman's rank correlation coefficient with val_acc: {tau}")
        logging.info(f"p-value: {p_value}")

        tau, p_value = spearmanr(acc_scores, fgsm_scores)
        logging.info(f"Spearman's rank correlation coefficient with fgsm_score: {tau}")
        logging.info(f"p-value: {p_value}")

        tau, p_value = spearmanr(acc_scores, pgd_scores)
        logging.info(f"Spearman's rank correlation coefficient with pgd_score: {tau}")
        logging.info(f"p-value: {p_value}")
        
        summary_stats[dataset] = results_dt

    # Save final summary to output folder
    final_file = os.path.join(output_folder, "summary_stats_all.pkl")
    with open(final_file, "wb") as f:
        pickle.dump(summary_stats, f)

    logging.info(f">>> Saved all summary to {final_file}\n")

if __name__ == "__main__":
    # weight_path = "/content/drive/MyDrive/RosNasBenchmark/weights/NAS-Bench-201-v1_1-096897.pth"
    weight_path = "weights/NAS-Bench-201-v1_1-096897.pth"
    api = API(weight_path, verbose=False)
    # api = 0
    main(api)
    api.close()