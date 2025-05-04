import os
import sys

# # Get absolute path to the lib directory
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# LIB_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "lib"))

# # Add lib and current directory to sys.path with high priority
# if LIB_DIR not in sys.path:
#     sys.path.insert(0, LIB_DIR)

# if CURRENT_DIR not in sys.path:
#     sys.path.insert(0, CURRENT_DIR)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
# Add current directory to Python path for measures
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
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    return train_transform, valid_transform


def data_transform_cifar100_train():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4304), (0.2675, 0.2565, 0.2761)),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4304), (0.2675, 0.2565, 0.2761)),
        ]
    )
    return train_transform, valid_transform


def data_transform_imagenet_train():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, valid_transform


def main(api):
    set_seed(2024)
    summary_stats = defaultdict(list)
    datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    n_archs = 15625
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
            with open("NasBench201/data/[NB201][CIFAR-10]_data.p", "rb") as file: 
                data = pickle.load(file) 
            data_info = data["200"]

            # load robust scores  
            with open("Robustness_Score/cifar10.json", "r") as file: 
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
            with open("NasBench201/data/[NB201][CIFAR-100]_data.p", "rb") as file: 
                data = pickle.load(file) 
            data_info = data["200"]

            # load robust scores 

        elif dataset == "ImageNet16-120":
            train_transform, _ = data_transform_imagenet_train()
            train_data = dset.ImageNet(
                root="./data", train=True, download=True, transform=train_transform
            )
            train_loader_1 = torch.utils.data.DataLoader(
                train_data, batch_size=1, shuffle=True, num_workers=2
            )
            train_loader_2 = torch.utils.data.DataLoader(
                train_data, batch_size=8, shuffle=True, num_workers=2
            )
            with open("NasBench201/data/[NB201][ImageNet16-120]_data.p", "rb") as file: 
                data = pickle.load(file) 
            data_info = data["200"]
        else:
            raise ValueError(f"Dataset {dataset} not supported") 
        
        criterion = torch.nn.CrossEntropyLoss().to(device)

        robust_scores = [] 
        acc_scores = [] 
        fgsm_scores = []  
        pgd_scores = [] 
        for i in range(n_archs):
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
            print(f"Architecture {i} completed")
            print(f"Robust score: {score_robust}")
            
        with open(f"summary_stats_{dataset}.pkl", "wb") as f:
            pickle.dump({dataset: results_dt}, f)

        print(f">>> Saved {dataset} summary to summary_stats_{dataset}.pkl\n")

        tau, p_value = spearmanr(acc_scores, robust_scores)
        print(f"Spearman's rank correlation coefficient with val_acc: {tau}")
        print(f"p-value: {p_value}")

        tau, p_value = spearmanr(acc_scores, fgsm_scores)
        print(f"Spearman's rank correlation coefficient with fgsm_score: {tau}")
        print(f"p-value: {p_value}")

        tau, p_value = spearmanr(acc_scores, pgd_scores)
        print(f"Spearman's rank correlation coefficient with pgd_score: {tau}")
        print(f"p-value: {p_value}")
        
        summary_stats[dataset] = results_dt

    with open("summary_stats_all.pkl", "wb") as f:
        pickle.dump(summary_stats, f)

    print(">>> Saved all summary to summary_stats_all.pkl\n")


if __name__ == "__main__":
    weight_path = "weights/NAS-Bench-201-v1_1-096897.pth"
    api = API(weight_path, verbose=False)
    main(api)
    api.close()
