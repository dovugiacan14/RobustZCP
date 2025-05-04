import json
import os
import torch
import argparse
import numpy as np
import random
import sys
from datetime import datetime

# Add lib directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
# Add current directory to Python path for measures
sys.path.append(os.path.dirname(__file__))

from functions import procedure, procedure_test_reg
from datasets import get_datasets
from config_utils import load_config
from model_search import Network as Super_network
from model import NetworkCIFAR as Network
from genotypes import Genotype
from scipy import stats
import predictive
import utils
import torchvision.datasets as dset
import matplotlib.pyplot as plt

# Set device to CPU
device = torch.device('cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(
    description="NAS-Bench-201", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# use for train the model
parser.add_argument(
    "--workers", type=int, default=0, help="number of data loading workers (default: 2)"
)

parser.add_argument(
    "--datasets", type=str, default="cifar10", help="The applied datasets."
)
parser.add_argument(
    "--xpaths", type=str, default="cifar.python", help="The root path for this dataset."
)

parser.add_argument("--batch_size_1", type=int, default=1, help="batch size for NTK.")
parser.add_argument(
    "--batch_size_2", type=int, default=8, help="batch size for regularization."
)
parser.add_argument(
    "--init_channels", type=int, default=16, help="number of initial channels"
)
parser.add_argument("--layers", type=int, default=8, help="number of layers.")
parser.add_argument(
    "--auxiliary", type=bool, default=False, help="whether use auxiliary head"
)
parser.add_argument("--h", type=float, default=50, help="h")
parser.add_argument("--seed", type=int, default=11, help="random seed")

args = parser.parse_args()


def get_random_genotype(seed):
    torch.manual_seed(seed)
    model = Super_network(16, 10, 8)
    genotype = model.genotype()
    return genotype


def count_normal_skip(genotype):
    count = 0
    normal = genotype.normal
    for i in range(len(normal)):
        if normal[i][0] == "skip_connect":
            count = count + 1
    return count


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_transform, valid_transform = utils._data_transforms_cifar10_eval(args)
    train_data = dset.CIFAR10(
        root=args.xpaths, train=True, download=True, transform=train_transform
    )

    train_loader_1 = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size_1,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    train_loader_2 = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size_2,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    CIFAR_CLASSES = 10

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)  # Move criterion to CPU

    # Load pre-computed robust architectures information from the file 
    robust_network_info_0 = json.load(
        open("robust_arch_dataset/robust_bench_seed_0_1000.json", "r")
    )
    robust_network_info_1 = json.load(
        open("robust_arch_dataset/robust_bench_seed_1000_2000.json", "r")
    )
    robust_network_info_2 = json.load(
        open("robust_arch_dataset/robust_bench_seed_2000_3000.json", "r")
    )
    robust_network_info_3 = json.load(
        open("robust_arch_dataset/robust_bench_seed_3000_4000.json", "r")
    )
    robust_network_info = (
        robust_network_info_0
        + robust_network_info_1
        + robust_network_info_2
        + robust_network_info_3
    )

    acc_robust_list = np.array([])
    score_robust_list = np.array([])
    
    # Create a dictionary to store all results
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "datasets": args.datasets,
            "xpaths": args.xpaths,
            "batch_size_1": args.batch_size_1,
            "batch_size_2": args.batch_size_2,
            "init_channels": args.init_channels,
            "layers": args.layers,
            "auxiliary": args.auxiliary,
            "h": args.h,
            "seed": args.seed
        },
        "architectures": []
    }

    print(len(robust_network_info))
    for arch in range(len(robust_network_info)):
        # Step 1: Get the genotype 
        genotype_info = robust_network_info[arch]["net"]["genotype"]
        genotype = Genotype(
            normal=genotype_info["normal"],
            normal_concat=genotype_info["normal_concat"],
            reduce=genotype_info["reduce"],
            reduce_concat=genotype_info["reduce_concate"],
        )
        print(genotype)

        # Step 2: Build the network
        network = Network(
            args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype
        )
        network = network.to(device)  # Move network to CPU

        # Step 3: Evaluate the architecture using procedure()
        score_robust, _, _, _ = procedure(
            train_loader_1,
            train_loader_2,
            network,
            criterion,
            None,
            None,
            "train",
            grad=False,
            h=args.h,
            device=device
        )

        # Step 4: Store the results
        acc_robust = robust_network_info[arch]["max_test_top1_adv"]
        acc_robust_list = np.append(acc_robust_list, acc_robust)
        score_robust_list = np.append(score_robust_list, score_robust.cpu())

        # Step 5: Store results for this architecture
        arch_result = {
            "architecture_id": arch,
            "genotype": {
                "normal": genotype_info["normal"],
                "normal_concat": genotype_info["normal_concat"],
                "reduce": genotype_info["reduce"],
                "reduce_concat": genotype_info["reduce_concate"]
            },
            "robust_accuracy": float(acc_robust),
            "zero_cost_score": float(score_robust.cpu().numpy())
        }
        results["architectures"].append(arch_result)

    print("len(acc_robust_list)")
    print(len(acc_robust_list))
    print(acc_robust_list)

    # Step 6: Calculate the correlation between the robust accuracy and the zero-cost score
    tau_robust, p_robust = stats.kendalltau(acc_robust_list, score_robust_list)
    print("=============tau_robust============")
    print(tau_robust)

    # Step 7: Add correlation results
    results["correlation"] = {
        "kendall_tau": float(tau_robust),
        "p_value": float(p_robust)
    }

    # Step 8: Save results to JSON file
    try:
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"zero_cost_scores_{timestamp}.json"
        
        # Get the absolute path of the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, output_file)
        
        # Save the results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Attempted to save to: {output_path}")

    print("finish")


if __name__ == "__main__":
    main()
