import json
import os
import torch
import argparse
import numpy as np
import random
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
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

parser = argparse.ArgumentParser(description='NAS-Bench-201', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# use for train the model
parser.add_argument('--workers',     type=int,   default=0,      help='number of data loading workers (default: 2)')

parser.add_argument('--datasets',    type=str,   default='cifar10',      help='The applied datasets.')
parser.add_argument('--xpaths',      type=str,   default='cifar.python',      help='The root path for this dataset.')

parser.add_argument('--batch_size_1',   type=int,    default=1,      help='batch size for NTK.')
parser.add_argument('--batch_size_2',   type=int,    default=8,      help='batch size for regularization.')
parser.add_argument('--init_channels',   type=int,    default=16,              help='number of initial channels')
parser.add_argument('--layers',   type=int,    default=8,              help='number of layers.')
parser.add_argument('--auxiliary',   type=bool,    default=False,              help='whether use auxiliary head')
parser.add_argument('--h',   type=float,    default=50.0,              help='h')


args = parser.parse_args()

def get_random_genotype(seed):
    torch.manual_seed(seed)
    model = Super_network(16, 10, 8)
    genotype = model.genotype()
    return genotype

def main():

    #train_data, valid_data, xshape, class_num = get_datasets(args.datasets, args.xpaths, -1)

    #split_info = load_config('../../configs/nas-benchmark/cifar-split.txt', None, None)

    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
    #                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(split_info.train),
    #                                       num_workers=args.workers, pin_memory=True)

    train_transform, valid_transform = utils._data_transforms_cifar10_eval(args)
    train_data = dset.CIFAR10(root=args.xpaths, train=True, download=True, transform=train_transform)
    #valid_data = dset.CIFAR10(root=args.data_loc, train=False, download=True, transform=valid_transform)

    train_loader_1 = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size_1, shuffle=True, pin_memory=True, num_workers=2)

    train_loader_2 = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size_2, shuffle=True, pin_memory=True, num_workers=2)


    CIFAR_CLASSES = 10

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    robust_network_info_0 = json.load(open('robust_arch_dataset/robust_bench_seed_0_1000.json', 'r'))
    robust_network_info_1 = json.load(open('robust_arch_dataset/robust_bench_seed_1000_2000.json', 'r'))
    robust_network_info_2 = json.load(open('robust_arch_dataset/robust_bench_seed_2000_3000.json', 'r'))
    robust_network_info_3 = json.load(open('robust_arch_dataset/robust_bench_seed_3000_4000.json', 'r'))
    robust_network_info = robust_network_info_0 + robust_network_info_1 + robust_network_info_2 + robust_network_info_3
    #robust_network_info = robust_network_info_0
    '''
    for seed in range(1000, 2000):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        genotype = get_random_genotype(seed)
        print(genotype)
        network = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
        network = network.cuda()
    '''



    acc_robust_list = np.array([])

    grad_norm_list = np.array([])
    snip_list = np.array([])
    grasp_list = np.array([])
    fisher_list = np.array([])
    jacob_cov_list = np.array([])
    plain_list = np.array([])
    synflow_bn_list = np.array([])
    synflow_list = np.array([])

    print(len(robust_network_info))
    for arch in range(len(robust_network_info)):
        genotype_info = robust_network_info[arch]['net']['genotype']
        genotype = Genotype(normal=genotype_info['normal'], normal_concat=genotype_info['normal_concat'], reduce=genotype_info['reduce'], reduce_concat=genotype_info['reduce_concate'])
        print(genotype)

        network = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
        network = network.cuda()

        
        measures = predictive.find_measures(network,
                                            train_loader_2,
                                            ('random', 1, 10),
                                            torch.device("cuda:"+str(0) if torch.cuda.is_available() else "cpu"))
        print(measures)
        grad_norm_list = np.append(grad_norm_list, measures['grad_norm'])
        snip_list = np.append(snip_list, measures['snip'])
        grasp_list = np.append(grasp_list, measures['grasp'])
        fisher_list = np.append(fisher_list, measures['fisher'])
        jacob_cov_list = np.append(jacob_cov_list, measures['jacob_cov'])
        plain_list = np.append(plain_list, measures['plain'])
        synflow_bn_list = np.append(synflow_bn_list, measures['synflow_bn'])
        synflow_list = np.append(synflow_list, measures['synflow']) 

        acc_robust = robust_network_info[arch]['max_test_top1_adv']
        acc_robust_list = np.append(acc_robust_list, acc_robust)

    print(acc_robust_list)
    
    print(grad_norm_list)
    print(snip_list)
    print(grasp_list)
    print(fisher_list)
    print(jacob_cov_list)
    print(plain_list)
    print(synflow_bn_list)
    print(synflow_list)

    tau_grad_norm, p_grad_norm = stats.kendalltau(acc_robust_list, grad_norm_list)
    print('=============tau_grad_norm=============')
    print(tau_grad_norm)

    tau_snip, p_snip = stats.kendalltau(acc_robust_list, snip_list)
    print('=============tau_snip=============')
    print(tau_snip)

    tau_grasp, p_grasp = stats.kendalltau(acc_robust_list, grasp_list)
    print('=============tau_grasp=============')
    print(tau_grasp)

    tau_fisher, p_fisher = stats.kendalltau(acc_robust_list, fisher_list)
    print('=============tau_fisher=============')
    print(tau_fisher)

    tau_jacob_cov, p_jacob_cov = stats.kendalltau(acc_robust_list, jacob_cov_list)
    print('=============tau_jacob_cov=============')
    print(tau_jacob_cov)

    tau_plain, p_plain = stats.kendalltau(acc_robust_list, plain_list)
    print('=============tau_plain=============')
    print(tau_plain)

    tau_synflow_bn, p_synflow_bn = stats.kendalltau(acc_robust_list, synflow_bn_list)
    print('=============tau_synflow_bn=============')
    print(tau_synflow_bn)

    tau_synflow, p_synflow = stats.kendalltau(acc_robust_list, synflow_list)
    print('=============tau_synflow=============')
    print(tau_synflow)    




    print('finish')

if __name__ == '__main__':
    main()
