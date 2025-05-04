import argparse
#import datasets
import random
import numpy as np
import torch
import os
#from scores import get_score_func
#from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
#from statistics import mean, stdev
import time
#from utils import add_dropout

from model_search import Network as Super_network
from model import NetworkCIFAR as Network
import utils
import torchvision.datasets as dset

from functions import procedure, procedure_test_reg, procedure_adversaial_example, procedure_eigen

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results/ICML', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nds_darts_fix-w-d', type=str, help='the nas search space to use')
parser.add_argument('--batch_size_1', default=1, type=int)
parser.add_argument('--batch_size_2', default=32, type=int)
parser.add_argument('--kernel', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--activations', action='store_true')
parser.add_argument('--cosine', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=1000, type=int)
parser.add_argument('--n_runs', default=1, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int,
                    help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels, 16,  36')
parser.add_argument('--layers', type=int, default=20, help='total number of layers, 8, 20')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--h',   type=float,    default=50.0,              help='h')
parser.add_argument('--xpaths',      type=str,   default='cifar.python',      help='The root path for this dataset.')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def get_random_genotype(seed):
    torch.manual_seed(seed)
    model = Super_network(16, 10, 8)
    genotype = model.genotype()
    return genotype

def count_normal_skip(genotype):
    count = 0
    normal = genotype.normal
    for i in range(len(normal)):
        if normal[i][0] == 'skip_connect':
            count = count + 1
    return count

def count_normal_sep33(genotype):
    count = 0
    normal = genotype.normal
    for i in range(len(normal)):
        if normal[i][0] == 'sep_conv_3x3':
            count = count + 1
    return count

def count_normal_sep55(genotype):
    count = 0
    normal = genotype.normal
    for i in range(len(normal)):
        if normal[i][0] == 'sep_conv_5x5':
            count = count + 1
    return count

def count_normal_dil33(genotype):
    count = 0
    normal = genotype.normal
    for i in range(len(normal)):
        if normal[i][0] == 'dil_conv_3x3':
            count = count + 1
    return count

def count_normal_dil55(genotype):
    count = 0
    normal = genotype.normal
    for i in range(len(normal)):
        if normal[i][0] == 'dil_conv_5x5':
            count = count + 1
    return count

def count_normal_max(genotype):
    count = 0
    normal = genotype.normal
    for i in range(len(normal)):
        if normal[i][0] == 'max_pool_3x3':
            count = count + 1
    return count

def count_normal_avg(genotype):
    count = 0
    normal = genotype.normal
    for i in range(len(normal)):
        if normal[i][0] == 'avg_pool_3x3':
            count = count + 1
    return count

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# searchspace = nasspace.get_search_space(args)
train_transform, valid_transform = utils._data_transforms_cifar10_eval(args)
train_data = dset.CIFAR10(root=args.xpaths, train=True, download=True, transform=train_transform)
# valid_data = dset.CIFAR10(root=args.data_loc, train=False, download=True, transform=valid_transform)

train_loader_1 = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size_1, shuffle=True, pin_memory=True, num_workers=0)

train_loader_2 = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size_2, shuffle=True, pin_memory=True, num_workers=0)

os.makedirs(args.save_loc, exist_ok=True)

times = []
chosen = []
acc = []
val_acc = []
topscores = []
topconvs = []
topreg_negatives = []


order_fn = np.nanargmax

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

CIFAR_CLASSES = 10
runs = trange(args.n_runs, desc='acc: ')
for N in runs:
    start = time.time()
    # indices = np.random.randint(0,len(searchspace),args.n_samples)
    scores = []
    convs = []
    reg_negatives = []
    genotype_list = []

    npstate = np.random.get_state()
    ranstate = random.getstate()
    torchstate = torch.random.get_rng_state()
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    count = 0
    for seed in range(3000, 4000):
        print(count)
        print(seed)

        genotype = get_random_genotype(seed)

        if False:
        #if count_normal_skip(genotype) != 1:
            continue
        else:
            count = count + 1
            print(genotype)
            network = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)


            network.to(device)
            #if args.dropout:
            #    add_dropout(network, args.sigma)





            random.setstate(ranstate)
            np.random.set_state(npstate)
            torch.set_rng_state(torchstate)

            conv, score_robust, _, _, reg_negative, _, _, _ = procedure(train_loader_1, train_loader_2, network, criterion, None, None, 'train', grad=False, h=args.h)





            scores.append(score_robust.cpu())
            convs.append(conv.cpu())
            reg_negatives.append(reg_negative.cpu())
            genotype_list.append(genotype)

    # print(len(scores))
    # print(scores)
    # print(order_fn(scores))

    test_list = scores
    m_sorted = sorted(enumerate(test_list), key=lambda x: x[1], reverse=True)
    sorted_inds = [m[0] for m in m_sorted]
    sorted_nums = [m[1] for m in m_sorted]
    rank_list = []
    for i in range(len(test_list)):
        for j in range(len(sorted_inds)):
            if sorted_inds[j] == i:
                rank_list.append(j + 1)
    skip = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    sep33 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    sep55 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    dil33 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    dil55 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    max = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    avg = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    
    for i in range(len(rank_list)):
        if 1 <= rank_list[i] <= 100:
            count_skip = count_normal_skip(genotype_list[i])
            skip[count_skip] = skip[count_skip] + 1
            count_sep33 = count_normal_sep33(genotype_list[i])
            sep33[count_sep33] = sep33[count_sep33] + 1
            count_sep55 = count_normal_sep55(genotype_list[i])
            sep55[count_sep55] = sep55[count_sep55] + 1
            count_dil33 = count_normal_dil33(genotype_list[i])
            dil33[count_dil33] = dil33[count_dil33] + 1
            count_dil55 = count_normal_dil55(genotype_list[i])
            dil55[count_dil55] = dil55[count_dil55] + 1
            count_max = count_normal_max(genotype_list[i])
            max[count_max] = max[count_max] + 1
            count_avg = count_normal_avg(genotype_list[i])
            avg[count_avg] = avg[count_avg] + 1
    print('======================hh=========================')
    print(skip)
    print(sep33)
    print(sep55)
    print(dil33)
    print(dil55)
    print(max)
    print(avg)
    best_arch = genotype_list[order_fn(scores)]

    # indices[order_fn(scores)]
    # uid = searchspace[best_arch]
    topscores.append(scores[order_fn(scores)])
    topconvs.append(convs[order_fn(scores)])
    topreg_negatives.append(reg_negatives[order_fn(scores)])
    chosen.append(best_arch)
    # acc.append(searchspace.get_accuracy(uid, acc_type, args.trainval))
    # hh = searchspace.get_final_accuracy(uid, acc_type, False)
    # print(hh)
    # acc.append(hh)
    print('========scores=========')
    print(scores)
    print('========convs=========')
    print(convs)
    print('========reg_negatives=========')
    print(reg_negatives)
    print('========topscores=========')
    print(topscores)
    print('========topconvs=========')
    print(topconvs)
    print('========topreg_negatives=========')
    print(topreg_negatives)
    print('========chosen=========')
    print(chosen)


    times.append(time.time() - start)
    print('========times=========')
    print(times)


