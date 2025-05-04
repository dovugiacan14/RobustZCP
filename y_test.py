import json 
import pickle 

path = "exps/Robust_ZCP/robust_arch_dataset/robust_bench_seed_0_1000.json"
path2 = "NasBench201/data/[NB201][CIFAR-10]_data.p"

with open(path, 'r') as f:
    data = json.load(f)

with open(path2, 'rb') as f:
    data2 = pickle.load(f)


print(data2)
print(data)

AVAILABLE_OPERATIONS = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
]


def extract_operation(op_string):
    return op_string.split("~")[0]

def encode_architecture(architecture_str):
    nodes = architecture_str.split("+")
    node_ops = [list(map(extract_operation, n.strip()[1:-1].split("|"))) for n in nodes]
    encoded_architecture = [AVAILABLE_OPERATIONS.index(op) for ops in node_ops for op in ops]
    return "".join(map(str, encoded_architecture))

arch_str = "|nor_conv_3x3~0|+|avg_pool_3x3~0|skip_connect~1|+|nor_conv_1x1~0|nor_conv_3x3~1|skip_connect~2|"

print(encode_architecture(arch_str))


