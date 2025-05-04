import json
import matplotlib.pyplot as plt


custom_color1 = (70 / 255, 158 / 255, 180 / 255)
custom_color2 = (78 / 255, 98 / 255, 171 / 255)
custom_color3 = (203 / 255, 233 / 255, 157 / 255)
custom_color4 = (135 / 255, 207 / 255, 164 / 255)
custom_color5 = (253 / 255, 185 / 255, 106 / 255)
custom_color6 = (245 / 255, 117 / 255, 71 / 255)


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


def handle_data(data):
    natural_acc = []
    adv_acc = []
    for i in range(len(data)):
        natural_acc.append(data[i]['max_test_top1_original'])
        adv_acc.append(data[i]['max_test_top1_adv'])
    return natural_acc, adv_acc


data1 = read_json_file('/home/yuqi/zcp_nips/robust_bench_seed_0_1000.json')
data2 = read_json_file('/home/yuqi/zcp_nips/robust_bench_seed_1000_2000.json')
data3 = read_json_file('/home/yuqi/zcp_nips/robust_bench_seed_2000_3000.json')
data4 = read_json_file('/home/yuqi/zcp_nips/robust_bench_seed_3000_4000.json')

print(data1[0])

nat1, adv1 = handle_data(data1)
nat2, adv2 = handle_data(data2)
nat3, adv3 = handle_data(data3)
nat4, adv4 = handle_data(data4)

nat = nat1 + nat2 + nat3 + nat4
adv = adv1 + adv2 + adv3 + adv4

plt.scatter(nat, adv, color=custom_color6)
plt.xlabel('Natural Accuracy (%)')
plt.ylabel('Adversarial Robustness (%)')
plt.grid(alpha=0.4)
plt.show()