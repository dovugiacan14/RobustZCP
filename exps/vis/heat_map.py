import numpy as np
import matplotlib.pyplot as plt

default_font = {'size': 24}
tag_y = ["FGSM", "PGD", "Square", "APGD"]
tag_x = ['0.1/255', '0.5/255', '1.0/255', '2.0/255','3.0/255', '4.0/255', '8.0/255']

zcp = [[0.38, 0.39, 0.43, 0.46, 0.47, 0.47, 0.46],
       [0.38, 0.21, 0.17, 0.26, 0.30, 0.32, 0.28],
       [0.38, 0.35, 0.24, 0.12, 0.13, 0.16, 0.09],
       [0.38, 0.03, -0.04, 0.05, 0.11, 0.02, -0.07]]

values = np.array(zcp)

fig, ax = plt.subplots(1, 1, figsize=(20, 5), dpi=500)
plt.subplots_adjust(top=0.94, bottom=0.22, left=0.15, right=0.99, hspace=0,
                    wspace=0)

ax.set_xticks(np.arange(len(tag_x)))
ax.set_yticks(np.arange(len(tag_y)))
ax.set_xticklabels(tag_x)
ax.set_yticklabels(tag_y)
plt.setp(ax.get_xticklabels())

for i in range(len(tag_y)):
    for j in range(len(tag_x)):
        text = ax.text(j, i, values[i, j],
                       ha="center", va="center", color="black")


plt.imshow(values, cmap='coolwarm', origin='upper', aspect="auto")
plt.colorbar().ax.tick_params(labelsize=24)
plt.xlabel('Total Perturbation Scale e', default_font)
plt.ylabel('Attack Method', default_font)
plt.tick_params(labelsize=24)
plt.tight_layout()
plt.show()