import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

data = loadmat(r"/Users/zhuxiaoqiang/Downloads/thirdDataSet/47imaginary_part/imaginary101.mat") #101  103
test = data['myData']
swap = test[:, :, 600:630]
newd = np.rollaxis(swap, 0, 3)
phase = abs(np.array(newd)[:,:,2])  # 0, 1, 2
sns.set(style='whitegrid',color_codes=True,)
sns.heatmap(phase, cbar=True)

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 15,
         }
plt.xlabel('Packets', font2)
plt.ylabel('Channel index', font2)
# plt.savefig('heatMapAntennas3.pdf', bbox_inches='tight', dpi=500)
plt.show()