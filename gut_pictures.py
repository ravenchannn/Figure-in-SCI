#脑肠亚型分类代码-肠
#Written by Raven
from __future__ import division
import matplotlib.pyplot as plt  # 加载画图的包
# 导入各种包
#####
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
#########################################################
#                                     实验目的：肠亚型结果图绘制                                #
#                                        written by Raven                                         #
#########################################################
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)    #加载所需中文字体
import warnings  #忽略warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定某块GPU跑代码
#####





# 载入数据远程CPU
#####
random_seed = 42
np.random.seed(random_seed)
#####





import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
# # alpha
# 读取数据
group = pd.read_csv("/mnt/disk1/wyr/result_gut_net/differ/Train/unmovecor_regenus_means_all.csv")
bacteria = group.columns.tolist()[1:]  # 提取除第一列外的其他列名
data = group.iloc[:, 1:].values.T
row_sums = group.iloc[:, 1:].sum(axis=1)
normalized_group = group.iloc[:, 1:].div(row_sums, axis=0)
groups = ['Gut Subtype1', 'Gut Subtype2', 'Gut Subtype3', 'Healthy Controls']
# 使用tab20颜色循环
colors = [
    '#7fc97f',
    '#beaed4',
    '#fdc086',
    '#ffff99',
    '#386cb0',
    '#bf5b16',
    '#f0027f',
    '#666666',
'#8dd3c7',
 '#FFD700',
 '#6959CD',
 '#fb8072',
 '#80b1d3',
 '#FAEBD7',
 '#b3de69',
 '#fccde5',
 '#d9d9d9',
 '#bc80bd',
 '#ccebc5',
 '#ffed6f',
]
# 创建一个新的图形对象，并设置大小
fig, ax = plt.subplots(figsize=(8, 6))
# 绘制堆叠条形图
normalized_group.plot(kind='bar', stacked=True, color=colors, ax=ax)
# 设置标签和标题
ax.set_xlabel('Groups', fontsize=11)
ax.set_ylabel('Genus Relative Abundance', fontsize=11)
ax.set_title('Differences of Genus Relative Abundance \n between three GSs and HCs (Train Set)', fontsize=12,loc='right')
# 水平显示横坐标名称
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, rotation=0,fontsize=10)
# 设置图例位置并设置放置位置及列数，移除外边框
ax.legend(loc='upper left', bbox_to_anchor=(1, 0.95), ncol=1, frameon=False)
plt.tight_layout()
plt.savefig('/mnt/disk1/wyr/result_gut_net/differ/Train/regenus.tiff',dpi=300)
plt.show()








# import matplotlib.pyplot as plt
# accent_cmap = plt.cm.get_cmap('Accent')
# colors = accent_cmap.colors
# for i, color in enumerate(colors):
#     hex_code = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
#     print(f" '{hex_code}',")
# # ACCENT
#  #7fc97f
#  #beaed4
#  #fdc086
#  #ffff99
#  #386cb0
#  #f0027f
#  #bf5b16
#  #666666
#
# # Set3
#  #8dd3c7
#  #ffffb3
#  #bebada
#  #fb8072
#  #80b1d3
#  #fdb462
#  #b3de69
#  #fccde5
#  #d9d9d9
#  #bc80bd
#  #ccebc5
#  #ffed6f