#脑肠亚型分类代码-BG
#Written by Raven
from __future__ import division

import random

import matplotlib.pyplot as plt  # 加载画图的包
# 导入各种包
#####
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
import seaborn as sns
#########################################################
#                     实验目的：个体脑-肠网络耦合特征画图、网络                        #
#                                        written by Raven                                         #
#########################################################
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)    #加载所需中文字体
import warnings  #忽略warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定某块GPU跑代码
#####


################
#####耦合特征#####
################
# 每种亚型求平均耦合特征
bgs1 = pd.read_csv("/mnt/disk1/wyr/result_bg_sz/pics/coupling/BGS1.csv")
bgs2 = pd.read_csv("/mnt/disk1/wyr/result_bg_sz/pics/coupling/BGS2.csv")
hc = pd.read_csv("/mnt/disk1/wyr/result_bg_sz/pics/coupling/HC.csv")
BGS1 = bgs1.mean()
BGS2 = bgs2.mean()
HC = hc.mean()
BGS1 = pd.DataFrame(BGS1)
BGS2 = pd.DataFrame(BGS2)
HC = pd.DataFrame(HC)
BGS1=BGS1.transpose()
BGS2=BGS2.transpose()
HC=HC.transpose()
BGS1.to_csv("/mnt/disk1/wyr/result_bg_sz/pics/coupling/BGS1_mean.csv",index=False)
BGS2.to_csv("/mnt/disk1/wyr/result_bg_sz/pics/coupling/BGS2_mean.csv",index=False)
HC.to_csv("/mnt/disk1/wyr/result_bg_sz/pics/coupling/HC_mean.csv",index=False)
# 与HC相减看差异
A = BGS1-HC
B = BGS2-HC
# 找到p<0.05保留
p1=pd.read_csv("/mnt/disk1/wyr/result_bg_sz/pics/coupling/BGS1P.csv")
p2=pd.read_csv("/mnt/disk1/wyr/result_bg_sz/pics/coupling/BGS2P.csv")
mask1 = p1 > 0.05
A[mask1] = 0
mask2 = p2 > 0.001
B[mask2] = 0
A=A.transpose()
B=B.transpose()
A.to_csv("/mnt/disk1/wyr/result_bg_sz/pics/coupling/BGS1_differ.csv") # 这里不设置header是为了更好excel筛选
B.to_csv("/mnt/disk1/wyr/result_bg_sz/pics/coupling/BGS2_differ.csv")
# 绘制偏差条形图
AA=pd.read_csv('/mnt/disk1/wyr/result_bg_sz/pics/coupling/BGS1_differ.csv',header=None) # 这里是excel筛选好的，所以重新导入
BB=pd.read_csv('/mnt/disk1/wyr/result_bg_sz/pics/coupling/BGS2_differ.csv',header=None)
# 红色#D86161，蓝色#6D9AEF
BB = BB.sort_values(by=BB.columns[1])
positive_data = BB[BB.iloc[:,1] > 0]
negative_data = BB[BB.iloc[:,1] < 0]
fig, ax = plt.subplots(figsize=(10, 7))
negative_bars=ax.barh(negative_data.iloc[:,0], negative_data.iloc[:,1], color='#6D9AEF', label='Significant decrease')
positive_bars=ax.barh(positive_data.iloc[:,0], positive_data.iloc[:,1], color='#D86161', label='Significant increase')
ax.set_xlabel('Differences between BGS2 and HC',fontsize=16)
ax.set_ylabel('Coupling Features',fontsize=16)
ax.legend(fontsize=12)
for bar in positive_bars:
    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}',
            va='center', ha='left', fontsize=8, color='black')
for bar in negative_bars:
    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}',
            va='center', ha='right', fontsize=8, color='black')
plt.tight_layout()
plt.savefig('/mnt/disk1/wyr/result_bg_sz/pics/coupling/BGS2.tiff',dpi=300)
plt.show()







################
#####网络特征#####
################
# FC.txt进行循环重排序
#####
import os
import pandas as pd
import numpy as np
new_order = [2,	3,	14,	15,	22,	23,	30,	31,	34,	35,	64,	65,	66,	67,	84,	85,	0,	1,	16,	17,
             18,	19,	56,	57,	68,	69,	78,	79,	80,	81,	42,	43,	44,	45,	46,	47,	48,	49,	50,	51,
             52,	53,	54,	55,	58,	59,	60,	61,	28,	29,	32,	33,	62,	63,	6,	7,	8,	9,	10,	11,	12,	13,
             36,	37,	40,	41,	70,	71,	72,	73,	74,	75,	76,	77,	4,	5,	20,	21,	24,	25,	26,	27,	38,
             39,	82,	83,	86,	87,	88,	89] # 八大网络顺序
input_folder = '/mnt/disk1/wyr/result_bg_sz/FC/'
output_folder = '/mnt/disk1/wyr/result_bg_sz/FC_reorder/'
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        df = pd.read_csv(os.path.join(input_folder, filename), header=None, delimiter='\t')
        new_df = df.iloc[new_order, new_order]
        new_df.to_csv(os.path.join(output_folder, f'reordered_{filename}'), header=None, index=None, sep='\t')
#####

# 98个矩阵取中间八大网络内连接（平均）→转置（90*43）
#####
import os
import numpy as np
result_matrix = []
input_folder = '/mnt/disk1/wyr/result_bg_sz/FC_reorder/'
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        data = np.loadtxt(os.path.join(input_folder, filename))  # 读取文件数据
        avg_values = np.mean(data, axis=0)  # 对列求平均，得到1*90的平均值向量
        result_matrix.append(avg_values)  # 将平均值向量添加到结果矩阵中
result_matrix = np.array(result_matrix)
print(result_matrix.shape)
M=result_matrix
M=pd.DataFrame(result_matrix)
M.to_csv('/mnt/disk1/wyr/result_bg_sz/pics/net/net.csv',header=None,index=False)
print(M)
#####

# 与relative abudance（43*20）循环点乘（90*20*43）→向量化（43*1800）
#####
brain = M.transpose()
gut = pd.read_csv('/mnt/disk1/wyr_data/bg_net_sz/GLM_removed_covariate/regenus.csv',encoding='gb18030')
result={}
results={}
for i in range(len(gut)):
    A = pd.DataFrame(brain.iloc[:, i].values.reshape(-1, 1))  # 将 Series 转换为 DataFrame
    B = pd.DataFrame(gut.iloc[i, :].values.reshape(1, -1))  # 将 Series 转换为 DataFrame
    results[i] = np.dot(A,B)
    result[i] = np.dot(A, B).flatten()  # 或者使用 .ravel() 方法
    result[i]=pd.DataFrame(result[i])
    results[i]=pd.DataFrame(results[i])
result = np.array(list(result.values()))
result = result.reshape(result.shape[0],-1)
result = pd.DataFrame(result)
result.to_csv('/mnt/disk1/wyr/result_bg_sz/pics/net/net_features_matrix.csv',header=None,index=False)
print(result)
#####

# 打标→excel完成并加入了列标题

# p值&每种亚型求平均网络特征
#####
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
cluster_demographic = pd.read_csv("/mnt/disk1/wyr/result_bg_sz/pics/net/net_features_matrix.csv")
cluster_demographic.reset_index(drop=True, inplace=True)
category_column_all = cluster_demographic.columns[0]
variable_columns_all = cluster_demographic.columns[1:]
result_df_all = pd.DataFrame(index=variable_columns_all, columns=['Kruskal-Wallis Statistic', 'P-value_all'])
for variable in variable_columns_all:
    group_data = [cluster_demographic[cluster_demographic[category_column_all] == category][variable]
                  for category in cluster_demographic[category_column_all].unique()]
    statistic, p_value = kruskal(*group_data)
    result_df_all.loc[variable, 'Kruskal-Wallis Statistic'] = statistic
    result_df_all.loc[variable, 'P-value_all'] = p_value
p_values_fdr = multipletests(result_df_all['P-value_all'], alpha=0.05, method='fdr_bh')[1]
result_df_all['P-value (FDR)'] = p_values_fdr
p_values_b = multipletests(result_df_all['P-value_all'], alpha=0.05, method='bonferroni')[1]
result_df_all['P-value (Bonferroni)'] = p_values_b
print('P-Value of all', result_df_all)
result_df_all.to_csv('/mnt/disk1/wyr/result_bg_sz/pics/net/net_p2x.csv')
grouped_data = cluster_demographic.groupby('id')
means = grouped_data.mean()
std = grouped_data.std()
sem = std / np.sqrt(len(grouped_data))
print('means',means)
means.to_csv('/mnt/disk1/wyr/result_bg_sz/pics/net/net_means.csv', index=False)

from itertools import combinations
from scipy.stats import kruskal
category_column_all = cluster_demographic.columns[0]
variable_columns_all = cluster_demographic.columns[1:]
result_df = pd.DataFrame(columns=['Feature', 'Group1', 'Group2', 'KW Statistic', 'P-value'])
# 对每两组进行两两比较
for group1, group2 in combinations(cluster_demographic[category_column_all].unique(), 2):
    for variable in variable_columns_all:
        data_group1 = cluster_demographic.loc[cluster_demographic[category_column_all] == group1, variable]
        data_group2 = cluster_demographic.loc[cluster_demographic[category_column_all] == group2, variable]
        # 执行 KW 检验
        stat, p_value = kruskal(data_group1, data_group2)
        # 将结果添加到DataFrame
        row_data = [variable, group1, group2, stat, p_value]
        result_df = pd.concat([result_df, pd.DataFrame([row_data], columns=result_df.columns)])
# 进行 FDR校正
p_values_fdr = multipletests(result_df['P-value'], alpha=0.05, method='fdr_bh')[1]
result_df['P-value (FDR)'] = p_values_fdr
# 进行 Bonferroni 校正
p_values_b = multipletests(result_df['P-value'], alpha=0.05, method='bonferroni')[1]
result_df['P-value (Bonferroni)'] = p_values_b
print(result_df)
result_df.to_csv('/mnt/disk1/wyr/result_bg_sz/pics/net/net_p3.csv',index=False)
#####

# 与HC相减看差异, 找到p<0.05保留→转化回矩阵方便绘制弦图
A = means.iloc[0]-means.iloc[2]
B = means.iloc[1]-means.iloc[2]
C=pd.DataFrame(A)
D=pd.DataFrame(B)
A=C.transpose()
B=D.transpose()
# 找到p<0.05保留
p1= pd.read_csv("/mnt/disk1/wyr/result_bg_sz/pics/net/BGS1P.csv")
p2= pd.read_csv("/mnt/disk1/wyr/result_bg_sz/pics/net/BGS2P.csv")
mask1 = p1 > 0.01
A[mask1] = 0
mask2 = p2 > 0.01
B[mask2] = 0
A=np.array(A)
B=np.array(B)
A=A.reshape(90, 20)
B=B.reshape(90, 20)
A=pd.DataFrame(A)
B=pd.DataFrame(B)
A.to_csv("/mnt/disk1/wyr/result_bg_sz/pics/net/BGS1_differ.csv",header=None,index=False)
B.to_csv("/mnt/disk1/wyr/result_bg_sz/pics/net/BGS2_differ.csv",header=None,index=False)
#####

# 这中间一步需要对differ文件生成0值形成一个B-GxB-G的矩阵。之前是BXG
# 向量矩阵化→绘制弦图
#####
# 绘制弦图
#####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
class BezierCurve:
    def __init__(self, control_points):
        self.control_points = np.array(control_points)
    def evaluate(self, t):
        n = len(self.control_points) - 1
        return np.sum([self.control_points[i] * self.bernstein_poly(i, n, t) for i in range(n + 1)], axis=0)
    def bernstein_poly(self, i, n, t):
        return np.math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
def draw_chord_plot(df):
    plt.rcParams.update({'font.size': 30})
    class_sizes = [3,16,14,14,4,6,8,12,16,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    class_names = ['','DMN', 'SMN', 'VSN', 'DAN', 'VAN', 'FPN', 'SCN', 'LN','',
                   'Collinsella', 'Faecalibacterium', 'Clostridium', 'Blautia', 'Gemmiger', 'Bacteroides', 'Prevotella',
                    'Lachnospira', 'Roseburia', 'Bifidobacterium', 'Bilophila', 'Dorea', 'Oscillospira', 'Streptococcus',
                    'Anaerostipes', 'Parabacteroides', 'Phascolarctobacterium', 'Coprococcus', 'Ruminococcus', 'Veillonella']
    custom_colors = [
    'white',
    '#008F7A',
    '#0089BA',
    '#2C73D2',
    '#845EC2',
    '#D65DB1',
    '#FF6F91',
    '#FF9671',
    '#FFC75F',
    'white',
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
     '#ffed6f']
    custom_cmap = ListedColormap(custom_colors)
    cmap = custom_cmap
    # cmap = plt.get_cmap('magma', len(class_sizes))
    colors = [cmap(i) for i in range(len(class_sizes))]
    plt.figure(figsize=(30, 15))
    plt.pie(class_sizes, startangle=0, colors=colors,
            wedgeprops=dict(width=0.2, linewidth=5, edgecolor='white'))
    centre_circle = plt.Circle((0, 0),0.85, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # sorted_values = df.values.flatten()
    # sorted_values.sort()
    # # top_values = sorted_values[-22:][::-1]  # 取前 10 大的值
    # # down_values = sorted_values[:22][::-1]  # 取前 10 大的值
    # 绘制连接曲线
    for i in range(116):
        for j in range(i + 1,116):
            value = df.iloc[i, j]
            if value > 0:#in top_values:
                angle_i = (i * 360 / 116)+2
                angle_j = (j * 360 / 116)+2
                x_i, y_i = 0.8 * np.cos(np.radians(angle_i)), 0.8 * np.sin(np.radians(angle_i))
                x_j, y_j = 0.8 * np.cos(np.radians(angle_j)), 0.8 * np.sin(np.radians(angle_j))
                # 使用自定义的贝塞尔曲线
                bezier_curve = BezierCurve([(x_i, y_i), (0,0),(x_j, y_j)])
                t_values = np.linspace(0, 1, 90)
                curve_points = np.array([bezier_curve.evaluate(t) for t in t_values])
                # 线条粗细和颜色
                color = '#D86161'##6D9AEF78
                # 线条粗细和颜色
                max_line_width = 15
                linewidths = max_line_width * np.concatenate((np.linspace(1, 0.1, 45), np.linspace(0.1, 1,45)))
                # 添加曲线
                for t in range(1, len(curve_points)):
                    plt.plot([curve_points[t - 1, 0], curve_points[t, 0]],
                             [curve_points[t - 1, 1], curve_points[t, 1]],
                             color=color, linewidth=linewidths[t], alpha=1)
    # for i in range(20):# 减小连接
        for j in range(i + 1, 116):
            value = df.iloc[i, j]
            if value < 0:
                angle_i = (i * 360 / 116)+2
                angle_j = (j * 360 / 116)+2
                x_i, y_i = 0.8 * np.cos(np.radians(angle_i)), 0.8 * np.sin(np.radians(angle_i))
                x_j, y_j = 0.8 * np.cos(np.radians(angle_j)), 0.8 * np.sin(np.radians(angle_j))
                # 使用自定义的贝塞尔曲线
                bezier_curve = BezierCurve([(x_i, y_i), (0,0),(x_j, y_j)])
                t_values = np.linspace(0, 1, 90)
                curve_points = np.array([bezier_curve.evaluate(t) for t in t_values])
                # 线条粗细和颜色
                color = '#6D9AEF'
                max_line_width = 15
                linewidths = max_line_width * np.concatenate((np.linspace(1, 0.1, 45), np.linspace(0.1, 1, 45)))
                for t in range(1, len(curve_points)):
                    plt.plot([curve_points[t - 1, 0], curve_points[t, 0]],
                             [curve_points[t - 1, 1], curve_points[t, 1]],
                             color=color, linewidth=linewidths[t], alpha=1)
    # 外周圆环图例
    class_names = ['','DMN', 'SMN', 'VSN', 'DAN', 'VAN', 'FPN', 'SCN', 'LN','',
                   'Collinsella', 'Faecalibacterium', 'Clostridium', 'Blautia', 'Gemmiger', 'Bacteroides', 'Prevotella',
                    'Lachnospira', 'Roseburia', 'Bifidobacterium', 'Bilophila', 'Dorea', 'Oscillospira', 'Streptococcus',
                    'Anaerostipes', 'Parabacteroides', 'Phascolarctobacterium', 'Coprococcus', 'Ruminococcus', 'Veillonella']
    # legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=30, label=l)
    #                   for c, l in zip(colors, class_names)]
    # plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 0.99), loc='upper right', frameon=False)
    plt.axis('equal')
# 使用示例
df = pd.read_csv('/mnt/disk1/wyr/result_bg_sz/pics/net/BGS2_differ.csv',header=None)
draw_chord_plot(df)
# plt.title('Abnormal Relationships of Genus-level Gut Flora between GS1 and HCs')
plt.savefig('/mnt/disk1/wyr/result_bg_sz/pics/net/BGS2_neg.tiff',dpi=300)
plt.tight_layout()
plt.show()
#####






# 平均矩阵
# 计算平均FC
#####
df = pd.read_csv('/mnt/disk1/wyr/result_bg_sz/BG_FC/label.csv')
output_directory = '/mnt/disk1/wyr/result_bg_sz/BG_FC/'
os.makedirs(output_directory, exist_ok=True)
label_matrix_sum = {}
label_matrix_count = {}
for index, row in df.iterrows():
    label = row['id']  # 替换为实际的标签列名
    file_path = row['File Name']  # 替换为实际的文件位置列名
    try:
        with open(file_path, 'r') as file:
            matrix_data = np.loadtxt(file, dtype=float).reshape((360, 21))
            if label in label_matrix_sum:
                label_matrix_sum[label] += matrix_data
                label_matrix_count[label] += 1
            else:
                label_matrix_sum[label] = matrix_data
                label_matrix_count[label] = 1
            print(f"处理文件 {file_path} 完成")
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {e}")
for label, matrix_sum in label_matrix_sum.items():
    average_matrix = matrix_sum / label_matrix_count[label]
    output_file_path = os.path.join(output_directory, f"{label}_average_matrix.txt")
    with open(output_file_path, 'w') as output_file:
        for row_values in average_matrix:
            output_file.write(' '.join(map(str, row_values)) + '\n')
    print(f"标签: {label}, 平均值已保存到 {output_file_path}")
#####
# 读取个体脑网络
folder_path = '/mnt/disk1/wyr/result_bg_sz/BG_FC/'
file_names = [f for f in os.listdir(folder_path) if f.endswith('matrix.txt')]
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    data = np.loadtxt(file_path)
    plt.figure(figsize=(10, 8))
    sns.heatmap(data,cmap='magma',vmin=-0.06,vmax=0.06)
    plt.title(f'Genus Level Intestinal Flora Correlation Matrix of {file_name}')
    plt.tight_layout()  # 调整图像边距
    # plt.xlabel(f'Brain Correlation Matrix of {file_name}')
    plt.savefig(f'/mnt/disk1/wyr/result_bg_sz/BG_FC/label {file_name}.tiff',dpi=300)
    # plt.savefig('/mnt/disk1/wyr/result_brain_net/FC/Mean Brain HeatMap of Subtype1')
    plt.show()






