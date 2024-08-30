#脑肠亚型分类代码-脑
#Written by Raven
from __future__ import division

import math

#########################################################
#                               实验目的：绘制个体肠网络热图                                   #
#                                        written by Raven                                         #
#########################################################






#导入各种包
#####
import matplotlib.pyplot as plt  #加载画图的包
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)    #加载所需中文字体
import warnings  #忽略warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定某块GPU跑代码
#####





# # 批量复制文件名
# ###
# folder_path = 'G:\\Gut316_gutonly\\gutnet'
# file_names = os.listdir(folder_path)
# df = pd.DataFrame({'File Name': file_names})
# excel_file_path = 'G:\Gut316_gutonly\gutnet\\file_name.xlsx'
# df.to_excel(excel_file_path, index=False)
# print(f'文件名已成功复制到 {excel_file_path}')
# ####
#
#
#
#
#
# # 计算平均FC
# #####
# df = pd.read_csv('D:\\AAARavenResults\\gut_net\\differ\\label_FC.csv')
# output_directory = 'D:\\AAARavenResults\\gut_net\\label'
# os.makedirs(output_directory, exist_ok=True)
# # 用于存储每个标签的矩阵累加结果和计数
# label_matrix_sum = {}
# label_matrix_count = {}
# # 循环遍历每一行
# for index, row in df.iterrows():
#     label = row['id']  # 替换为实际的标签列名
#     file_path = row['File Name']  # 替换为实际的文件位置列名
#     # 读取txt文件并计算平均值
#     try:
#         with open(file_path, 'r') as file:
#             # 读取矩阵数据并转换为NumPy数组
#             matrix_data = np.loadtxt(file, dtype=float).reshape((20, 20))
#
#             # 累加矩阵结果
#             if label in label_matrix_sum:
#                 label_matrix_sum[label] += matrix_data
#                 label_matrix_count[label] += 1
#             else:
#                 label_matrix_sum[label] = matrix_data
#                 label_matrix_count[label] = 1
#
#             print(f"处理文件 {file_path} 完成")
#     except Exception as e:
#         print(f"处理文件 {file_path} 时发生错误: {e}")
# # 计算每个标签的平均矩阵并保存
# for label, matrix_sum in label_matrix_sum.items():
#     average_matrix = matrix_sum / label_matrix_count[label]
#     # 构建存储平均值的txt文件路径
#     output_file_path = os.path.join(output_directory, f"{label}_average_matrix.txt")
#     # 将平均值写入新的txt文件
#     with open(output_file_path, 'w') as output_file:
#         for row_values in average_matrix:
#             output_file.write(' '.join(map(str, row_values)) + '\n')
#     print(f"标签: {label}, 平均值已保存到 {output_file_path}")
# #####
#
#
#
#
#
# # 读取个体脑网络
# folder_path = 'D:\AAARavenResults\gut_net\label'
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
# labels = ['Collinsella', 'Faecalibacterium', 'Clostridium', 'Blautia', 'Gemmiger', 'Bacteroides', 'Prevotella', 'Lachnospira',
#           'Roseburia', 'Bifidobacterium', 'Bilophila', 'Dorea', 'Oliverpabstia', 'Streptococcus', 'Anaerostipes', 'Parabacteroides',
#           'Phascolarctobacterium', 'Coprococcus', 'Ruminococcus', 'Veillonella']
# for file_name in file_names:
#     file_path = os.path.join(folder_path, file_name)
#     data = np.loadtxt(file_path)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(data,cmap='magma')
#     plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45,rotation_mode='anchor', ha='right',fontsize=10)
#     plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0,fontsize=10)
#     plt.title(f'Genus Level Intestinal Flora Correlation Matrix of {file_name}')
#     plt.tight_layout()  # 调整图像边距
#     # plt.xlabel(f'Brain Correlation Matrix of {file_name}')
#     plt.savefig(f'D:\\AAARavenResults\\gut_net\\differ\\Train\\FC\\label {file_name}.tiff',dpi=300)
#     # plt.savefig('/mnt/disk1/wyr/result_brain_net/FC/Mean Brain HeatMap of Subtype1')
#     plt.show()





# # GSs - HC提取p<0.05,其余置零
# #####
# # 构成FC统计Dataframe堆叠成一个vstack方便求P
# import numpy as np
# import pandas as pd
# from scipy.stats import kruskal
# # 读取包含组别信息的 CSV 文件
# df = pd.read_csv('D:\\AAARavenResults\\gut_net\\differ\\label_FC.csv')
# # 初始化空的矩阵列表，用于存放每个组别的矩阵
# matrix_groups = {group: [] for group in df['id'].unique()}
# # 读取每个文件中的矩阵数据并存入对应的组别
# for index, row in df.iterrows():
#     group = row['id']
#     filename = row['File Name']
#     with open(filename, 'r') as file:
#         matrix = file.read()
#     matrix = np.fromstring(matrix, sep=' ').reshape(20, 20)
#     matrix_flatten = matrix.reshape(1,-1)
#     matrix_groups[group].append(matrix_flatten)
# p_value_matrices = []
# A = matrix_groups[0]
# B = matrix_groups[1]
# C = matrix_groups[2]
# D = matrix_groups[3]
# group0 = np.vstack(A)
# group1 = np.vstack(B)
# group2 = np.vstack(C)
# group3 = np.vstack(D)
# whole = np.vstack((group0,group1,group2,group3))
# whole = pd.DataFrame(whole)
# label0 = np.full(137,0,dtype=int)
# label1 = np.full(17,1,dtype=int)
# label2 = np.full(39,2,dtype=int)
# label3 = np.full(123,3,dtype=int)
# label0 = label0.reshape(-1,1)
# label1 = label1.reshape(-1,1)
# label2 = label2.reshape(-1,1)
# label3 = label3.reshape(-1,1)
# label0 = pd.DataFrame(label0)
# label1 = pd.DataFrame(label1)
# label2 = pd.DataFrame(label2)
# label3 = pd.DataFrame(label3)
# label = label0._append(label1)
# label = label._append(label2)
# label = label._append(label3)
# label = np.array(label)
# label = pd.DataFrame(label)
# whole.insert(0,'label',label) # 按照0123排序
# # #
# # #
# # #
# #统计FC两两之间（p）
# from itertools import combinations
# from scipy.stats import kruskal
# from statsmodels.stats.multitest import multipletests
# category_column_all = whole.columns[0]
# variable_columns_all = whole.columns[1:]
# # 存储结果的DataFrame
# result_df = pd.DataFrame(columns=['Feature', 'Group1', 'Group2', 'KW Statistic', 'P-value'])
# # 对每两组进行两两比较
# for group1, group2 in combinations(whole[category_column_all].unique(), 2):
#     for variable in variable_columns_all:
#         data_group1 = whole.loc[whole[category_column_all] == group1, variable]
#         data_group2 = whole.loc[whole[category_column_all] == group2, variable]
#         if len(set(data_group1.values.flatten())) > 1 and len(set(data_group2.values.flatten())) > 1:
#             # 进行 Kruskal-Wallis 检验
#             stat, p_value = kruskal(data_group1.values, data_group2.values)
#         else:
#             # 数值全部相同，将 p-value 设置为 1
#             stat, p_value = None, 1.0
#         # 打印结果
#         print("Kruskal-Wallis Statistic:", stat)
#         print("P-value:", p_value)
#         # 将结果添加到DataFrame
#         row_data = [variable, group1, group2, stat, p_value]
#         result_df = pd.concat([result_df, pd.DataFrame([row_data], columns=result_df.columns)])
# # 进行 FDR校正
# p_values_fdr = multipletests(result_df['P-value'], alpha=0.05, method='fdr_bh')[1]
# result_df['P-value (FDR)'] = p_values_fdr
# # 进行 Bonferroni 校正
# p_values_b = multipletests(result_df['P-value'], alpha=0.05, method='bonferroni')[1]
# result_df['P-value (Bonferroni)'] = p_values_b
# print(result_df)
# result_df.to_csv('D:\\AAARavenResults\\gut_net\\differ\\Train\\FC\\FC_P_value_all.csv')
# # #
# # #
# # #
# # 做减法，提取其余置0
# result_df = pd.read_csv('D:\\AAARavenResults\\gut_net\\differ\\Train\\FC\\FC_P_value_all.csv')
# P01 = result_df.iloc[0:400,6]
# P01 = np.array(P01)
# P01 =P01.reshape(20, 20)
# P02 = result_df.iloc[400:800,6]
# P02 = np.array(P02)
# P02 =P02.reshape(20, 20)
# P03 = result_df.iloc[800:1200,6]
# P03 = np.array(P03)
# P03 =P03.reshape(20, 20)
# P12 = result_df.iloc[1200:1600,6]
# P12 = np.array(P12)
# P12 =P12.reshape(20, 20)
# P13 = result_df.iloc[1600:2000,6]
# P13 = np.array(P13)
# P13 =P13.reshape(20, 20)
# P23 = result_df.iloc[2000:2400,6]
# P23 = np.array(P23)
# P23 =P23.reshape(20, 20)
# # 对应-GS与HC的差异
# P0 = P03
# P1 = P13
# P2 = P23
# # 读取平均FC，找到显著的边连接
# av0 = pd.read_csv('D:\\AAARavenResults\\gut_net\\label\\0_average_matrix.txt', sep=' ',header=None)
# av1 = pd.read_csv('D:\\AAARavenResults\\gut_net\\label\\1_average_matrix.txt', sep=' ',header=None)
# av2 = pd.read_csv('D:\\AAARavenResults\\gut_net\\label\\2_average_matrix.txt', sep=' ',header=None)
# av3 = pd.read_csv('D:\\AAARavenResults\\gut_net\\label\\3_average_matrix.txt', sep=' ',header=None)
# # P0中有值大于0.01，则将av0中对应位置的数值置0，P0不变，改av0
# A = av0-av3
# B = av1-av3
# C = av2-av3
# mask0 = P0 > 0.05
# mask1 = P1 > 0.05
# mask2 = P2 > 0.05
# A[mask0] = 0
# B[mask1] = 0
# C[mask2] = 0
# # 输出
# A.to_csv('D:\AAARavenResults\gut_net\differ\Train\FC\GS1_FC.csv',index=False,header=None)
# B.to_csv('D:\AAARavenResults\gut_net\differ\Train\FC\GS2_FC.csv',index=False,header=None)
# C.to_csv('D:\AAARavenResults\gut_net\differ\Train\FC\GS3_FC.csv',index=False,header=None)
# # #####





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
    class_sizes = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    class_names = ['Collinsella', 'Faecalibacterium', 'Clostridium', 'Blautia', 'Gemmiger', 'Bacteroides', 'Prevotella',
                    'Lachnospira', 'Roseburia', 'Bifidobacterium', 'Bilophila', 'Dorea', 'Oscillospira', 'Streptococcus',
                    'Anaerostipes', 'Parabacteroides', 'Phascolarctobacterium', 'Coprococcus', 'Ruminococcus', 'Veillonella']
    custom_colors = [
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
            wedgeprops=dict(width=0.5, linewidth=15, edgecolor='white'))
    centre_circle = plt.Circle((0, 0),0.85, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    sorted_values = df.values.flatten()
    sorted_values.sort()
    # top_values = sorted_values[-22:][::-1]  # 取前 10 大的值
    # down_values = sorted_values[:22][::-1]  # 取前 10 大的值
    # 绘制连接曲线
    for i in range(20):
        for j in range(i + 1, 20):
            value = df.iloc[i, j]
            if value > 0:#in top_values:
                angle_i = (i * 360 / 20)+10
                angle_j = (j * 360 / 20)+10
                x_i, y_i = 0.8 * np.cos(np.radians(angle_i)), 0.8 * np.sin(np.radians(angle_i))
                x_j, y_j = 0.8 * np.cos(np.radians(angle_j)), 0.8 * np.sin(np.radians(angle_j))
                # 使用自定义的贝塞尔曲线
                bezier_curve = BezierCurve([(x_i, y_i), (0,0),(x_j, y_j)])
                t_values = np.linspace(0, 1, 90)
                curve_points = np.array([bezier_curve.evaluate(t) for t in t_values])
                # 线条粗细和颜色
                color = '#D86161'##6D9AEF78
                # 线条粗细和颜色
                max_line_width = 35
                linewidths = max_line_width * np.concatenate((np.linspace(1, 0.1, 45), np.linspace(0.1, 1,45)))
                # 添加曲线
                for t in range(1, len(curve_points)):
                    plt.plot([curve_points[t - 1, 0], curve_points[t, 0]],
                             [curve_points[t - 1, 1], curve_points[t, 1]],
                             color=color, linewidth=linewidths[t], alpha=1)
    # for i in range(20):# 减小连接
        for j in range(i + 1, 20):
            value = df.iloc[i, j]
            if value < 0:
                angle_i = (i * 360 / 20)+10
                angle_j = (j * 360 / 20)+10
                x_i, y_i = 0.8 * np.cos(np.radians(angle_i)), 0.8 * np.sin(np.radians(angle_i))
                x_j, y_j = 0.8 * np.cos(np.radians(angle_j)), 0.8 * np.sin(np.radians(angle_j))
                # 使用自定义的贝塞尔曲线
                bezier_curve = BezierCurve([(x_i, y_i), (0,0),(x_j, y_j)])
                t_values = np.linspace(0, 1, 90)
                curve_points = np.array([bezier_curve.evaluate(t) for t in t_values])
                # 线条粗细和颜色
                color = '#6D9AEF'
                max_line_width = 35
                linewidths = max_line_width * np.concatenate((np.linspace(1, 0.1, 45), np.linspace(0.1, 1, 45)))
                for t in range(1, len(curve_points)):
                    plt.plot([curve_points[t - 1, 0], curve_points[t, 0]],
                             [curve_points[t - 1, 1], curve_points[t, 1]],
                             color=color, linewidth=linewidths[t], alpha=1)
    # 外周圆环图例
    class_names = ['Collinsella', 'Faecalibacterium', 'Clostridium', 'Blautia', 'Gemmiger', 'Bacteroides', 'Prevotella',
                    'Lachnospira', 'Roseburia', 'Bifidobacterium', 'Bilophila', 'Dorea', 'Oscillospira', 'Streptococcus',
                    'Anaerostipes', 'Parabacteroides', 'Phascolarctobacterium', 'Coprococcus', 'Ruminococcus', 'Veillonella']
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=30, label=l)
                      for c, l in zip(colors, class_names)]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 0.99), loc='upper right', frameon=False)
    plt.axis('equal')
# 使用示例
df = pd.read_csv('D:\AAARavenResults\gut_net\differ\Train\FC\GS2_FC.csv',header=None)
draw_chord_plot(df)
# plt.title('Abnormal Relationships of Genus-level Gut Flora between GS1 and HCs')
plt.savefig('D:\AAARavenResults\gut_net\differ\Train\FC\GS2_FC_colorbar.tiff',dpi=300)
plt.show()
#####