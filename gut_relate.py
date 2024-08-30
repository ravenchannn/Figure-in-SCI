#脑肠亚型分类代码-脑
#Written by Raven
from __future__ import division

import math

#########################################################
#                                           实验目的：肠相关                                          #
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





# 热图
#热图
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
# gut = pd.read_csv("/mnt/disk1/wyr/result_brain_net/new_relate/BS1_brain_boot.csv")
# scales= pd.read_csv("/mnt/disk1/wyr/result_brain_net/new_relate/BS1_mp_boot.csv")
gut1 = pd.read_csv("/mnt/disk1/wyr/result_brain_net/new_relate/Brain_BS1.csv")
scales1= pd.read_csv("/mnt/disk1/wyr/result_brain_net/new_relate/MP_BS1.csv")
columns1 = gut1.columns
columns3 = scales1.columns
# result_data = []
# # gut_blood
# for col1 in columns1:
#     for col2 in columns2:
#         data1 = gut[col1]
#         data2 = blood[col2]
#         spearman_corr, p_value = spearmanr(data1, data2)
#         result_data.append({
#             'Gut_Column': col1,
#             'Blood_Column': col2,
#             'Spearman_Rho': spearman_corr,
#             'P_Value': p_value,
#         })
# result_df = pd.DataFrame(result_data)
# print(result_df)
# result_df.to_csv('/mnt/disk1/wyr/result_gut_net/relate/GS3_Gut_Blood.csv', index=False)
# result_df['Gut_Column'] = pd.Categorical(result_df['Gut_Column'], categories=gut.columns, ordered=True)
# result_df['Blood_Column'] = pd.Categorical(result_df['Blood_Column'], categories=blood.columns, ordered=True)
# heatmap_data = result_df.pivot_table(index='Gut_Column', columns='Blood_Column', values='Spearman_Rho')
# kk = result_df.pivot_table(index='Gut_Column', columns='Blood_Column', values='P_Value')
# heatmap_data = heatmap_data.reindex(index=gut.columns, columns=blood.columns)
# # plt.figure(figsize=(25,8.5))
# annot_data = kk.applymap(lambda x: '*' if x < 0.05 else '**' if x < 0.01 else '')
# annot_kws = {'size': 10}
# sns.heatmap(heatmap_data, annot=annot_data, fmt="", cmap='magma', linewidths=0.5,annot_kws=annot_kws,square=True,cbar_kws={"aspect":100})
# plt.xlabel('Blood_Column',fontsize=20)
# plt.ylabel('Gut_Column',fontsize=20)
# plt.xticks(fontsize=10,rotation=45, rotation_mode='anchor', ha='right')
# plt.yticks(fontsize=10)
# plt.tight_layout()
# plt.savefig('/mnt/disk1/wyr/result_gut_net/relate/GS3_gut_blood.tiff',dpi=300)
# plt.show()
#
# result_data = []
# # gut_scales
# for col1 in columns1:
#     for col2 in columns3:
#         data1 = gut[col1]
#         data2 = scales[col2]
#         spearman_corr, p_value = spearmanr(data1, data2)
#         result_data.append({
#             'Gut_Column': col1,
#             'Scales_Column': col2,
#             'Spearman_Rho': spearman_corr,
#             'P_Value': p_value,
#         })
# result_df = pd.DataFrame(result_data)
# print(result_df)
# result_df.to_csv('/mnt/disk1/wyr/result_gut_net/relate/GS3_Gut_Scales.csv', index=False)
# result_df['Gut_Column'] = pd.Categorical(result_df['Gut_Column'], categories=gut.columns, ordered=True)
# result_df['Scales_Column'] = pd.Categorical(result_df['Scales_Column'], categories=scales.columns, ordered=True)
# heatmap_data = result_df.pivot_table(index='Gut_Column', columns='Scales_Column', values='Spearman_Rho')
# kk = result_df.pivot_table(index='Gut_Column', columns='Scales_Column', values='P_Value')
# heatmap_data = heatmap_data.reindex(index=gut.columns, columns=scales.columns)
# plt.figure(figsize=(25,8.5))
# annot_data = kk.applymap(lambda x: '*' if x < 0.05 else '**' if x < 0.01 else '')
# annot_kws = {'size': 10}
# sns.heatmap(heatmap_data, annot=annot_data, fmt="", cmap='magma', linewidths=0.5,annot_kws=annot_kws,square=True,cbar_kws={"aspect":100})
# plt.xlabel('Scales_Column',fontsize=20)
# plt.ylabel('Gut_Column',fontsize=20)
# plt.xticks(fontsize=10,rotation=45, rotation_mode='anchor', ha='right')
# plt.yticks(fontsize=10)
# plt.savefig('/mnt/disk1/wyr/result_gut_net/relate/GS3_gut_scales.tiff',dpi=300)
# plt.show()
#
# result_data = []
# # scales_blood
# for col1 in columns2:
#     for col2 in columns3:
#         data1 = blood[col1]
#         data2 = scales[col2]
#         spearman_corr, p_value = spearmanr(data1, data2)
#         result_data.append({
#             'Blood_Column': col1,
#             'Scales_Column': col2,
#             'Spearman_Rho': spearman_corr,
#             'P_Value': p_value,
#         })
# result_df = pd.DataFrame(result_data)
# print(result_df)
# result_df.to_csv('/mnt/disk1/wyr/result_gut_net/relate/GS3_Blood_Scales.csv', index=False)
# result_df['Blood_Column'] = pd.Categorical(result_df['Blood_Column'], categories=blood.columns, ordered=True)
# result_df['Scales_Column'] = pd.Categorical(result_df['Scales_Column'], categories=scales.columns, ordered=True)
# heatmap_data = result_df.pivot_table(index='Blood_Column', columns='Scales_Column', values='Spearman_Rho')
# kk = result_df.pivot_table(index='Blood_Column', columns='Scales_Column', values='P_Value')
# heatmap_data = heatmap_data.reindex(index=blood.columns, columns=scales.columns)
# plt.figure(figsize=(25,12))
# annot_data = kk.applymap(lambda x: '*' if x < 0.05 else '**' if x < 0.01 else '')
# annot_kws = {'size': 10}
# sns.heatmap(heatmap_data, annot=annot_data, fmt="", cmap='magma', linewidths=0.5,annot_kws=annot_kws,square=True,cbar_kws={"aspect":100})
# plt.xlabel('Scales_Column',fontsize=20)
# plt.ylabel('Blood_Column',fontsize=20)
# plt.xticks(fontsize=10,rotation=45, rotation_mode='anchor', ha='right')
# plt.yticks(fontsize=10)
# plt.savefig('/mnt/disk1/wyr/result_gut_net/relate/GS3_blood_scales.tiff',dpi=300)
# plt.show()


# 散点图颜色：GS1#87bba4，GS2#9e3150，GS3#fbc864,GHC#54686f，BS1#e73847，BS2#457b9d，BHC#1d3557,#BGS1#B1C44D,#bg2#EEB0AF,BGHC#243c57
# gut_scales
for col1 in columns1:
    for col2 in columns3:
        data1 = gut1[col1]
        data2 = scales1[col2]
        data3 = gut1[col1]
        data4 = scales1[col2]
        pearson_corr1, p_value1 = pearsonr(data1, data2)
        pearson_corr2, p_value2 = pearsonr(data3, data4)
        num_comparisons = len(columns1) * len(columns3)  # 总的比较数量
        alpha_bonf = p_value2 * num_comparisons
        print('p:{alpha_bonf}', alpha_bonf)
        if alpha_bonf < 0.05:
            sns.regplot(x=data1.iloc[0:150], y=data2.iloc[0:150],
                        scatter_kws={"s": 50, "color": "#e73847", "edgecolor": "none"},
                        line_kws={"color": "#e73847", "alpha": 1})
            plt.xlabel(f'{col1}', fontsize=15)
            plt.ylabel(f'{col2}', fontsize=15)
            text = f'r= {pearson_corr1:.2f}  p={"{:.2e}".format(alpha_bonf)}'
            plt.annotate(text, xy=(1.1, -0.2), xycoords='axes fraction', ha='right', va='bottom', fontsize=15,bbox=dict(boxstyle='round', alpha=0))
            plt.tight_layout()
            plt.savefig(f'/mnt/disk1/wyr/result_brain_net/new_relate/Boot and Bonfer/BS1_{col1}_{col2}_regression.tiff', dpi=300)
            plt.show()