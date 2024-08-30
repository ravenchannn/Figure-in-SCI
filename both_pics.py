

import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from scipy.stats import zscore

df = pd.read_csv("/mnt/disk1/wyr/result_brain_net/MCCB&PANSS_radar.csv")
categories = list(df)[1:]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)
# 绘制 HC 数据
values = df.loc[2].drop('label').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, color='#1d3557',alpha=1,linewidth=1.5, linestyle='solid', label="HC")
# ax.fill(angles, values, color='#1d3557', alpha=0.5)
# 绘制 BS1 数据
values = df.loc[1].drop('label').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, color='#457B9D', alpha=1,linewidth=1.5, linestyle='solid', label="BS2")
# ax.fill(angles, values, color='#457B9D', alpha=0.5)
# 绘制 BS2 数据
values = df.loc[0].drop('label').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, color='#e73847', alpha=1,linewidth=1.5, linestyle='solid', label="BS1")
# ax.fill(angles, values, color='#e73847', alpha=0.5)

## 绘制 BS1 数据
# values = df.loc[2].drop('label').values.flatten().tolist()
# values += values[:1]
# ax.plot(angles, values, color='#fbc864', alpha=1,linewidth=1, linestyle='solid', label="GS3")
# ax.fill(angles, values, color='#fbc864', alpha=0.75)

# 设置图例顺序
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,1,0]  # BS1, BS2, HC 的顺序
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', bbox_to_anchor=(-0.1, 0.1))
plt.xticks(angles[:-1], categories)
ax.spines['polar'].set_color('white')
custom_labels = ['M1','M2','M3','M4','M5','MT/5','P','N','G','T/2']
ax.set_xticklabels(custom_labels, fontsize=18)
plt.tight_layout()
plt.savefig("/mnt/disk1/wyr/result_brain_net/radar_BS.tiff",dpi=300)
plt.show()






import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
ICA = pd.read_csv("/mnt/disk1/wyr/result_bg_sz/ICA_all.csv",header=None)
label = pd.read_csv("/mnt/disk1/wyr/result_bg_sz/label_3types.csv")
ICA=np.array(ICA)
label=np.array(label)
x=ICA[:,0]
y=ICA[:,1]
label1 = label[:,1]
label2 = label[:,2]
label3 = label[:,3]
# plt.figure(figsize=(8, 6))
# plt.scatter(x,y,color='k',alpha=0.5)
# plt.savefig('/mnt/disk1/wyr/result_bg_sz/ICA.tiff',dpi=300)
# plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(x[label1 == 0], y[label1 == 0], label='BS1', color='#e73847',alpha=0.8)
plt.scatter(x[label1 == 1], y[label1 == 1], label='BS2', color='#457b9d',alpha=0.8)
# plt.scatter(x[label3 == 0], y[label3 == 0], label='BGS1', color='#B1C44D',marker='*',s=7.5,alpha=0.8)
# plt.scatter(x[label3 == 1], y[label3 == 1], label='BGS2', color='#EEB0AF',marker='x',s=7.5,alpha=0.8)
plt.savefig('/mnt/disk1/wyr/result_bg_sz/ICA_BS_BGS.tiff',dpi=300)
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(x[label2 == 0], y[label2 == 0], label='GS1', color='#87bba4')
plt.scatter(x[label2 == 1], y[label2 == 1], label='GS2', color='#9e3150')
plt.scatter(x[label2 == 2], y[label2 == 2], label='GS3', color='#fbc864')
plt.savefig('/mnt/disk1/wyr/result_bg_sz/ICA_GS.tiff',dpi=300)
plt.show()

# # GS
# for i in range(3):
#     mask2 = label2 == i
#     x_min, x_max = np.min(x[mask2]), np.max(x[mask2])  # 计算 x 轴范围
#     y_min, y_max = np.min(y[mask2]), np.max(y[mask2])  # 计算 y 轴范围
#     x_mean, y_mean = np.mean(x[mask2]), np.mean(y[mask2])  # 计算均值作为椭圆中心
#     x_span, y_span = x_max - x_min, y_max - y_min  # 计算 x 和 y 的极值范围
#     x_std, y_std = np.std(x[mask2]), np.std(y[mask2])  # 计算标准差作为椭圆长短轴长度
#     ell = Ellipse((x_mean, y_mean), 2 * x_span, 2 * y_span,angle=[0,15,-15][i], alpha=0.5, color=['#87bba4', '#9e3150', '#fbc864'][i], linewidth=2, fill=True)
#     plt.gca().add_patch(ell)
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
ICA = pd.read_csv("/mnt/disk1/wyr/result_bg_sz/BGS1_ICA.csv",header=None)
# # Fisher-Z变换
# def fisher_z_transform(data):
#     return np.arctanh(data)
# # 对数据进行Fisher-Z变换
# ICA_transformed = fisher_z_transform(ICA)
# ICA=np.array(ICA_transformed)
# # 计算Z分数
# def calculate_z_scores(data):
#     mean = np.mean(data, axis=0)
#     std_dev = np.std(data, axis=0)
#     z_scores = (data - mean) / std_dev
#     return z_scores
# ICA_z_scores = calculate_z_scores(ICA)
# ICA=np.array(ICA_z_scores)
ICA=np.array(ICA)
x=ICA[:,0]
y=ICA[:,1]
label1=np.array([1,
0,
1,
1,
1,
1,
0,
0,
1,
1,
1,
1,
1,
1,
1,
1,
0,
0,
0,
1,
0,
1,
1,
1,
1,
0,
1,
1,
1,
1,
1,
1])
label2=np.array([0,
0,
2,
0,
1,
0,
2,
1,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
2,
0,
0,
0,
0,
0,
0,
0,
2,
2,
0,
0,
0])
# label3=np.array([0,
# 1,
# 0,
# 0,
# 0,
# 0,
# 1,
# 0,
# 0,
# 0,
# 0,
# 1,
# 0,
# 0,
# 1,
# 1,
# 0,
# 1,
# 0,
# 1,
# 0,
# 0,
# 0,
# 1,
# 0,
# 0,
# 0,
# 0,
# 0,
# 0,
# 0,
# 1,
# 0,
# 0,
# 0,
# 1,
# 1,
# 0,
# 0,
# 0,
# 0,
# 0,
# 0])
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='black',alpha=0.8)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.savefig('/mnt/disk1/wyr/result_bg_sz/ALLLL.tiff',dpi=300)
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(x[label1 == 0], y[label1 == 0], label='BS1', color='#e73847',alpha=0.8)
plt.scatter(x[label1 == 1], y[label1 == 1], label='BS2', color='#457b9d',alpha=0.8)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.savefig('/mnt/disk1/wyr/result_bg_sz/BGS1_V_BS.tiff',dpi=300)
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(x[label2 == 0], y[label2 == 0], label='GS1', color='#87bba4',alpha=0.8)
plt.scatter(x[label2 == 1], y[label2 == 1], label='GS2', color='#9e3150',alpha=0.8)
plt.scatter(x[label2 == 2], y[label2 == 2], label='GS3', color='#fbc864',alpha=0.8)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.savefig('/mnt/disk1/wyr/result_bg_sz/BGS1_V_GS.tiff',dpi=300)
plt.show()
# plt.figure(figsize=(8, 6))
# plt.scatter(x[label3 == 0], y[label3 == 0], label='BGS1', color='#B1C44D',alpha=0.8)
# plt.scatter(x[label3 == 1], y[label3 == 1], label='BGS2', color='#EEB0AF',alpha=0.8)
# plt.xlabel('Variable 1')
# plt.ylabel('Variable 2')
# plt.savefig('/mnt/disk1/wyr/result_bg_sz/BGS.tiff',dpi=300)
# plt.show()






# 雷达图求面积
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
from math import pi, sin, cos
# 读取数据
df = pd.read_csv("/mnt/disk1/wyr/result_brain_net/MCCB&PANSSall_radar.csv")
max_values = df.drop(columns=['label']).max()
normalized_df = df.copy()
for col in normalized_df.columns[1:]:
    normalized_df[col] = normalized_df[col] / max_values[col]
# 定义类别
categories = list(normalized_df)[1:]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
# 计算多边形面积的函数
def polygon_area(coords):
    x, y = zip(*coords)
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(len(coords) - 1)))
# 准备存储面积的列表
areas_data = []
# 计算每个个体的PANSS和MCCB面积
for index, row in normalized_df.iterrows():
    label = row['label']
    values = row.drop('label').values.flatten().tolist()
    values += values[:1]
    # 分割PANSS和MCCB
    panss_values = values[6:] + [values[6]]  # 前5个类别+起始点
    mccb_values = values[:6] + [values[0]]  # 后5个类别+起始点
    panss_angles = angles[6:] + [angles[6]]
    mccb_angles = angles[:6] + [angles[0]]
    panss_coords = [(v * cos(a), v * sin(a)) for a, v in zip(panss_angles, panss_values)]
    mccb_coords = [(v * cos(a), v * sin(a)) for a, v in zip(mccb_angles, mccb_values)]
    panss_area = polygon_area(panss_coords)
    mccb_area = polygon_area(mccb_coords)
    areas_data.append([label, panss_area, mccb_area])
# 创建新的 DataFrame 并保存为 CSV 文件
areas_df = pd.DataFrame(areas_data, columns=['label', 'PANSS_area', 'MCCB_area'])
print(areas_df)
areas_df.to_csv('/mnt/disk1/wyr/result_brain_net/radar_area.csv', index=False)
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from scipy.stats import f_oneway
cluster_demographic = areas_df
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
# 进行 FDR校正
p_values_fdr = multipletests(result_df_all['P-value_all'], alpha=0.05, method='fdr_bh')[1]
result_df_all['P-value (FDR)'] = p_values_fdr
# 进行 Bonferroni 校正
p_values_b = multipletests(result_df_all['P-value_all'], alpha=0.05, method='bonferroni')[1]
result_df_all['P-value (Bonferroni)'] = p_values_b
print('P-Value of all', result_df_all)
result_df_all.to_csv('/mnt/disk1/wyr/result_brain_net/radar_pall.csv',index=False)
# 存储结果的DataFrame
result_df = pd.DataFrame(columns=['Feature', 'Group1', 'Group2', 'KW Statistic', 'P-value'])
# 对每两组进行两两比较
for group1, group2 in combinations(cluster_demographic[category_column_all].unique(), 2):
    for variable in variable_columns_all:
        data_group1 = cluster_demographic.loc[cluster_demographic[category_column_all] == group1, variable]
        data_group2 = cluster_demographic.loc[cluster_demographic[category_column_all] == group2, variable]
        # 执行 KW 检验
        stat, p_value = f_oneway(data_group1, data_group2)
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
result_df.to_csv('/mnt/disk1/wyr/result_brain_net/radar_p.csv',index=False)








# 桑基图
import plotly.graph_objects as go
# 定义桑基图节点
nodes = [
    "GS1", "GS2", "GS3",  # 左侧节点
    "B-GS1", "B-GS2",     # 中间节点
    "BS1", "BS2"          # 右侧节点
]
# 定义桑基图链接
links = {
    'source': [0, 1, 1, 2, 3, 4, 4],  # 源节点
    'target': [3, 3, 4, 4, 5, 6, 7],  # 目标节点
    'value':  [10, 20, 10, 10, 5, 15, 5]  # 链接值（可调整以匹配实际数据）
}
# 创建桑基图
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes,
        color=["#243c57", "#EEB0AF", "#B1C44D", "#243c57", "#EEB0AF", "#B1C44D", "#243c57", "#B1C44D"]
    ),
    link=dict(
        source=links['source'],
        target=links['target'],
        value=links['value'],
        color=["#243c57", "#EEB0AF", "#B1C44D", "#243c57", "#EEB0AF", "#B1C44D", "#243c57"]
    )
)])
# 更新图表布局
fig.update_layout(title_text="桑基图示例", font_size=10)
fig.show()
