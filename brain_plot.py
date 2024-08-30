#脑肠亚型分类代码-图
#Written by Raven
from __future__ import division

from imp import reload
#########################################################
#                        实验目的：绘制nii脑图全脑图（全脑比较）                         #
#                                        written by Raven                                         #
#########################################################
#首先在excel中计算BS1\BS2和HC差异
#利用这个代码绘制nii图绘制全脑变化图
#示例在D:\AAARavenResults\brain_net\差异分析结果\example\sMRI





#导入各种包
#####
import numpy as np
import matplotlib.pyplot as plt  #加载画图的包
import pandas as pd  #加载数据处理所用的包
import plotly.io
import scipy.stats as stats  #加载统计学所需的包
from sklearn import preprocessing
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler, MinMaxScaler

font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)    #加载所需中文字体
import warnings  #忽略warnings
warnings.filterwarnings('ignore')
#####





import nibabel as nib
import pandas as pd
import numpy as np

# 加载AAL模板
aal_img = nib.load("/mnt/disk1/wyr_data/aal.nii")


data = [
0.141718000588507,
0.115635494234462,
0.246513271035815,
0.266358871967437,
0.387834132760769,
0.416370619200912,
0.193695354207475,
0.208198151452518,
0.309495803393946,
0.339592959652394,
0.156129814594561,
0.150023084128653,
0.148834314341387,
0.158041583081347,
0.198740915285205,
0.191792614506478,
0.130544462132062,
0.148446300216648,
0.178253746510481,
0.141058013942359,
0.310513791168154,
0.303838933261811,
0.314610714550721,
0.310180350129924,
0.320896960600208,
0.292120691419515,
0.456485362894173,
0.43560120015285,
0.153620447879289,
0.192946482600598,
0.204819271518998,
0.15690192128118,
0,
0,
-0.245196131201042,
-0.189514509612183,
0.112579757749698,
0.124488806963807,
0,
0.186499191248719,
0.249277460439802,
0.292636918762751,
-0.723649092018064,
-0.903968790944146,
-0.586768830297149,
-0.784371864158194,
-0.538468562738897,
-0.782082882447189,
-0.274752791927522,
-0.726016115857452,
0,
-0.553917123647235,
0,
-0.58839790904446,
0,
-0.156325585465469,
0.1066152804901,
0,
0,
-0.342114846490105,
0.198334442489908,
0,
0.181951273995728,
0.088687076324562,
0.209437349216153,
-0.171433102190445,
-0.260014409772331,
-0.394670272282354,
0,
0,
0.181461543356924,
0.19803067670546,
0.117660370877085,
0.103365601497236,
0.158009590250032,
0.124598876451092,
0.10871374587832,
0.122969104442733,
0.12276629360095,
0.177622593133301,
0.10327350804302,
0.0634044748785356,
0.238129122036298,
0.205483038326131,
0.0934270867046624,
0,
0,
0,
0.175746114066184,
0.132774435999239
]


# 创建一个与AAL模板相同形状的数组，初始化为零
new_image_data = np.zeros(aal_img.shape)

# 映射数据到AAL脑区
for region_index, value in enumerate(data):
    # 使用AAL模板的数值作为脑区标签
    region_label = region_index + 1
    new_image_data[aal_img.get_fdata() == region_label] = value

# 创建一个新的NIfTI图像
new_image = nib.Nifti1Image(new_image_data, affine=aal_img.affine)

# 保存新的NIfTI图像
nib.save(new_image, "/mnt/disk1/wyr/result_brain_net/fMRI√/BSs_ALFF.nii")
