# Date： 2023/5/29 02:13
# Author: Mr. Q
# Introduction：使用GridSearchCV进行参数搜索找到最佳化参数的随机森林

import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from subprocess import call
from IPython.display import Image
from sklearn import tree
import graphviz
import dtreeviz
from sklearn.tree import plot_tree
#解决中文乱码问题
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# #查找自己電腦的字體，本機為Mac，選用了庫中的'Arial'字體
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['/System/Library/Fonts/supplemental/Arial.ttf']
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
pd.set_option('display.max_columns', None)  # 让pandas显示所有列

start_time = time.time()
print(start_time)
print("======================================")
# 读取数据集
CRM = pd.read_csv("/Users/wood/Desktop/UXLab/dataclean_v2/ML/RF_input/input-1.csv")

CRM.columns = ['BNB_age(D)', 'BNB_I', "R_bnb'", "F_bnb'", 'cluster']

# 进行非数值型数据处理
typeseries = CRM.dtypes
lbl = preprocessing.LabelEncoder()
for i in typeseries.index:
    if typeseries[i] == 'object':
        CRM[i] = lbl.fit_transform(CRM[i].astype(str))

# 划分训练集和测试集
CRM_y = CRM["cluster"].copy()
CRM_x = CRM.drop('cluster', axis=1)
x_train, x_test, y_train, y_test = train_test_split(CRM_x, CRM_y, test_size=0.1, random_state=42)

# 定义随机森林分类器
rf = RandomForestClassifier()

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],  # 决策树的数量
    'max_depth':[None, 5, 10],  # 决策树的最大深度
    'min_samples_split': [2, 5, 10],  # 内部节点再划分所需的最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶子节点最少样本数
    'max_features': ['auto', 'sqrt']  # 最大特征数
}

# 使用GridSearchCV进行参数搜索
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)
# 最佳参数-2 : {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}

# 使用最佳参数的模型进行预测
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(x_test)

# 输出多分类预测的各项检测指标
print("混淆矩陣:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

# # 显示混淆矩阵和分类报告
# print("分類報告")
# print(classification_report(y_test, y_pred))

# 特征重要性可视化
feature_importances = best_rf.feature_importances_
feature_names = CRM_x.columns

plt.barh(feature_names, feature_importances)
plt.xlabel('特徵重要性')
plt.ylabel('特徵')
plt.title('努力型-隨機森林特徵重要性')
plt.show()
# 打印具体数值
for i, importance in enumerate(feature_importances):
    print(f"特徵 '{feature_names[i]}': {importance:.4f}")
plt.show()

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('預測標籤')
plt.ylabel('真實標籤')
plt.title('努力型-混淆矩陣')
plt.show()
# 打印分類報告
classification_rep = classification_report(y_test, y_pred)
print("分類報告:\n", classification_rep)

print("======================================")
#決策樹可視化
# tree = best_rf.estimators_[5]
print("—————————————————1———————————————————")
# plt.figure(figsize=(10, 10))
# # 這裡設置深度，以讓圖像化顯示更加完整；, rounded=True, precision=2是額外加的
# plot_tree(tree, feature_names=feature_names, class_names=[str(c) for c in best_rf.classes_], filled=True, max_depth=3, rounded=True, precision=2)
# plt.show()
print("—————————————————2———————————————————")
estimator = best_rf.estimators_[5]
dot_data = export_graphviz(estimator, out_file=None, feature_names=feature_names, class_names=[str(c) for c in best_rf.classes_], filled=True)
graph = graphviz.Source(dot_data)
graph.render("/Users/wood/Desktop/UXLab/dataclean_v2/ML/RF_input/tree-努力型-MM")  # 保存为tree.pdf文件
print("—————————————————3———————————————————")

print("======================================")
print("ok!!")

end_time = time.time()
print(end_time)
duration = end_time - start_time
print(f"程式執行時間為 {duration:.2f} 秒")
