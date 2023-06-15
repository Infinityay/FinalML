# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
from pylab import *
from tabulate import tabulate

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法正常显示的问题
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def plot_heatmap(data, features):
    # 计算交叉表并进行格式化
    cts = []
    for feature in features:
        ct = pd.crosstab(index=data[feature], columns=data['是否死亡'], values=data['辅助值'], aggfunc=np.sum,
                         margins=True, normalize='columns').round(2) * 100
        cts.append(ct)

    # 创建画布和子图
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # 绘制子图
    for i, ax in enumerate(axs.flat):
        sns.heatmap(cts[i], annot=True, cmap='Blues', fmt='g', ax=ax)
        ax.set_title(features[i] + '与患者死亡之间的关系')

    # 调整子图之间的间距和位置
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # 显示图形
    plt.show()


# 读取数据
data = pd.read_csv('dataset/processed_data.csv')

data = data.assign(辅助值=pd.Series(1, index=data.index))

print(tabulate(data.info(), headers='keys', tablefmt='psql'))

cols = ['年龄', '血小板计数', '肌酸激酶', '射血分数', '血清钠浓度', '血清肌肽']
print(tabulate(data[cols].describe().round(2), headers='keys', tablefmt='psql'))

# --------------------1.绘制连续型变量的箱线图--------------------
fig, ax = plt.subplots(3, 2, figsize=[10, 10])
num_features_set1 = ['年龄', '血清肌肽', '血清钠浓度']
num_features_set2 = ['血小板计数', '射血分数', '肌酸激酶']
for i in range(0, 3):
    sns.boxenplot(data[num_features_set1[i]], ax=ax[i, 0], color='steelblue')
    sns.boxenplot(data[num_features_set2[i]], ax=ax[i, 1], color='steelblue')
    ax[i, 0].set_title(num_features_set1[i])
    ax[i, 1].set_title(num_features_set2[i])

sns.pairplot(data[['年龄', '血小板计数', '射血分数', '肌酸激酶', '血清肌肽', '血清钠浓度', '是否死亡']],
             hue='是否死亡', palette='husl', corner=True)
plt.show()

# --------------------2.绘制连续型变量之间，是否死亡特征的分布图，散点图--------------------
spearman_corr = data[['年龄', '血小板计数', '射血分数', '肌酸激酶', '血清肌肽',
                      '血清钠浓度']].corr(method='spearman')
sns.heatmap(spearman_corr, cmap='coolwarm', annot=True)
plt.show()

# ---------------------3.绘制患者的不同习惯或病史特征统计图 --------------------
fig = plt.subplots(figsize=[10, 6])
bar1 = data.吸烟史.value_counts().values
bar2 = data.高血压.value_counts().values
bar3 = data.糖尿病.value_counts().values
bar4 = data.贫血.value_counts().values
ticks = np.arange(0, 3, 2)
width = 0.3
plt.bar(ticks, bar1, width=width, color='teal', label='是否有吸烟史')
plt.bar(ticks + width, bar2, width=width, color='darkorange', label='是否有高血压')
plt.bar(ticks + 2 * width, bar3, width=width, color='limegreen', label='是否有糖尿病')
plt.bar(ticks + 3 * width, bar4, width=width, color='tomato', label='是否有贫血')
plt.xticks(ticks + 1.5 * width, ["是", "否"])
plt.ylabel('患者数量')
plt.legend()
plt.title("患者的不同生活习惯或病史特征统计图")
plt.show()

# ------------------- 4.绘制'吸烟史', '高血压', '糖尿病', '贫血'与患者是否死亡的交叉关系 -------------------
features = ['吸烟史', '高血压', '糖尿病', '贫血']
plot_heatmap(data, features)

# -------------------5.绘制性别与患者是否死亡的交叉关系 -------------------
ct = pd.crosstab(index=data['性别'], columns=data['是否死亡'], values=data['辅助值'], aggfunc=np.sum, margins=True,
                 normalize='columns').round(2) * 100
sns.heatmap(ct, annot=True, cmap='Blues', fmt='g')
plt.title('性别与患者死亡之间的关系')
plt.show()

# ------------------- 6.绘制连续型特征和离散型特征直接的相关小提琴图-------------------
fig, ax = plt.subplots(6, 5, figsize=[20, 22])
cat_features = ['性别', '吸烟史', '贫血', '糖尿病', '高血压']
num_features = ['年龄', '血清肌肽', '血清钠浓度', '血小板计数', '射血分数', '肌酸激酶']
for i in range(0, 6):
    for j in range(0, 5):
        sns.violinplot(data=data, x=cat_features[j], y=num_features[i], hue='是否死亡', split=True, palette='husl',
                       facet_kws={'despine': False}, ax=ax[i, j])
        ax[i, j].legend(title='死亡', loc='upper center')
plt.show()

# ------------------7.绘制心力衰竭患者是否死亡的饼图-----------------------------------
death_num = data['是否死亡'].value_counts()
labels = ['幸存者', '死亡']
sizes = death_num.values
colors = ['green', 'red']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('心力衰竭患者是否死亡的人数分布')
plt.show()

# -------------------8.绘制所有特征值的Spearman相关系数-------------------
data = data.drop(columns=['辅助值'])
sns.heatmap(data.corr(method='spearman'), annot=True, fmt='.2f', cmap='Reds', xticklabels=data.columns.values,
            yticklabels=data.columns.values, cbar=False)
plt.xticks(rotation=45)
plt.show()

# -------------------9.绘制死亡事件的相关系数-------------------
corr = data.corrwith(data['是否死亡'], method='spearman').to_frame()
corr.columns = ['是否死亡']
corr['绝对值'] = corr['是否死亡'].abs()  # 添加绝对值列
corr_sorted = corr.sort_values(by='绝对值', ascending=False)  # 按照绝对值排序
plt.subplots(figsize=(5, 5))
sns.heatmap(corr_sorted[['是否死亡']], annot=True, cmap=colors, linewidths=0.4, linecolor='black')
plt.xticks(rotation=0)  # 不旋转X轴标签
plt.yticks(rotation=0)  # 旋转Y轴标签
plt.title('死亡事件的相关系数（按绝对值从大到小排列）')
plt.show()

# -------------------10.绘制存活率折线图-------------------
# 将时间转换为月份
data['时间'] = data['时间'] // 30
# 计算每个月的存活率
survival_rate = 1 - data.groupby('时间')['是否死亡'].mean()
plt.plot(survival_rate.index, survival_rate.values)
plt.xlabel('随访时间（月）')
plt.ylabel('存活率')
plt.title('存活率折线图')
plt.show()
