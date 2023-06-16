# 《基于心力衰竭患者数据的早期死亡风险预测模型》



（报告大致分六部分内容：摘要，绪论，方法/模型，实验，总结：

- **摘要**简单介绍背景，方法和实验结果；
- **绪论**主要写：问题背景，发展历程，并简单介绍本工作；
- **方法/模型**包含：问题定义（通常指定义问题中的各种变量），详细讲解你的方法原理或者模型结构，如有需要写以写明实现细节；
- **实验**主要写：数据集的介绍；数据集下的实验结果；如有需要可以补充消融实验，详尽分析实验现象。
- **总结**你的工作，并展望可能的改进。





## 数据集介绍

这个数据集最初由Ahmed等人在2017年发布，作为他们对巴基斯坦费萨拉巴德心脏病学院和联合医院心力衰竭患者生存分析的补充[3]。2020年，Chicco 和 Jurman 访问并分析了该数据集，使用一系列机器学习技术来预测心力衰竭[4]。Kaggle 上托管的数据集引用了这些作者及其研究论文。

该数据集包含299名匿名患者的健康记录，并具有12种临床和生活方式特征。它主要包括105名女性和194名男性心力衰竭患者的临床和生活方式特征。

本文分析的数据取自于 [Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) 平台分享的心血管疾病数据集，共有13个字段299条病人诊断记录。具体的字段概要如下：

|           特征           |                   解释                   |     度量单位     |
| :----------------------: | :--------------------------------------: | :--------------: |
|           age            |                病人的年龄                |        年        |
|         anaemia          |               病人是否贫血               |      布尔值      |
| creatinine_phosphokinase |           血液中肌酸肌酶的水平           |      mcg/L       |
|         diabetes         |             病人是否患糖尿病             |      布尔值      |
|    ejection_fraction     | 病人的心脏每次收缩时血液离开心脏的百分比 |      百分比      |
|   high_blood_pressure    |             病人是否患高血压             |      布尔值      |
|        platelets         |                血小板浓度                | kiloplatelets/mL |
|     serum_creatinine     |             血清中肌肽的浓度             |      mg/dL       |
|       serum_sodium       |              血清中纳的浓度              |      mEq/L       |
|           sex            |                病人的性别                |      布尔值      |
|         smoking          |             病人是否有吸烟史             |      布尔值      |
|           time           |                时间跟踪期                |        天        |
|  DEATH_EVENT（目标值）   |        病人是否在时间跟踪期内死亡        |      布尔值      |

> mcg/L: 每升微克. 
>
> mg/mL: 每微升毫克  
>
> mEq/L: 每升当量



## 探索性数据分析

### 基本数据统计

本文将数据集的13个特征分别划分为连续型特征，离散型特征和目标值。为了达成这个目的，在代码中本文将重新排列和重命名一些特征值，添加另一个称为辅助值的特征。

代码展示如下：

```python
# 读取数据集
data = pd.read_csv('dataset/dataset.csv')
# 重命名为中文
data = data.rename(columns={'age': '年龄', 'time': '时间', 'sex': '性别',
                            'smoking': '吸烟史',
                            'diabetes': '糖尿病',
                            'anaemia': '贫血',
                            'platelets': '血小板计数',
                            'high_blood_pressure': '高血压',
                            'creatinine_phosphokinase': '肌酸激酶',
                            'ejection_fraction': '射血分数',
                            'serum_creatinine': '血清肌肽',
                            'serum_sodium': '血清钠浓度',
                            'DEATH_EVENT': '是否死亡'})

data = data.assign(辅助值=pd.Series(1, index=data.index))

# 打印数据信息
print(tabulate(data.info(), headers='keys', tablefmt='psql'))
```

数据信息展示如下：

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 299 entries, 0 to 298
Data columns (total 14 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   年龄      299 non-null    float64
 1   贫血      299 non-null    int64  
 2   肌酸激酶    299 non-null    int64  
 3   糖尿病     299 non-null    int64  
 4   射血分数    299 non-null    int64  
 5   高血压     299 non-null    int64  
 6   血小板计数   299 non-null    float64
 7   血清肌肽    299 non-null    float64
 8   血清钠浓度   299 non-null    int64  
 9   性别      299 non-null    int64  
 10  吸烟史     299 non-null    int64  
 11  时间      299 non-null    int64  
 12  是否死亡    299 non-null    int64  
 13  辅助值     299 non-null    int64  
dtypes: float64(3), int64(11)
memory usage: 32.8 KB
```

从基本数据信息我们可以观察出所使用的数据集中没有缺失值，年龄，肌酸激酶，射血分数，血小板计数，血清肌肽和时间是连续性特征，贫血，糖尿病，高血压，性别和吸烟史是离散型特征，而是否死亡属于目标变量。

本文对目标变量“是否死亡”的简单统计如下：

![image-20230522164402063](https://cdn.infinityday.cn//typora/image-20230522164402063.png)

使用代码如下：

```python
death_num = data['是否死亡'].value_counts()
labels = ['幸存者', '死亡']
sizes = death_num.values
colors = ['green', 'red']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('心力衰竭患者是否死亡的人数分布')
plt.show()
```

根据对目标变量“是否死亡”绘制的饼图，我们观察得到该数据集存在样本不平衡的问题，这个问题我们将在模型构建阶段解决。在接下来的章节中，我们将对除目标变量以外的连续型特征和离散型特征进行统计分析。

### 连续型特征统计分析

在连续型特征统计分析中，因为随访时间这一特征属于客观因素，不属于病人自身的问题而直接引起或诱导心力衰竭死亡，本文在连续型特征统计分析中将时间这一特征暂时去掉不作分析。

数据集中有许多连续型特征，因此本文使用`df.describe()`查看它们的描述性统计。使用代码如下：

```python
cols = ['年龄', '血小板计数', '肌酸激酶', '射血分数', '血清钠浓度', '血清肌肽']
print(tabulate(data[cols].describe().round(2), headers='keys', tablefmt='psql'))
```

连续型特征统计表格展示如下：

|       | 年龄  | 血小板计数 | 肌酸激酶 | 射血分数 | 血清钠浓度 | 血清肌肽 |
| ----- | ----- | ---------- | -------- | -------- | ---------- | -------- |
| count | 299   | 299        | 299      | 299      | 299        | 299      |
| mean  | 60.83 | 263358     | 581.84   | 38.08    | 136.63     | 1.39     |
| std   | 11.89 | 97804.2    | 970.29   | 11.83    | 4.41       | 1.03     |
| min   | 40    | 25100      | 23       | 14       | 113        | 0.5      |
| 25%   | 51    | 212500     | 116.5    | 30       | 134        | 0.9      |
| 50%   | 60    | 262000     | 250      | 38       | 137        | 1.1      |
| 75%   | 70    | 303500     | 582      | 45       | 140        | 1.4      |
| max   | 95    | 850000     | 7861     | 80       | 148        | 9.4      |

为了更加清晰的展示连续型数据以及分析结果，本文采用增强箱线图来展示值的分布和分散情况。在增强箱线图中，中间的黑线是中位数，两端的菱形表示异常值。

![image-20230521154003267](https://cdn.infinityday.cn//typora/image-20230521154003267.png)

结合统计数据与增强箱线图可视化过程，我们可以得到以下结论：

- 年龄：患者的平均年龄为61岁（mean = 60.83），大多数患者（75%）在70岁以下且40岁以上。
- 血小板计数：这是一种负责修复受损血管的血细胞。在该数据集中患者的血小板数量介于25100到850000之间，平均值为263358。根据正常人的血小板计数为100,000-300,000千个/毫升[5]，在我们的数据集中大多数患者（75%）的血小板计数都处于此范围。
- 肌酸激酶：这是一种存在于血液中并有助于修复受损组织的酶，它是一种指标，用于衡量心肌损伤的程度，高水平的CPK意味着心力衰竭或损伤。男性正常水平为55-170 mcg/L，女性为30-135 mcg/L [7]。在我们的数据集中，由于所有患者都曾经历过心力衰竭，所以平均值（550 mcg/L）和中位数（250 mcg/L）远高于正常水平。
- 射血分数：这是每次收缩时从室腔中泵出多少血液（以百分比表示）。根据人体解剖学：心脏有四个房间，其中心房接收来自身体不同部位的血液，并将其输送回去；左室是最厚实的房间，在向全身泵送血液方面起着重要作用，而右室则将氧合后的静脉回流输送至肺部。在健康成年人中，该比例为55％；降低射出分数意味着值小于40％的心力衰竭[6] 。在我们的数据集中，大部分（75%）的患者的射血分数小于45％，这和预期相符。
- 血清钠浓度：指患者血液中钠离子含量，大于 135 mEq / L 的高水平称为高钠血症，这在心力衰竭患者中很常见[10]。在我们的数据集中，平均值和中位数均大于 135 mEq / L，血清钠的箱线图也明显展示了心力衰竭患者普遍拥有高钠血症。
- 血清肌肽：这是作为肌肉代谢产物之一而产生的废物。尤其在肌肉分解时会增加此类创造素，并被肾脏过滤掉；增加水平则表明心输出量不佳且可能出现肾功能障碍[8]。它是一种用于衡量肾脏功能的程度的指标，正常范围应该介于0.84至1.21 mg/dL [9] 之间，在我们的数据集中，平均值（1.39mg/dL）超过正常范围上限；中位数（1.10mg/dL）处于正常范围。

在对连续型特征的统计数据进行相关分析后，本小节还将分析与统计这些连续型特征之间的关联关系。

![image-20230521161202739](https://cdn.infinityday.cn//typora/image-20230521161202739.png)

根据绘制的散点图和直方图，本文发现了以下结论：

- 幸存者比大多数心力衰竭后死亡的患者有更高的射血分数，同时他们的血清肌肽和肌酸激酶也稍微低一些，并且他们通常年龄在80岁以下。
- 根据每两个特征的散点图可以发现连续型特征之间没有强相关性。
- 根据每个特征的直方图可以发现：连续型特征的数据大致都符合正态分布的形状，这表明数据的可信度较好。

接下来我们将计算 Spearman 相关系数进行验证是否连续型特征之间没有强相关性这一结论。（考虑使用 Spearman 相关系数是因为我们不确定特征值来自哪种分布）

使用代码如下：

```python
# 计算斯皮尔曼相关系数
spearman_corr = data[['年龄', '血小板计数', '射血分数', '肌酸激酶', '血清肌肽',
                      '血清钠浓度', '是否死亡']].corr(method='spearman')
# 显示表格
# print(pd.DataFrame(spearman_corr))
# 绘制热力图显示相关系数
sns.heatmap(spearman_corr, cmap='coolwarm', annot=True)
plt.show()

```

相关系数热力图分析如下：

![image-20230521162428539](https://cdn.infinityday.cn//typora/image-20230521162428539.png)

观察结果显示，所有连续型特征两两之间的相关系数绝对值都非常小，这代表它们之间的相关性不大。但是年龄-血清肌酐和血清肌酐-血清钠之间的相关系数绝对值在0.3左右，明显好于其他特征之间的相关程度。

这是因为：从文献[11]中我们可以看到，随着年龄增长，血清肌酐含量会增加，这解释了它们之间略微正向的关系。文献还告诉我们[12]，在慢性肾脏疾病的情况下，钠/血清肌酐比值较高，这意味着两者之间存在负相关关系。轻微的负相关系数也暗示了患者中肾脏问题的普遍存在。

### 离散型特征统计分析

接下来，本文将进行对离散型特征的统计分析，这包括”吸烟史“，”糖尿病“，”贫血“，”高血压“和“性别”五个特征。

本文首先分析除了“性别”之外的离散性特征（这是由于性别是先天因素，而其他的更多是患者个人生活习惯或者病史），通过观察这四个特征我们可以发现，它们都是属于患者自己的生活习惯或者病史。本文用简单的条形图进行分析如下：

绘制代码：

```python
# 绘制离散型数据的条形图
fig = plt.subplots(figsize=[10, 6])
bar1 = data.吸烟史.value_counts().values
bar2 = data.高血压.value_counts().values
bar3 = data.糖尿病.value_counts().values
bar4 = data.贫血.value_counts().values
ticks = np.arange(0, 3, 2)
width = 0.3
plt.bar(ticks, bar1, width=width, color='teal', label='smoker')
plt.bar(ticks + width, bar2, width=width, color='darkorange', label='high blood pressure')
plt.bar(ticks + 2 * width, bar3, width=width, color='limegreen', label='diabetes')
plt.bar(ticks + 3 * width, bar4, width=width, color='tomato', label='anaemic')
plt.xticks(ticks + 1.5 * width, ['Yes', 'No'])
plt.ylabel('Number of patients')
plt.legend()
plt.title("幸存者与已死亡患者的不同生活习惯特征统计图")
plt.show()
```

![image-20230521164300268](D:/latex%20files/CQUThesis-1.50/figures/four_descrete.png)

通过这个条形图我们可以看出，患有心力衰竭症状的病人具有吸烟史，高血压，糖尿病，贫血症的人数占比是更高的，但是这并不能说明有这些生活习惯和病史就会导致心力衰竭症状的患者的死亡，因此我们还需进一步分析生活习惯和病史特征和目标特征死亡之间的关系。

本文利用 pandas 中的交叉表函数来分别统计分析各个离散型特征与心力衰竭症状病人死亡之间的关系。使用代码如下：

```python
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
        ax.set_title(features[i] + '与死亡之间的关系')

    # 调整子图之间的间距和位置
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # 显示图形
    plt.show()

    
# 可视化 2*2 交叉表
features = ['吸烟史', '高血压', '糖尿病', '贫血']
plot_heatmap(data, features)

# 可视化性别交叉表
ct = pd.crosstab(index=data['性别'], columns=data['是否死亡'], values=data['辅助值'], aggfunc=np.sum, margins=True,
                 normalize='columns').round(2) * 100
sns.heatmap(ct, annot=True, cmap='Blues', fmt='g')
plt.title('性别与患者死亡之间的关系')
plt.show()
```

热力图如下：

![image-20230521170934975](D:/latex%20files/CQUThesis-1.50/figures/crossTable1)

![image-20230521170926833](D:/latex%20files/CQUThesis-1.50/figures/crossTable2)

根据上述热力图数据统计，本文将其归纳为：

- 65%的男性心脏病患者和35%的女性心脏病患者死亡。
- 48%的死亡患者贫血，而41%的幸存者也贫血。
- 42%的死亡患者和42%的幸存者都是糖尿病患者。
- 31%的死亡人数为吸烟者，而33％的幸存者为吸烟者。
- 41％去世人员有高血压，而33％幸存人员也有高血压。

因此本文认为：每个离散型数据特征都对患者是否死亡影响不大，这些离散型数据特征（除了性别）似乎是导致心力衰竭的原因，但不是导致心力衰竭患者死亡的直接原因。生存者和死亡者的生活方式特征分布几乎相似。唯一的区别在于高血压方面，这可能对心脏病患者的生存有更大的影响。



### 总体特征分析

在这个章节中，本文将连续型特征数据和离散型特征数据进行整体特征分析，将五个离散型特征和六个连续型特征（除去时间特征）进行配对后，通过绘制小提琴图来可视化所有特征之间的关系以及其他有效信息。

绘制代码如下：

```python
fig, ax = plt.subplots(6, 5, figsize=[20, 22])
cat_features = ['性别', '吸烟史', '贫血', '糖尿病', '高血压']
num_features = ['年龄', '血清肌肽', '血清钠浓度', '血小板计数', '射血分数', '肌酸激酶']
for i in range(0, 6):
    for j in range(0, 5):
        sns.violinplot(data=data, x=cat_features[j], y=num_features[i], hue='是否死亡', split=True, palette='husl',
                       facet_kws={'despine': False}, ax=ax[i, j])
        ax[i, j].legend(title='死亡', loc='upper center')
plt.show()

```

绘制的小提琴图汇总如下：

![image-20230521175214278](https://cdn.infinityday.cn//typora/image-20230521175214278.png)

本文对小提琴图进行相关归纳总结如下：

- 整体数据曲线大致符合高斯曲线，数据可信度较高
- 性别特征分析：死亡患者中，男性的射血分数似乎比女性低。此外，肌酸磷酸激酶在男性中似乎比女性高。
- 吸烟特征分析：死亡吸烟者的射血分数略低于死亡非吸烟者。存活的吸烟患者肌酸激酶水平似乎比非吸烟者高。
- 贫血特征分析：贫血患者倾向于具有较低的肌酸激酶水平和较高的血清肌肽水平。在贫血患者中，射血分数较低的人死亡率高于幸存者。
- 糖尿病特征分析： 糖尿病患者往往具有较低血清钠水平，并且在死亡群体中，其心脏射血分数也相对更差。
- 高血压特征分析：心脏射血分数在高血压死亡患者中的变化似乎比没有高血压的死亡患者更大。







## 数据预处理

在探索性数据分析一章中，我们从数据集中排除了随访时间（因为它充满随机性），因为本文更希望关注临床特征并尝试发现有意义的内容。

然而，根据存活率与随访时间的折线图我们可以看到，随访时间可能是患者生存的重要因素，并不应该完全从这项研究中剔除。因此，接下来我们将调查加上随访时间这一特征，建立机器学习模型来探寻这些特征与患者生存之间可能存在的关系。

![image-20230604204509616](https://cdn.infinityday.cn//typora/image-20230604204509616.png)

### 样本不平衡

数据不平衡是指在分类任务中，不同类别的样本数量显著不同。这可能导致模型对大多数类别表现良好，但在少数类别上效果较差。为了解决这个问题，我们可以采取以下几种方法：

1. **重采样方法**：通过对样本进行重采样，使得各个类别的样本数量相对均衡。有两种常见的重采样方法：
   - **过采样（Oversampling）**：增加少数类别的样本数量。这可以通过复制少数类别的样本或生成新的样本（如通过插值方法，例如SMOTE算法）来实现。
   - **欠采样（Undersampling）**：减少多数类别的样本数量。这可以通过随机去除多数类别的样本或使用聚类方法将多数类别的样本进行合并来实现。
2. **修改损失函数**：在训练模型时，可以修改损失函数，使其对少数类别的预测错误施加更大的惩罚。这种方法也被称为代价敏感学习（Cost-sensitive learning）。例如，在二元分类问题中，可以为少数类别的样本分配较高的权重。
3. **集成方法**：集成方法可以有效地处理数据不平衡问题。例如：
   - **Bagging**：Bootstrap Aggregating（Bagging）方法可以通过对原始数据进行有放回抽样，生成多个子集，然后训练多个基分类器并进行投票或平均来提高泛化性能。
   - **Boosting**：Boosting方法通过迭代地训练一系列弱分类器，并根据其在训练数据上的表现为它们分配权重，以提高整体模型的性能。AdaBoost和Gradient Boosting是两种常见的Boosting算法。

综上所述，处理数据不平衡的方法有很多，可以根据具体问题和场景选择合适的方法。在本文中，我们使用一种惩罚模型（而不是像SMOTE这样的重采样技术）。这个惩罚模型通过采用简单的加权方案，其权重是类别频率的倒数。并且我们还将使用重复分层k折交叉验证技术，确保我们模型的聚合指标不会过于乐观，并反映训练和测试数据中固有的不平衡性。

### 特征缩放

本文将针对连续型特征进行特征缩放的操作为了让这些连续型特征在同一尺度上进行操作。我们使用`sklearn.preprocessing`中的`StandardScaler()`方法来缩放这些连续变量特征，使其具有平均值0和方差1。以下是特征缩放的代码:

```python
discrete_feature = data[['性别', '吸烟史', '糖尿病', '高血压', '贫血']]

numerical_feature = data[['年龄', '血小板计数', '射血分数', '肌酸激酶', '血清肌肽', '血清钠浓度', '时间']]

features_name = ['性别', '吸烟史', '糖尿病', '高血压', '贫血', '年龄', '血小板计数',
                 '射血分数', '肌酸激酶', '血清肌肽',
                 '血清钠浓度', '时间']

target = data['是否死亡']

# 标准化数值型特征
scaler = StandardScaler()
scaled_feature = pd.DataFrame(scaler.fit_transform(numerical_feature.values),
                              columns=numerical_feature.columns)
scaled_predictors = pd.concat([discrete_feature, scaled_feature], axis=1)
```

连续型特征缩放前的部分数据展示：

![image-20230604203640064](https://cdn.infinityday.cn//typora/image-20230604203640064.png)

连续型特征缩放后的部分数据展示：

![image-20230604203705379](https://cdn.infinityday.cn//typora/image-20230604203705379.png)

### 数据降维

本文采用斯皮尔曼系数计算所有特征与死亡事件的相关性，使用代码如下：

```python
sns.heatmap(data.corr(method='spearman'), annot=True, fmt='.2f', cmap='Reds', xticklabels=data.columns.values,
            yticklabels=data.columns.values, cbar=False)
plt.xticks(rotation=45)
plt.show()
```

绘制的Spearman相关系数图如下：

![image-20230604194110564](https://cdn.infinityday.cn//typora/image-20230604194110564.png)

为了更加直观的观察所有特征与目标变量（是否死亡）的之间的关系，我单独绘制了死亡事件特征的相关系数。

![image-20230604200426688](D:/latex%20files/CQUThesis-1.50/figures/deathCor.png)

我们可以看到，时间，血清肌肽，射血分数，年龄，血清钠浓度与患者是否死亡相关性较高。

接下来，我将使用决策树，随机森林，XGBoost模型来探寻这些特征的重要程度。

可视化特征重要性函数如下：

```python
def visualize_importances(importances, features_name, title: str):
    """
    :param importances: 特征重要性列表
    :param features_name: 特征名称列表
    """
    # Create a sorted list of feature importances and corresponding feature names
    feature_importances = sorted(zip(importances, features_name), reverse=True)
    sorted_features = [f[1] for f in feature_importances]
    sorted_importances = [f[0] for f in feature_importances]

    # Create a horizontal bar chart to visualize the feature importances
    fig, ax = plt.subplots()
    ax.barh(range(len(sorted_importances)), sorted_importances, align='center')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Feature importance')
    ax.set_ylabel('Feature')
    ax.invert_yaxis()
    plt.title(title)
    plt.show()

```

决策树给出的特征重要性：

![image-20230610145242599](D:/latex%20files/CQUThesis-1.50/figures/decisionIpt.png)

随机森林给出的特征重要性：

![image-20230610145249602](D:/latex%20files/CQUThesis-1.50/figures/randomForestIpt.png)

XGBoost给出的特征重要性：

![image-20230610145253330](D:/latex%20files/CQUThesis-1.50/figures/XGBoostIpt.png)

LightGBM给出的特征重要性：

![image-20230610145310775](D:/latex%20files/CQUThesis-1.50/figures/LightGBMIpt.png)

CatBoost给出的特征重要性：

![image-20230610145327320](D:/latex%20files/CQUThesis-1.50/figures/CatBoostIpt.png)

综合以上相关性与特征重要性分析，本文只保留时间，血清肌肽，射血分数作为特征输入，以此来训练相应的机器学习模型。

## 模型构建

到目前为止，本文已经对数据进行了详实的统计分析和预处理，包括样本不平衡和特征缩放。接下来，我们将构建传统的机器学习模型：逻辑回归，支持向量机，决策树，随机森林，朴素贝叶斯分类器，K近邻分类器和XGBoost模型，最后通过10次重复的10折分层交叉验证综合对比这些模型的效果。

k折交叉验证是一种众所周知的迭代验证方法，特别适用于可能不完全代表研究人群的小数据集。该数据集被分成k个子集，模型在前k-1个子集上进行训练，在最后一个第k个子集上进行测试。这个过程重复k次，并计算性能指标的平均值[2]。

当目标标签不平衡时，分层k折交叉验证就派上用场了。由于通常对不平衡目标进行的k折交叉验证可能导致一些训练集只有一个目标标签可供训练，因此需要进行分层操作。换言之，先前的过程重复执行，但这次要确保每个训练集中目标标签比例得到维持[3] [4]。StratifiedKFold是scikit-learn的一个交叉验证策略，主要用于解决分类问题中训练集和测试集分布不平衡问题，从而更准确地估计模型的性能。在交叉验证中，我们将数据集分成训练集和测试集，然后使用训练集来拟合模型，使用测试集来评估模型的性能。StratifiedKFold可以确保在分割数据集时，每个折中的类别分布与整个数据集中的类别分布相同。这在类别分布不平衡的情况下尤其重要。参数n_splits指定了K-fold交叉验证中的折数，shuffle = True表示在分割数据之前对样本进行随机排序。





![image-20230616100257274](D:/latex%20files/CQUThesis-1.50/figures/result.png)

```




-----------------------------逻辑回归--------------------------------
Best parameters found:  {'solver': 'saga', 'penalty': 'l2', 'C': 0.001}
Best accuracy score found:  0.7935057471264367
每次模型拟合时间：0.0026 ± 0.0057
模型准确率：0.79 ± 0.07
模型平衡准确率：0.79 ± 0.08
模型精确度：0.67 ± 0.12
模型召回率：0.77 ± 0.13
模型ROC AUC：0.88 ± 0.07
模型f1-score：0.71 ± 0.10

-----------------------------决策树--------------------------------
Best parameters found:  {'max_depth': 7, 'max_leaf_nodes': None, 'min_samples_leaf': 2, 'min_samples_split': 5}
Best accuracy score found:  0.8196436781609197
每次模型拟合时间：0.0014 ± 0.0043
模型准确率：0.82 ± 0.07
模型平衡准确率：0.82 ± 0.08
模型精确度：0.70 ± 0.12
模型召回率：0.81 ± 0.15
模型ROC AUC：0.83 ± 0.09
模型f1-score：0.74 ± 0.11

-----------------------------SVM--------------------------------
Best parameters found:  {'kernel': 'linear', 'gamma': 0.0069519279617756054, 'C': 1.6101694915254237}
Best accuracy score found:  0.7887931034482757
每次模型拟合时间：0.0027 ± 0.0058
模型准确率：0.79 ± 0.08
模型平衡准确率：0.78 ± 0.08
模型精确度：0.66 ± 0.12
模型召回率：0.77 ± 0.14
模型ROC AUC：0.87 ± 0.07
模型f1-score：0.70 ± 0.10

-----------------------------朴素贝叶斯--------------------------------
每次模型拟合时间：0.0002 ± 0.0016
模型准确率：0.81 ± 0.06
模型平衡准确率：0.74 ± 0.08
模型精确度：0.82 ± 0.15
模型召回率：0.53 ± 0.16
模型ROC AUC：0.86 ± 0.08
模型f1-score：0.63 ± 0.14

-----------------------------KNN--------------------------------
Best parameters found:  {'n_neighbors': 12, 'p': 1}
Best accuracy score found:  0.8628275862068966
每次模型拟合时间：0.0019 ± 0.0049
模型准确率：0.86 ± 0.06
模型平衡准确率：0.82 ± 0.08
模型精确度：0.85 ± 0.11
模型召回率：0.71 ± 0.15
模型ROC AUC：0.91 ± 0.06
模型f1-score：0.76 ± 0.12

-----------------------------XGBoost--------------------------------
Best parameters found:  {'colsample_bytree': 0.7840836333818126, 'gamma': 0.188257872169941, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 3, 'subsample': 0.6286337947573846}
Best accuracy score found:  0.8560344827586208
每次模型拟合时间：0.0277 ± 0.0070
模型准确率：0.86 ± 0.06
模型平衡准确率：0.85 ± 0.07
模型精确度：0.77 ± 0.12
模型召回率：0.82 ± 0.12
模型ROC AUC：0.91 ± 0.06
模型f1-score：0.79 ± 0.09

-----------------------------lightGBM--------------------------------
Best parameters found:  {'colsample_bytree': 0.8217039565936457, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_samples': 24, 'num_leaves': 130, 'reg_alpha': 0.9178373274179173, 'reg_lambda': 0.09976948643662853, 'subsample': 0.6448473897613661}
Best accuracy score found:  0.8467011494252874
每次模型拟合时间：0.0109 ± 0.0073
模型准确率：0.85 ± 0.07
模型平衡准确率：0.84 ± 0.07
模型精确度：0.75 ± 0.12
模型召回率：0.81 ± 0.12
模型ROC AUC：0.91 ± 0.06
模型f1-score：0.77 ± 0.10

-----------------------------catBoost--------------------------------
Best parameters found:  {'random_strength': 1.0, 'min_child_samples': 5, 'learning_rate': 0.1, 'l2_leaf_reg': 5, 'grow_policy': 'SymmetricTree', 'depth': 9, 'bagging_temperature': 0.1}
Best accuracy score found:  0.8696551724137931
每次模型拟合时间：3.3362 ± 0.5912
模型准确率：0.87 ± 0.06
模型平衡准确率：0.85 ± 0.06
模型精确度：0.81 ± 0.12
模型召回率：0.79 ± 0.08
模型ROC AUC：0.93 ± 0.04
模型f1-score：0.80 ± 0.09

-----------------------------随机森林--------------------------------
Best parameters found:  {'max_depth': 10, 'min_samples_split': 4, 'n_estimators': 217}
Best accuracy score found:  0.8560689655172414
每次模型拟合时间：0.2272 ± 0.0092
模型准确率：0.86 ± 0.07
模型平衡准确率：0.83 ± 0.08
模型精确度：0.81 ± 0.13
模型召回率：0.75 ± 0.15
模型ROC AUC：0.92 ± 0.05
模型f1-score：0.77 ± 0.11


```

通过对比多个机器学习模型在心力衰竭医疗数据分类任务中的表现，我们得到了以下结果：

| 模型名称   | 每次拟合时间（平均） | 准确率（平均） | 平衡准确率（平均） | 精确度（平均） | 召回率（平均） | ROC AUC（平均） | f1-score（平均） |
| ---------- | -------------------- | -------------- | ------------------ | -------------- | -------------- | --------------- | ---------------- |
| 逻辑回归   | 0.0023 ± 0.0054      | 0.79 ± 0.08    | 0.78 ± 0.08        | 0.65 ± 0.12    | 0.77 ± 0.14    | 0.88 ± 0.07     | 0.70 ± 0.10      |
| 决策树     | 0.0016 ± 0.0046      | 0.82 ± 0.07    | 0.82 ± 0.08        | 0.70 ± 0.12    | 0.81 ± 0.15    | 0.83 ± 0.09     | 0.74 ± 0.11      |
| SVM        | 0.0035 ± 0.0064      | 0.81 ± 0.07    | 0.79 ± 0.08        | 0.71 ± 0.13    | 0.73 ± 0.14    | 0.88 ± 0.07     | 0.71 ± 0.11      |
| 朴素贝叶斯 | 0.0005 ± 0.0028      | 0.81 ± 0.06    | 0.74 ± 0.08        | 0.82 ± 0.15    | 0.53 ± 0.16    | 0.86 ± 0.08     | 0.63 ± 0.14      |
| KNN        | 0.0016 ± 0.0046      | 0.86 ± 0.06    | 0.82 ± 0.08        | 0.85 ± 0.11    | 0.71 ± 0.15    | 0.91 ± 0.06     | 0.76 ± 0.12      |
| 随机森林   | 0.2825 ± 0.0133      | 0.86 ± 0.06    | 0.83 ± 0.08        | 0.81 ± 0.13    | 0.76 ± 0.14    | 0.91 ± 0.06     | 0.77 ± 0.11      |
| XGBoost    | 0.0388 ± 0.0115      | 0.85 ± 0.06    | 0.84 ± 0.07        | 0.77 ± 0.12    | 0.80 ± 0.13    | 0.91 ± 0.06     | 0.77 ± 0.10      |
| lightGBM   | 0.0115 ± 0.0070      | 0.85 ± 0.08    | 0.84 ± 0.08        | 0.77 ± 0.14    | 0.79 ± 0.13    | 0.90 ± 0.06     | 0.78 ± 0.12      |
| catBoost   | 1.9315 ± 0.3317      | 0.87 ± 0.05    | 0.84 ± 0.06        | 0.81 ± 0.11    | 0.77 ± 0.10    | 0.92 ± 0.04     | 0.79 ± 0.08      |

从数据表中可以看出，对于这项任务，准确率最高的模型是 catBoost，平均每次拟合时间也相对较长；而决策树和 KNN 模型的表现也非常优秀，且拟合时间较短。而与之相比，逻辑回归和 SVM 的拟合时间和表现略逊一筹。朴素贝叶斯表现最不佳，精度和召回率都没有达到高标准。其中随机森林、XGBoost 和 lightGBM 表现比较稳定，但与 catBoost 相比准确度略低一些。

![image-20230610153021844](https://cdn.infinityday.cn//typora/image-20230610153021844.png)

因此，综合性能和效率，我们会推荐使用决策树、KNN 和 catBoost 这些模型进行心力衰竭医疗数据的分类任务。



### 逻辑回归

```python
def logistic(scaled_predictors, target, penalized: bool, k: int):
    """
    构建逻辑回归模型并使用k折分层交叉验证最后评估模型效果

    :param scaled_predictors: 特征变量
    :param target: 目标变量“是否死亡”
    :param penalized: 构建逻辑回归模型时是否引入平衡类别权值，如果 penalized = True, 引入权值，更加注重小类别
    :param k: 分层k折交叉验证
    :return: 结果
    """
    strat_kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=2023)

    if penalized:
        logreg_clf = LogisticRegression(class_weight='balanced')
    else:
        logreg_clf = LogisticRegression()

    # 定义超参数搜索空间
    param_distributions = {
        'C': np.logspace(-3, 3, num=20),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    # 在搜索空间内随机搜索最佳超参数
    randomized_search = RandomizedSearchCV(logreg_clf, param_distributions, cv=strat_kfold, scoring='accuracy',
                                           n_iter=10)
    randomized_search.fit(scaled_predictors, target)

    # 输出最佳超参数和最佳得分
    print("Best parameters found: ", randomized_search.best_params_)
    print("Best accuracy score found: ", randomized_search.best_score_)

    # 使用最佳超参数训练逻辑回归模型，并在测试集上进行预测和评估
    best_logreg = randomized_search.best_estimator_
    score = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    result = cross_validate(best_logreg, scaled_predictors, target, cv=strat_kfold, scoring=score)

    return result

```



### 决策树

```python
def decision_tree(scaled_predictors, target, k):
    """
    :param scaled_predictors: 特征变量
    :param target: 目标变量“是否死亡”
    :param k: 分层k折交叉验证
    :return: 结果和最佳超参数
    """
    strat_kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=2023)

    # 定义决策树模型
    dt_clf = DecisionTreeClassifier(class_weight='balanced')

    # 定义超参数搜索空间
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_leaf_nodes': [None, 5, 10, 20],
    }

    # 在搜索空间内寻找最佳超参数
    grid_search = GridSearchCV(dt_clf, param_grid, cv=strat_kfold, n_jobs=-1, scoring='accuracy')
    grid_search.fit(scaled_predictors, target)

    # 输出最佳超参数和最佳得分
    print("Best parameters found: ", grid_search.best_params_)
    print("Best accuracy score found: ", grid_search.best_score_)

    # 使用最佳超参数训练决策树模型，并在测试集上进行预测和评估
    best_dt_clf = grid_search.best_estimator_
    score = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    result = cross_validate(best_dt_clf, scaled_predictors, target, cv=strat_kfold, scoring=score)

    # 计算特征重要性
    best_dt_clf.fit(scaled_predictors, target)
    importances = best_dt_clf.feature_importances_

    return result, importances

```



### 随机森林

```python
def random_forest(scaled_predictors, target, k: int):
    """
    构建随机森林模型并使用k折分层交叉验证最后评估模型效果

    :param scaled_predictors: 特征变量
    :param target: 目标变量
    :param k: 分层k折交叉验证
    :return: 结果
    """
    strat_kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=2023)

    rf_clf = RandomForestClassifier(class_weight='balanced')

    # 定义超参数的搜索范围
    param_distributions = {
        'n_estimators': randint(200, 300),  # 决策树的数量
        'max_depth': [None, 5, 10],  # 决策树的最大深度
        'min_samples_split': randint(2, 11)  # 每个节点上最小样本的数量
    }

    # 创建随机搜索对象
    randomized_search = RandomizedSearchCV(estimator=rf_clf, param_distributions=param_distributions, cv=strat_kfold,
                                           scoring='accuracy', n_iter=10)

    x = scaled_predictors.values
    y = target.values

    # 在训练数据上进行随机搜索
    randomized_search.fit(x, y)

    # 获取最佳超参数组合
    best_params = randomized_search.best_params_
    # 输出最佳超参数和最佳得分
    print("Best parameters found: ", randomized_search.best_params_)
    print("Best accuracy score found: ", randomized_search.best_score_)

    # 使用最佳超参数重新构建模型
    best_rf_clf = RandomForestClassifier(**best_params)

    # 进行交叉验证并计算特征重要性
    result = cross_validate(best_rf_clf, x, y, cv=strat_kfold,
                            scoring=['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1'])

    # 计算特征重要性
    best_rf_clf.fit(x, y)
    importances = best_rf_clf.feature_importances_

    return result, importances

```



### XGBoost

```python
def xgboost(scaled_predictors, target, penalized: bool, k: int):
    """
    构建XGBoost模型并使用k折分层交叉验证最后评估模型效果，并返回特征重要性

    :param scaled_predictors: 特征变量
    :param target: 目标变量“是否死亡”
    :param penalized: 构建XGBoost模型时是否引入平衡类别权值，如果penalized = true, 引入权值，更加注重小类别
    :param k: 分层k折交叉验证
    :return: 结果和特征重要性
    """
    strat_kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=2023)

    if penalized:
        xgb_clf = xgb.XGBClassifier(scale_pos_weight=len(target[target == 0]) / len(target[target == 1]))
    else:
        xgb_clf = xgb.XGBClassifier()

    # 定义超参数搜索空间
    param_distributions = {
        'max_depth': randint(3, 11),
        'min_child_weight': randint(1, 6),
        'gamma': uniform(0, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'learning_rate': [0.1, 0.01, 0.001],
    }

    # 在搜索空间内随机搜索最佳超参数
    randomized_search = RandomizedSearchCV(xgb_clf, param_distributions, cv=strat_kfold, n_jobs=-1, scoring='accuracy',
                                           n_iter=10)
    randomized_search.fit(scaled_predictors, target)

    # 输出最佳超参数和最佳得分
    print("Best parameters found: ", randomized_search.best_params_)
    print("Best accuracy score found: ", randomized_search.best_score_)

    # 使用最佳超参数训练XGBoost模型，并在测试集上进行预测和评估
    best_xgb_clf = randomized_search.best_estimator_
    score = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    result = cross_validate(best_xgb_clf, scaled_predictors, target, cv=strat_kfold, scoring=score)

    # 计算特征重要性
    best_xgb_clf.fit(scaled_predictors, target)
    importance = best_xgb_clf.feature_importances_

    return result, importance

```



### SVM

```python
def svm_model(scaled_predictors, target, penalty: bool, k: int):
    """
    构建SVM模型并使用k折分层交叉验证最后评估模型效果

    :param scaled_predictors: 特征变量
    :param target: 目标变量“是否死亡”
    :param penalty: 是否引入平衡类别权值，如果 penalty = True, 引入权值，更加注重小类别
    :param k: 分层k折交叉验证
    :return: 结果
    """
    strat_kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=2023)

    if penalty:
        svc = SVC(kernel='rbf', class_weight='balanced')
    else:
        svc = SVC(kernel='rbf')

    # 定义超参数搜索空间
    param_distributions = {
        'C': np.reciprocal(np.linspace(0.1, 10, num=20)),
        'gamma': np.logspace(-3, -1, num=20),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    # 在搜索空间内随机搜索最佳超参数
    randomized_search = RandomizedSearchCV(svc, param_distributions, cv=strat_kfold, scoring='accuracy', n_iter=10)
    randomized_search.fit(scaled_predictors, target)

    # 输出最佳超参数和最佳得分
    print("Best parameters found: ", randomized_search.best_params_)
    print("Best accuracy score found: ", randomized_search.best_score_)

    # 使用最佳超参数训练SVM模型，并在测试集上进行预测和评估
    best_svc = randomized_search.best_estimator_
    score = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    result = cross_validate(best_svc, scaled_predictors, target, cv=strat_kfold, scoring=score)

    return result


```



### 朴素贝叶斯

```python
def naive_bayes(scaled_predictors, target, k: int):
    """
    构建朴素贝叶斯分类器并使用k折分层交叉验证最后评估模型效果

    :param scaled_predictors: 特征变量
    :param target: 目标变量“是否死亡”
    :param k: 分层k折交叉验证
    :return: 结果
    """
    strat_kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=2023)

    nb_clf = GaussianNB()

    x = scaled_predictors.values
    y = target.values
    score = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    result = cross_validate(nb_clf, x, y, cv=strat_kfold,
                            scoring=score)
    return result
```



### KNN

```python
def knn(scaled_predictors, target, penalty: bool, k: int):
    """
    构建KNN模型并使用k折分层交叉验证最后评估模型效果

    :param scaled_predictors: 特征变量
    :param target: 目标变量
    :param penalty: 是否引入平衡类别权值，如果 penalty = True, 引入权值，更加注重小类别
    :param k: 分层k折交叉验证
    :return: 结果
    """
    strat_kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=2023)

    if penalty:
        knn_model = KNeighborsClassifier(weights='distance')
    else:
        knn_model = KNeighborsClassifier()

    # 定义超参数搜索空间
    param_dist = {
        'n_neighbors': list(range(1, 21)),  # 调整k的取值范围
        'p': [1, 2, 3, 4, 5, 6, 7]
    }

    # 在搜索空间内随机搜索最佳超参数
    random_search = RandomizedSearchCV(knn_model, param_distributions=param_dist, cv=strat_kfold, scoring='accuracy',
                                       n_iter=10)
    random_search.fit(scaled_predictors, target)

    # 输出最佳超参数和最佳得分
    print("Best parameters found: ", random_search.best_params_)
    print("Best accuracy score found: ", random_search.best_score_)

    # 使用最佳超参数训练KNN模型，并在测试集上进行预测和评估
    best_knn = random_search.best_estimator_
    score = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    result = cross_validate(best_knn, scaled_predictors, target, cv=strat_kfold, scoring=score)

    return result

```



## 超参数调整





## 结果汇总



模型性能比较

| 模型       | 准确率      | 平衡准确率  | 精确度      | 召回率      | ROC AUC     | f1-score    |
| ---------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 逻辑回归   | 0.79 ± 0.08 | 0.78 ± 0.08 | 0.65 ± 0.12 | 0.77 ± 0.14 | 0.88 ± 0.07 | 0.70 ± 0.10 |
| 决策树     | 0.82 ± 0.08 | 0.82 ± 0.08 | 0.70 ± 0.12 | 0.81 ± 0.14 | 0.83 ± 0.09 | 0.74 ± 0.11 |
| SVM        | 0.82 ± 0.06 | 0.76 ± 0.08 | 0.82 ± 0.15 | 0.59 ± 0.15 | 0.88 ± 0.07 | 0.67 ± 0.12 |
| 朴素贝叶斯 | 0.81 ± 0.06 | 0.74 ± 0.08 | 0.82 ± 0.15 | 0.53 ± 0.16 | 0.86 ± 0.08 | 0.63 ± 0.14 |
| KNN        | 0.86 ± 0.06 | 0.82 ± 0.08 | 0.84 ± 0.12 | 0.71 ± 0.15 | 0.91 ± 0.06 | 0.76 ± 0.11 |
| 随机森林   | 0.85 ± 0.06 | 0.82 ± 0.08 | 0.81 ± 0.14 | 0.74 ± 0.15 | 0.92 ± 0.06 | 0.76 ± 0.11 |
| XGBoost    | 0.85 ± 0.07 | 0.84 ± 0.07 | 0.77 ± 0.13 | 0.82 ± 0.12 | 0.91 ± 0.06 | 0.78 ± 0.10 |

请注意，表格中的数值是平均值和标准差。平均值表示模型在交叉验证中的平均性能，标准差表示模型性能的稳定性。

![image-20230604211345370](https://cdn.infinityday.cn//typora/image-20230604211345370.png)



## 参考文献

[3] Ahmad T, Munir A, Bhatti SH, Aftab M, Ali Raza M. Survival analysis of heart failure patients: a case study. Dataset.* [*https://plos.figshare.com/*](https://plos.figshare.com/) *articles/Survival_analysis_of_heart_failure_patients_A_case_study/ 5227684/1.*

*[4] Chicco and Jurman, BMC Medical Informatics and Decision Making (2020) 20:16*

【5】Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone

【6】[基础检验学/血小板计数 - 医学百科 (yixue.com)](https://yixue.com/基础检验学/血小板计数)

[4]: 
