# -*- coding: utf-8 -*-
# @Project : FinalML
# @Time    : 2023/5/26 16:56
# @Author  : infinityay
# @File    : ModelsComparation.py
# @Software: PyCharm
# @Contact me: https://github.com/Infinityay or stu.lyh@outlook.com
# @Comment :


import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法正常显示的问题
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import numpy as np
import matplotlib.pyplot as plt
import catboost as cb


def visualize_results(results, metrics, model_names):
    n_metrics = len(metrics)
    n_models = len(results)
    bar_width = 0.35
    opacity = 0.8
    colors = plt.cm.get_cmap('tab10', n_models)

    fig, ax = plt.subplots(figsize=(12, 8))
    index = np.arange(n_metrics)

    f1_scores = []
    for i, result in enumerate(results):
        means = [np.mean(result[f'test_{metric}']) for metric in metrics]
        f1_scores.append((model_names[i], means[metrics.index('f1')]))

    f1_scores.sort(key=lambda x: x[1], reverse=True)  # 按照f1分数降序排序

    for i, (model_name, _) in enumerate(f1_scores):
        result = results[model_names.index(model_name)]
        means = [np.mean(result[f'test_{metric}']) for metric in metrics]
        stds = [np.std(result[f'test_{metric}']) for metric in metrics]
        ax.bar(index + i * bar_width / n_models, means, bar_width / n_models, alpha=opacity, color=colors(i), yerr=stds,
               capsize=5)

    ax.set_xlabel('指标')
    ax.set_ylabel('得分')
    ax.set_title('模型性能比较')
    ax.set_xticks(index + (bar_width / n_models) * (n_models - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend([model_name for model_name, _ in f1_scores])
    plt.tight_layout()
    plt.show()


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
        'C': np.logspace(-3, 3, num=30),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    # 在搜索空间内随机搜索最佳超参数
    randomized_search = RandomizedSearchCV(logreg_clf, param_distributions, cv=strat_kfold, scoring='accuracy',
                                           n_iter=100)
    randomized_search.fit(scaled_predictors, target)

    # 输出最佳超参数和最佳得分
    print("Best parameters found: ", randomized_search.best_params_)
    print("Best accuracy score found: ", randomized_search.best_score_)

    # 使用最佳超参数训练逻辑回归模型，并在测试集上进行预测和评估
    best_logreg = randomized_search.best_estimator_
    score = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    result = cross_validate(best_logreg, scaled_predictors, target, cv=strat_kfold, scoring=score)

    return result


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
        'criterion': ['gini', 'entropy', 'log_loss']
    }

    # 在搜索空间内寻找最佳超参数
    grid_search = GridSearchCV(dt_clf, param_grid, cv=strat_kfold, scoring='accuracy')
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


def lightgbm(scaled_predictors, target, penalized: bool, k: int):
    """
    构建LightGBM模型并使用k折分层交叉验证最后评估模型效果，并返回特征重要性

    :param scaled_predictors: 特征变量
    :param target: 目标变量“是否死亡”
    :param penalized: 构建LightGBM模型时是否引入平衡类别权值，如果penalized = true, 引入权值，更加注重小类别
    :param k: 分层k折交叉验证
    :return: 结果和特征重要性
    """
    strat_kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=2023)

    if penalized:
        lgb_clf = lgb.LGBMClassifier(class_weight='balanced')
    else:
        lgb_clf = lgb.LGBMClassifier()

    # 定义超参数搜索空间
    param_distributions = {
        'max_depth': randint(3, 11),
        'min_child_samples': randint(20, 101),
        'num_leaves': randint(30, 151),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'learning_rate': [0.1, 0.01, 0.001],
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }

    # 在搜索空间内随机搜索最佳超参数
    randomized_search = RandomizedSearchCV(lgb_clf, param_distributions, cv=strat_kfold, n_jobs=-1, scoring='accuracy',
                                           n_iter=10)
    randomized_search.fit(scaled_predictors, target)

    # 输出最佳超参数和最佳得分
    print("Best parameters found: ", randomized_search.best_params_)
    print("Best accuracy score found: ", randomized_search.best_score_)

    # 使用最佳超参数训练LightGBM模型，并在测试集上进行预测和评估
    best_lgb_clf = randomized_search.best_estimator_
    score = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    result = cross_validate(best_lgb_clf, scaled_predictors, target, cv=strat_kfold, scoring=score)

    # 计算特征重要性
    best_lgb_clf.fit(scaled_predictors, target)
    importance = best_lgb_clf.feature_importances_

    return result, importance


def catboost(scaled_predictors, target, penalized: bool, k: int):
    """
    构建CatBoost模型并使用k折分层交叉验证最后评估模型效果，并返回特征重要性

    :param scaled_predictors: 特征变量
    :param target: 目标变量“是否死亡”
    :param penalized: 构建CatBoost模型时是否引入平衡类别权值，如果penalized = true, 引入权值，更加注重小类别
    :param k: 分层k折交叉验证
    :return: 结果和特征重要性
    """
    strat_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=2023)

    if penalized:
        cat_clf = CatBoostClassifier(scale_pos_weight=(len(target) - sum(target)) / sum(target), silent=True)
    else:
        cat_clf = CatBoostClassifier(silent=True)

    param_distributions = {
        'depth': [3, 5, 7, 9],  # 决策树的最大深度
        'learning_rate': [0.1, 0.01, 0.001],  # 学习率
        'l2_leaf_reg': [1, 3, 5, 10],  # L2正则化项的系数
        'bagging_temperature': [0.1, 0.5, 1.0],  # Bagging温度
        'random_strength': [0.1, 0.5, 1.0],  # 随机强度
        'grow_policy': ['SymmetricTree', 'Depthwise'],  # 树生长策略
        'min_child_samples': [1, 5, 10]  # 叶子节点中的最小样本数
    }

    # 在搜索空间内随机搜索最佳超参数
    randomized_search = RandomizedSearchCV(cat_clf, param_distributions, cv=strat_kfold, n_jobs=-1, scoring='accuracy',
                                           n_iter=10)
    randomized_search.fit(scaled_predictors, target)

    # 输出最佳超参数和最佳得分
    print("Best parameters found: ", randomized_search.best_params_)
    print("Best accuracy score found: ", randomized_search.best_score_)

    # 使用最佳超参数训练CatBoost模型，并在测试集上进行预测和评估
    best_cat_clf = randomized_search.best_estimator_
    score = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    result = cross_validate(best_cat_clf, scaled_predictors, target, cv=strat_kfold, scoring=score)

    # 计算特征重要性
    feature_importance = best_cat_clf.get_feature_importance()

    return result, feature_importance


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


def knn(scaled_predictors, target, penalty: bool, k: int):
    """
    使用网格搜索构建KNN模型并使用k折分层交叉验证最后评估模型效果

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
    param_grid = {
        'n_neighbors': list(range(1, 21)),  # 调整k的取值范围
        'p': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    # 在搜索空间内进行网格搜索
    grid_search = GridSearchCV(knn_model, param_grid=param_grid, cv=strat_kfold, scoring='accuracy')
    grid_search.fit(scaled_predictors, target)

    # 输出最佳超参数和最佳得分
    print("Best parameters found: ", grid_search.best_params_)
    print("Best accuracy score found: ", grid_search.best_score_)

    # 使用最佳超参数训练KNN模型，并在测试集上进行预测和评估
    best_knn = grid_search.best_estimator_
    score = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    result = cross_validate(best_knn, scaled_predictors, target, cv=strat_kfold, scoring=score)

    return result


def printResult(result, method: str):
    """
    打印交叉验证返回的评估结果
    :param result: 使用交叉验证返回的评估结果
    :param method: 使用的模型方法
    :return: null
    """

    # 访问每个指标的平均值和标准差
    accuracy_mean = result['test_accuracy'].mean()
    accuracy_std = result['test_accuracy'].std()

    balanced_accuracy_mean = result['test_balanced_accuracy'].mean()
    balanced_accuracy_std = result['test_balanced_accuracy'].std()

    precision_mean = result['test_precision'].mean()
    precision_std = result['test_precision'].std()

    recall_mean = result['test_recall'].mean()
    recall_std = result['test_recall'].std()

    roc_auc_mean = result['test_roc_auc'].mean()
    roc_auc_std = result['test_roc_auc'].std()

    f1_mean = result['test_f1'].mean()
    f1_std = result['test_f1'].std()

    fitime_mean = result['fit_time'].mean()
    fitime_mean_std = result['fit_time'].std()

    # 打印结果
    print("-----------------------------" + method + "--------------------------------")
    print("每次模型拟合时间：{:.4f} ± {:.4f}".format(fitime_mean, fitime_mean_std))
    print("模型准确率：{:.2f} ± {:.2f}".format(accuracy_mean, accuracy_std))
    print("模型平衡准确率：{:.2f} ± {:.2f}".format(balanced_accuracy_mean, balanced_accuracy_std))
    print("模型精确度：{:.2f} ± {:.2f}".format(precision_mean, precision_std))
    print("模型召回率：{:.2f} ± {:.2f}".format(recall_mean, recall_std))
    print("模型ROC AUC：{:.2f} ± {:.2f}".format(roc_auc_mean, roc_auc_std))
    print("模型f1-score：{:.2f} ± {:.2f}".format(f1_mean, f1_std))
    print()


data = pd.read_csv('dataset/processed_data.csv')
discrete_feature = data[['性别', '吸烟史', '糖尿病', '高血压', '贫血']]
numerical_feature = data[['年龄', '血小板计数', '射血分数', '肌酸激酶', '血清肌肽', '血清钠浓度', '时间']]
features_name = ['性别', '吸烟史', '糖尿病', '高血压', '贫血', '年龄', '血小板计数',
                 '射血分数', '肌酸激酶', '血清肌肽',
                 '血清钠浓度', '时间']
target = data['是否死亡']

# 标准化归一化后的特征
scaler = StandardScaler()
scaled_feature = pd.DataFrame(scaler.fit_transform(numerical_feature.values),
                              columns=numerical_feature.columns)

# 合并特征
scaled_predictors = pd.concat([discrete_feature, scaled_feature], axis=1)
scaled_predictors.to_csv("dataset/final_data.csv")

# 保留特征
features_name = ['血清肌肽', '射血分数', '时间']
scaled_predictors = scaled_predictors[features_name]

# 参数设置
k = 10

# =============逻辑回归=============
logistic_result = logistic(scaled_predictors, target, True, k=k)
printResult(logistic_result, '逻辑回归')

# =============决策树=============
decision_tree_result, decision_tree_features = decision_tree(scaled_predictors, target, k=k)
printResult(decision_tree_result, '决策树')
visualize_importances(importances=decision_tree_features, features_name=features_name, title="决策树给出的特征重要性")

# =============SVM=============
svm_result = svm_model(scaled_predictors, target, True, k)
printResult(svm_result, 'SVM')

# =============朴素贝叶斯============
naive_bayes_result = naive_bayes(scaled_predictors, target, k=k)
printResult(naive_bayes_result, '朴素贝叶斯')

# =============KNN============
knn_result = knn(scaled_predictors, target, k=k, penalty=True)
printResult(knn_result, 'KNN')

# =============XGBoost============
xgboost_result, feature_importances_ = xgboost(scaled_predictors, target, True, k=k)
printResult(xgboost_result, 'XGBoost')
visualize_importances(importances=feature_importances_, features_name=features_name,
                      title="XGBoost模型给出的特征重要性")

# =============LightGBM============
lightGBM_result, feature_importances_ = lightgbm(scaled_predictors, target, True, k=k)
printResult(lightGBM_result, 'lightGBM')
visualize_importances(importances=feature_importances_, features_name=features_name,
                      title="lightGBM模型给出的特征重要性")

# =============catBoost============
catBoost_result, feature_importances_ = catboost(scaled_predictors, target, True, k=k)
printResult(catBoost_result, 'catBoost')
visualize_importances(importances=feature_importances_, features_name=features_name,
                      title="catBoost模型给出的特征重要性")

# =============随机森林============
random_result, random_forest_importances = random_forest(scaled_predictors, target, k=k)
printResult(random_result, '随机森林')
visualize_importances(importances=random_forest_importances, features_name=features_name,
                      title="随机森林给出的特征重要性")

# # 不要随机森林
# results = [logistic_result, decision_tree_result, svm_result, naive_bayes_result, xgboost_result,
#            knn_result, lightGBM_result, catBoost_result]
# metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
# model_names = ['Logistic regression', 'Decsion tree', 'SVM', 'Naive bayes'
#     , 'XGBoost', 'KNN', 'lightGBM', 'catBoost']
# visualize_results(results, metrics, model_names=model_names)

# 可视化结果
results = [logistic_result, decision_tree_result, svm_result, random_result, naive_bayes_result, xgboost_result,
           knn_result, lightGBM_result, catBoost_result]
metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'f1']
model_names = ['Logistic regression', 'Decsion tree', 'SVM', 'Random forest', 'Naive bayes'
    , 'XGBoost', 'KNN', 'LightGBM', 'CatBoost']
visualize_results(results, metrics, model_names=model_names)
