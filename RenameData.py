# -*- coding: utf-8 -*-
# @Project : FinalML
# @Time    : 2023/5/23 11:53
# @Author  : infinityay
# @File    : RenameData.py
# @Software: PyCharm
# @Contact me: https://github.com/Infinityay or stu.lyh@outlook.com
# @Comment :


import pandas as pd

# 数据读取
data = pd.read_csv('dataset/dataset.csv')
# 数据项重命名
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
# 保存数据
data.to_csv('dataset/processed_data.csv')
