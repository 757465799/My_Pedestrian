# !/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random

trainval_percent = 0.9  # 验证集+训练集占总比例多少
train_percent = 7/9  # 训练数据集占验证集+训练集比例多少
jsonfilepath = '../Dataset/labelme_json/'
txtsavepath = './'
total_xml = os.listdir(jsonfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('./trainval.txt', 'w')
ftest = open('./test.txt', 'w')
ftrain = open('./train.txt', 'w')
fval = open('./val.txt', 'w')

for i in list:
    name = total_xml[i][:-5] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
