import os
import sys
import json
import time
import logging
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
import warnings

from resnet import resnet50

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from pretrain_utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
import numpy as np
from pretrain_eval import train, init_network
from importlib import import_module
import argparse
from pretrain_utils import build_dataset, build_iterator, get_time_dif

import random
import re
import pandas as pd
from nltk.corpus import stopwords
import csv
import os
import logging
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch
from transformers import AutoTokenizer
import numpy as np
# from args import args
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision import models

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()
logger = logging.getLogger('resbert.py')

stats_template = 'Epoch {epoch_idx}\n' \
                 '{mode} Accuracy: {acc}\n' \
                 '{mode} F1: {f1}\n' \
                 '{mode} ave. F1: {f1_ave}\n' \
                 '{mode} Recall: {recall}\n' \
                 '{mode} ave. Recall: {recall_ave}\n' \
                 '{mode} Precision: {prec}\n' \
                 '{mode} ave. Precision: {prec_ave}\n' \
                 '{mode} Loss: {loss}\n'


def train_model():
    # 数据准备——bert

    dataset = 'THUCNews'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    # dev_iter_bert = find_bertdata(dev_iter, args.feature)
    dev_iter_bert = train_iter
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 数据准备——resnet

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # 数据预处理。transforms提供一系列数据预处理方法
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪
                                     transforms.RandomHorizontalFlip(),  # 水平方向随机反转
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # 标准化
        "val": transforms.Compose([transforms.Resize(256),  # 图像缩放
                                   transforms.CenterCrop(224),  # 中心裁剪
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 获取数据集根目录(即当前代码文件夹路径)
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".\\"))
    # 获取flower图片数据集路径
    image_path = os.path.join(data_root, "data_set")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # ImageFolder是一个通用的数据加载器，它要求我们以root/class/xxx.png格式来组织数据集的训练、验证或者测试图片。
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"), transform=data_transform["val"])
    val_num = len(val_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    class_dict = dict((val, key) for key, val in flower_list.items())  # 将字典中键值对翻转。此处翻转为 {'0':daisy,...}

    # 将class_dict编码成json格式文件
    json_str = json.dumps(class_dict, indent=1)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 4  # 设置批大小。batch_size太大会报错OSError: [WinError 1455] 页面文件太小，无法完成操作。
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print("Using batch_size={} dataloader workers every process.".format(num_workers))

    # 加载训练集和测试集
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size,
                                   num_workers=num_workers, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=True)
    print("Using {} train_images for training, {} test_images for validation.".format(train_num, val_num))
    print()

    # 评价指标
    best_dev_acc = 0  # 准确率
    num_epochs = 4  # 批大小

    # 模型加载
    model = x.Model(config).to(config.device)
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer_bert = BertAdam(optimizer_grouped_parameters,
                              lr=config.learning_rate,
                              warmup=0.05,
                              t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()

    # resnet
    # 加载预训练权重
    # download url: https://download.pytorch.org/models/resnet34-b627a593.pth
    net = resnet50()
    model_weight_path = "./resnet50-11ad3fa6.pth"  # 预训练权重
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)

    # 改变in_channel符合fc层的要求，调整output为数据集类别5
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    net.to(device)

    # 损失函数
    loss_function = nn.CrossEntropyLoss()

    # 优化器
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer_resnet = optim.Adam(params, lr=0.0001)

    epochs = 10  # 训练迭代次数
    best_acc = 0.0
    save_path = '.\\resNet50_3.pth'  # 当前模型训练好后的权重参数文件保存路径
    batch_num = len(train_loader)  # 一个batch中数据的数量
    total_time = 0  # 统计训练过程总时间
    loss_fn = nn.CrossEntropyLoss().to(device)
    train_bar = tqdm(train_loader, file=sys.stdout)
    n_total_steps = len(train_bar)

    # 模型训练
    for epoch in range(num_epochs):

        net.train()
        train_loss = 0.0

        truths = []
        predictions = []
        trues = []
        preds = []


        logger.info('Training epoch: {}'.format(epoch))
        for (inputs, labels, paths), batch_ids in tqdm( train_iter):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # run bert
            input_ids = batch_ids[0].to(device)
            att_masks = batch_ids[1].to(device)
            labels_bert = batch_ids[2].to(device)
            model.zero_grad()
            _, logits_bert = model(input_ids)
            # record preds, trues
            _pred = logits_bert.cpu().data.numpy()
            _label = labels.cpu().data.numpy()
            preds.extend(_pred)
            trues.extend(_label)

            # run resnet
            outputs = net(inputs)

            # compute average
            ave = (outputs + logits_bert) / 2
            # _, pred = torch.max(ave, axis=1)
            predictions.extend(ave)
            truths.extend(labels)
            loss = loss_fn(ave, labels)
            train_loss += loss.item()

            loss.backward()
            # 反向传播 resnet
            optimizer_resnet.zero_grad()
            optimizer_resnet.step()

            # 反向传播 bert
            optimizer_bert.step()

        predictions = torch.as_tensor(predictions).cpu()
        truths = torch.as_tensor(truths).cpu()
        train_loss = train_loss / n_total_steps

        train_acc, train_f1, train_recall, train_prec, train_f1_ave, train_recall_ave, train_prec_ave = calculate_metrics(
            truths, predictions, average=None)
        print(stats_template.format(mode='train', epoch_idx=epoch, acc=train_acc, f1=train_f1, f1_ave=train_f1_ave,
                                    recall=train_recall,
                                    recall_ave=train_recall_ave, prec=train_prec, prec_ave=train_prec_ave,
                                    loss=train_loss))

        # validation
        # acc, f1, recall, prec, f1_ave, recall_ave, prec_ave, valid_loss = eval_avgbertresnet(dev_iter, dev_iter_bert,
        # net, model,
        # loss_fn, device)

        # write results to csv
        with open('123.csv', 'a') as output_file:
            cw = csv.writer(output_file, delimiter='\t')
            cw.writerow(["train-" + str(epoch),
                         '%0.4f' % train_acc,
                         '%0.4f' % train_prec_ave,
                         '%0.4f' % train_recall_ave,
                         '%0.4f' % train_f1_ave,
                         '%0.4f' % train_f1[0],
                         '%0.4f' % train_f1[1],
                         '%0.4f' % train_prec[0],
                         '%0.4f' % train_prec[1],
                         '%0.4f' % train_recall[0],
                         '%0.4f' % train_recall[1]])
            ''' cw.writerow(["valid-" + str(epoch),
                         '%0.4f' % acc,
                         '%0.4f' % prec_ave,
                         '%0.4f' % recall_ave,
                         '%0.4f' % f1_ave,
                         '%0.4f' % f1[0],
                         '%0.4f' % f1[1],
                         '%0.4f' % prec[0],
                         '%0.4f' % prec[1],
                         '%0.4f' % recall[0],
                         '%0.4f' % recall[1]]) '''

        # save_model(model, checkpoint_dir)
        if train_acc > best_acc:
            best_acc = train_acc
            # model.state_dict()保存学习到的参数
            torch.save(net.state_dict(), save_path)  # 保存当前最高的准确度
            torch.save(model.state_dict(), save_path)


def calculate_metrics(label, pred, average='binary'):
    logging.debug('Expected: \n{}'.format(label[:20]))
    logging.debug('Predicted: \n{}'.format(pred[:20]))

    acc = round(accuracy_score(label, pred), 4)
    f1 = [round(score, 4) for score in f1_score(label, pred, average=average)]
    recall = [round(score, 4) for score in recall_score(label, pred, average=average)]
    prec = [round(score, 4) for score in precision_score(label, pred, average=average)]

    f1_ave = f1_score(label, pred, average='weighted')
    recall_ave = recall_score(label, pred, average='weighted')
    prec_ave = precision_score(label, pred, average='weighted')

    return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave


def eval_avgbertresnet(dev_iter, dev_iter_bert, model_resnet, model_bert, loss_fn, device):
    device = torch.device(device)
    n_total_steps = len(dev_iter)
    model_bert.to(device)
    model_resnet.to(device)
    model_bert.eval()
    model_resnet.eval()
    dev_loss = 0
    predictions = []
    truths = []
    trues = []
    preds = []

    # forward pass
    with torch.no_grad():
        for (inputs, labels, paths), batch_ids in tqdm(zip(dev_iter, dev_iter_bert)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # run bert
            input_ids = batch_ids[0].to(device)
            att_masks = batch_ids[1].to(device)
            labels_bert = batch_ids[2].to(device)
            _, logits_bert = model_bert(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels_bert)
            # run resnet
            outputs = model_resnet(inputs)

            # compute average
            ave = (outputs + logits_bert) / 2
            _, pred = torch.max(ave, axis=1)
            predictions.extend(pred)
            truths.extend(labels)
            loss = loss_fn(ave, labels)
            dev_loss += loss.item()

    predictions = torch.as_tensor(predictions).cpu()
    truths = torch.as_tensor(truths).cpu()

    dev_loss = dev_loss / n_total_steps
    acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = calculate_metrics(truths, predictions, average=None)
    print(stats_template
          .format(mode='valid', epoch_idx='__', acc=acc, f1=f1, f1_ave=f1_ave, recall=recall,
                  recall_ave=recall_ave, prec=prec, prec_ave=prec_ave, loss=dev_loss))
    return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave, dev_loss


def save_model(model_resnet, model_bert, checkpoint_dir_resnet, checkpoint_dir_bert):
    checkpoint_dir_resnet = os.path.join(checkpoint_dir_resnet,
                                         '{model_name}_epochs_{epoch}_lr_{lr}.bin'.format(model_name='resnet',
                                                                                          epoch=args.epochs,
                                                                                          lr=args.lr))
    torch.save(model_resnet.state_dict(), checkpoint_dir_resnet)
    model_bert.save_pretrained(checkpoint_dir_bert)
    return


if __name__ == '__main__':
    train_model()
