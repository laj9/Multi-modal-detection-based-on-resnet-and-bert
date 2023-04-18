"""
# 训练脚本
#
"""

import os
import sys
import json
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
import warnings

from resnet import resnet50

warnings.filterwarnings("ignore")


def train_model():
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
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 10  # 训练迭代次数
    best_acc = 0.0
    save_path = '.\\resNet50_3.pth'  # 当前模型训练好后的权重参数文件保存路径
    batch_num = len(train_loader)  # 一个batch中数据的数量
    total_time = 0  # 统计训练过程总时间

    for epoch in range(epochs):
        # 开始迭代训练和测试
        start_time = time.perf_counter()  # 计算训练一个epoch的时间

        # train
        net.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)  # tqdm是Python进度条库，可以在Python长循环中添加一个进度条提示信息。

        for step, data in enumerate(train_bar):
            train_images, train_labels = data
            train_images = train_images.to(device)
            train_labels = train_labels.to(device)

            optimizer.zero_grad()  # 梯度置零。清空之前的梯度信息
            outputs = net(train_images)  # 前向传播
            #print(outputs)
            loss = loss_function(outputs, train_labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            train_loss += loss.item()  # 将计算的loss累加到train_loss中

            # desc：str类型，作为进度条说明，在进度条右边
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}.".format(epoch + 1, epochs, loss)

        predictions = []
        truths = []
        # validate
        net.eval()
        val_acc = 0.0
        val_bar = tqdm(val_loader, file=sys.stdout)
        f1 = 0.0
        recall = 0.0
        prec = 0.0

        with torch.no_grad():
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_y = net(val_images)  # 前向传播
                predict_y = torch.max(val_y, dim=1)[1]  # 在维度为1上找到预测Y的最大值，第0个维度是batch
                # 计算测试集精度。predict_y与val_labels进行比较(true=1, False=0)的一个batch求和，所有batch的累加精度值
                val_acc += torch.eq(predict_y, val_labels).sum().item()
                print(predict_y)
                # 新增
                predictions.extend(predict_y)
                truths.extend(val_labels)
                # 结尾

                logging.debug('Expected: \n{}'.format(truths[:20]))
                logging.debug('Predicted: \n{}'.format(predictions[:20]))

                # acc = round(accuracy_score(truths, predictions), 4)
                # f1 = [round(score, 4) for score in f1_score(truths, predictions, average='binary')]
                # recall = [round(score, 4) for score in recall_score(truths, predictions, average='binary')]
                # prec = [round(score, 4) for score in precision_score(truths, predictions, average='binary')]

                f1_ave = f1_score(truths, predictions, average='weighted')
                recall_ave = recall_score(truths, predictions, average='weighted')
                prec_ave = precision_score(truths, predictions, average='weighted')

                # acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = calculate_metrics(truths, predictions,
                #                                                                         average=None)

                # f1_ave += f1_score(predict_y, val_labels, average='weighted')
                # recall_ave += recall_score(val_labels, predict_y, average='weighted')
                # prec_ave += precision_score(val_labels, predict_y, average='weighted')

                val_bar.desc = "valid epoch[{}/{}].".format(epoch + 1, epochs)

        # 打印epoch数据结果
        print(val_num)
        print(batch_num)
        val_accurate = val_acc / val_num
        f1score = f1_ave
        val_recall = recall_ave
        val_precision =  prec_ave
        print(
            "[epoch {:.0f}] train_loss: {:.3f}  val_accuracy: {:.3f}  f1score: {:.3f}  recall: {:.3f}  precision: {:.3f}"
            .format(epoch + 1, train_loss / batch_num, val_accurate, f1score, val_recall, val_precision))

        epoch_time = time.perf_counter() - start_time  # 计算训练一个epoch的时间
        print("epoch_time: {}".format(epoch_time))
        total_time += epoch_time  # 统计训练过程总时间
        print()

        # 调整测试集最优精度
        if val_accurate > best_acc:
            best_acc = val_accurate
            # model.state_dict()保存学习到的参数
            torch.save(net.state_dict(), save_path)  # 保存当前最高的准确度

    # 将训练过程总时间转换为h:m:s格式打印
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print("Total_time: {:.0f}:{:.0f}:{:.0f}".format(h, m, s))

    print('Finished Training!')

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


if __name__ == '__main__':
    train_model()
