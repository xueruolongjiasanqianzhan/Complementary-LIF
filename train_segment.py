# -*-coding: utf-8 -*-
# 作    者：赵广振
# 开发时间：2023/3/30 19:51
import logging
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from logs import (
    plt_img,
    write_info_logs,
    write_target_logs,
    write_params_logs,
    write_best_logs,
)
from test_model.SAMFF_Net import ERTrans
from spikingjelly.clock_driven import functional   # ★ 新增：用于重置 SNN 状态


# -------------------- 设备与训练超参 --------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("torch version: ", torch.__version__)
print("GPU State:", device)

# 定义训练参数
batch_size = 124
learning_rate = 0.0005
num_epochs = 300

step_size = 50
gamma = 0.9


# -------------------- 数据集封装 --------------------
class MyDataset(Dataset):
    def __init__(self, data):
        # .npy 里是一个 dict，包含 'fea' 和 'label'
        self.data = np.load(data, allow_pickle=True).item()

    def __len__(self):
        return len(self.data["fea"])

    def __getitem__(self, index):
        fea = self.data["fea"][index]   # 原始形状 [C, T] = [61, 750]
        label = self.data["label"][index]
        fea = np.expand_dims(fea, axis=0)  # -> [1, C, T]
        # fea = np.transpose(fea, (1, 0))
        return fea, label


# -------------------- 训练函数（已加入 reset_net） --------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0

    for idx, (inputs, labels) in enumerate(dataloader):
        # ★ 每个 batch 开始前，清空所有 LIF 神经元的膜电位等状态
        functional.reset_net(model)

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)             # [B, 2]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    # 学习率调度
    scheduler.step()

    train_loss /= (idx + 1)
    train_acc = float(correct / (batch_size * int(len(dataloader))))

    return train_loss, train_acc


# -------------------- 测试函数（已加入 reset_net） --------------------
def test(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    count = 0

    Recall = 0.0       # 敏感度（召回率）
    Specificity = 0.0  # 特异度
    Precision = 0.0    # 精确度
    F1_Score = 0.0     # F1 分数
    AUC = 0.0          # ROC 曲线下面积

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            # ★ 测试阶段同样需要在每个 batch 前重置状态
            functional.reset_net(model)

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 真实标签 & 预测标签（转到 CPU）
            actual_labels = labels.to("cpu")
            predict_labels = outputs.argmax(1).to("cpu")

            # 累加损失
            test_loss += loss.item()
            # 累加预测正确个数
            correct += (outputs.argmax(1) == labels).sum().item()

            # 混淆矩阵
            conf_matrix = confusion_matrix(actual_labels, predict_labels)
            tp = conf_matrix[1, 1]  # 真正例
            tn = conf_matrix[0, 0]  # 真负例
            fp = conf_matrix[0, 1]  # 假正例
            fn = conf_matrix[1, 0]  # 假负例

            # 召回率
            Recall += recall_score(actual_labels, predict_labels)
            # 特异度
            Specificity += tn / (tn + fp)
            # 精确度（zero_division=1 避免除零报错）
            Precision += precision_score(
                actual_labels, predict_labels, zero_division=1
            )
            # F1
            F1_Score += f1_score(actual_labels, predict_labels)
            # AUC
            probs = F.softmax(outputs, dim=1).to("cpu")
            predicted_probs = probs[:, 1].to("cpu")
            AUC += roc_auc_score(actual_labels, predicted_probs)

            count += 1

    test_loss /= (idx + 1)
    test_acc = float(correct / (batch_size * int(len(dataloader))))
    test_recall = float(Recall / count)
    test_specificity = float(Specificity / count)
    test_precision = float(Precision / count)
    test_f1_score = float(F1_Score / count)
    test_auc = float(AUC / count)

    return (
        test_loss,
        test_acc,
        test_recall,
        test_specificity,
        test_precision,
        test_f1_score,
        test_auc,
    )


# -------------------- 训练主流程（k-fold） --------------------
# Self-data 路径
data_path1 = (
    "/home/guyue/zhao/PythonProject/dataset/Self-data/"
    "kFold/kFold(61x750)/new-segment/train/"
)
data_path2 = (
    "/home/guyue/zhao/PythonProject/dataset/Self-data/"
    "kFold/kFold(61x750)/new-segment/test/"
)
# 如果要换 Isfahan 数据，把上面两行注释掉，用下面两行
# data_path1 = '/home/guyue/zhao/PythonProject/dataset/Isfahan-data/kFold/0.5-50/2s/new-segment/train/'
# data_path2 = '/home/guyue/zhao/PythonProject/dataset/Isfahan-data/kFold/0.5-50/2s/new-segment/test/'

img_dir = (
    "all_result/work_I_result/test_result/self-data/segment/"
    "SAMFF_Net_4/BR-Split-7/img_1/"
)
log_dir = (
    "all_result/work_I_result/test_result/self-data/segment/"
    "SAMFF_Net_4/BR-Split-7/log_1/"
)
dir_model_path = (
    "all_result/work_I_result/save_model/Self/"
    "SAMFF_Net_4/BR-Split-7/1/"
)

for dir_ in [img_dir, log_dir, dir_model_path]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

train_files = os.listdir(data_path1)
test_files = os.listdir(data_path2)
train_files.sort(key=lambda x: int(x[5:-4]))  # 'train0.npy' 这种格式
test_files.sort(key=lambda x: int(x[4:-4]))   # 'test0.npy'

for i in range(10):
    train_path = os.path.join(data_path1, train_files[i])
    test_path = os.path.join(data_path2, test_files[i])
    model_name = "sub_{}.pth".format(i)
    model_path = os.path.join(dir_model_path, model_name)

    print(train_path)
    print(test_path)

    # DataLoader
    train_dataset = MyDataset(train_path)
    test_dataset = MyDataset(test_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    print("train_size: ", len(train_dataloader) * batch_size)
    print("test_size: ", len(test_dataloader) * batch_size)

    # 设置随机种子（多卡时也同步）
    torch.cuda.manual_seed_all(42)

    # 脑区划分
    region_indices = [
        [1, 60, 2, 50, 36, 37, 51, 11, 44, 3, 30, 17, 31, 4, 45, 12],
        [58, 52, 53, 59, 13, 14, 54, 55],
        [
            25,
            38,
            21,
            22,
            39,
            26,
            46,
            5,
            32,
            18,
            33,
            6,
            47,
            27,
            40,
            23,
            61,
            24,
            41,
            28,
        ],
        [15, 48, 7, 34, 19, 35, 8, 49, 16, 56, 42, 29, 43, 57],
        [9, 20, 10],
    ]

    # 初始化模型（这里用的是你新改好的 SNN+STAtten 版 ERTrans）
    model = ERTrans(
        samples=750,
        sa_emb_dim=128,
        d_ff=128,
        region_indices=region_indices,
        device=device,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 各种记录容器
    Train_Acc_list, Test_Acc_list = [], []
    Train_Loss_list, Test_Loss_list = [], []
    Train_Acc, Test_Acc = {}, {}
    Train_Loss, Test_Loss = {}, {}

    best_train_acc = 0.0
    best_test_acc = 0.0
    best_test_recall = 0.0
    best_test_specificity = 0.0
    best_test_precision = 0.0
    best_test_f1_score = 0.0
    best_test_auc = 0.0
    best_model = None

    # 日志和曲线保存路径
    img_name = str(i) + ".png"
    img_filepath = os.path.join(img_dir, img_name)
    log_name = str(i) + ".txt"
    log_filepath = os.path.join(log_dir, log_name)
    logging.FileHandler(log_filepath)

    print(
        "\nTime: {}, Batch_size: {}, Learning_rate: {}, Epochs: {}\n".format(
            time.strftime("%Y_%m_%d_%H:%M:%S"),
            batch_size,
            learning_rate,
            num_epochs,
        )
    )
    print(
        "==================================== Training start ===================================="
    )

    for epoch in range(num_epochs):
        train_loss, train_acc = train(
            model, train_dataloader, criterion, optimizer, device
        )
        (
            test_loss,
            test_acc,
            test_recall,
            test_specificity,
            test_precision,
            test_f1_score,
            test_auc,
        ) = test(model, test_dataloader, criterion)

        if train_acc > best_train_acc:
            best_train_acc = train_acc

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_recall = test_recall
            best_test_specificity = test_specificity
            best_test_precision = test_precision
            best_test_f1_score = test_f1_score
            best_test_auc = test_auc
            best_model = model.state_dict()

        Train_Acc_list.append(train_acc)
        Train_Loss_list.append(train_loss)
        Train_Acc["train"] = Train_Acc_list
        Train_Loss["train"] = Train_Loss_list

        Test_Acc_list.append(test_acc)
        Test_Loss_list.append(test_loss)
        Test_Acc["test"] = Test_Acc_list
        Test_Loss["test"] = Test_Loss_list

        print(
            "Epoch [{:03d}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, "
            "Test Loss: {:.4f}, Test Acc: {:.4f}".format(
                epoch + 1,
                num_epochs,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
            )
        )
        print(
            "\t\t\t\tTest Recall: {:.4f}, Test Precision: {:.4f}, "
            "Test F1-Score: {:.4f}".format(
                test_recall,
                test_precision,
                test_f1_score,
            )
        )

        # 写日志
        write_info_logs(
            log_filepath, epoch, train_loss, train_acc, test_loss, test_acc
        )
        write_target_logs(
            log_filepath,
            test_recall,
            test_specificity,
            test_precision,
            test_f1_score,
            test_auc,
        )

    print(
        "\nbest_train_acc: {:.4f}, best_test_acc: {:.4f}"
        "\nbest_test_recall: {:.4f}"
        "\nbest_test_specificity: {:.4f}"
        "\nbest_test_precision: {:.4f}"
        "\nbest_test_f1_score: {:.4f}"
        "\nbest_test_auc: {:.4f}".format(
            best_train_acc,
            best_test_acc,
            best_test_recall,
            best_test_specificity,
            best_test_precision,
            best_test_f1_score,
            best_test_auc,
        )
    )

    # 写入超参 & 最优结果
    write_params_logs(log_filepath, num_epochs, batch_size, learning_rate)
    write_best_logs(
        log_filepath,
        best_train_acc,
        best_test_acc,
        best_test_recall,
        best_test_specificity,
        best_test_precision,
        best_test_f1_score,
        best_test_auc,
    )
    print("{} SuccessFull......".format(log_name))

    # 如需保存最优模型，解开下面两行
    # torch.save(best_model, model_path)
    # print(model_name + " save successful!")

    # 保存训练/测试曲线
    plt = plt_img(Train_Acc, Test_Acc, Train_Loss, Test_Loss, show=False)
    plt.savefig(img_filepath, dpi=1200)
