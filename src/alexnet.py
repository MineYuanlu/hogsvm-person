#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: alexnet.py
# Author: 王梓瑄
# Date: 2025/10/28

# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# pip3 install Pillow

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models.alexnet import AlexNet
from torchvision.models import alexnet, AlexNet_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import time
import os
import tqdm
import PIL.Image as Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择GPU或CPU
DATA_DIR = "./assets"
MODEL_DIR = os.path.join(DATA_DIR, "alex-models")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


def model_file(index: int | None = None):
    """
    获取模型文件路径

    :param index: 模型编号(None代表默认模型)
    """
    if index is None:
        return os.path.join(MODEL_DIR, "alexnet.pth")
    else:
        return os.path.join(MODEL_DIR, f"alexnet-{index}.pth")


def make_transform(random: bool):
    """
    构造数据集的预处理函数

    :param random: 是否随机翻转图片
    :return: 数据集的预处理函数
    """
    funcs = [
        transforms.Resize(224),  # AlexNet 需要224x224
    ]
    if random:
        funcs.append(transforms.RandomHorizontalFlip())
    funcs.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(funcs)


def make_dataset(
    train=True,
    dataset_dir=DATA_DIR,
):
    """
    构造数据集

    :param train: 是否为训练集
    :param dataset_dir: 数据集目录
    :return: 数据集目录, 数据集加载器
    """
    transform = make_transform(random=train)
    dataset = CIFAR10(root=dataset_dir, train=train, download=True, transform=transform)
    return dataset


def make_dataloader(
    train: bool,
    batch_size=128,
    shuffle=True,
    dataset_dir=DATA_DIR,
):
    """
    构造数据集加载器

    :param train: 是否为训练集
    :param batch_size: 批大小
    :param shuffle: 是否打乱数据集
    :param dataset_dir: 数据集目录
    :return: 数据集目录, 数据集加载器
    """
    dataset = make_dataset(train, dataset_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,  # 增加数据加载进程数
        pin_memory=True,  # 锁页内存，加速GPU传输
        persistent_workers=True,  # 保持worker进程
        prefetch_factor=2,  # 预取批次
    )
    return loader


def make_model(classes: int):
    """
    构造AlexNet模型

    :param classes: 类别数
    """

    model = alexnet(weights=AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, classes)
    model = model.to(device)
    return model


def train_model(
    model: AlexNet,
    trainloader: DataLoader,
    label_smoothing=0.1,
    lr=1e-4,
    weight_decay=1e-4,
    epochs=20,
):
    """
    训练模型

    :param model: 待训练的模型
    :param trainloader: 训练集加载器
    :param label_smoothing: 标签平滑 (0.1: 减轻过拟合，提高泛化能力)
    :param lr: 学习率
    :param weight_decay: 权重衰减(5e-4: CIFAR任务常用经验值)
    :param epochs: 训练轮数
    """

    # criterion: 损失函数, 用于计算模型输出和标签之间的误差
    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,  # 标签平滑有助于减轻过拟合
    )
    # optimizer: 优化器, 用于更新模型参数以最小化损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,  # 学习率控制每次参数更新的步长
        weight_decay=weight_decay,  # 权重衰减有助于防止过拟合
    )
    # scheduler: 学习率调度器, 用于控制学习率随着训练过程的变化
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,  # 每10个epoch衰减一次
        gamma=0.5,  # 学习率乘以0.5
    )

    time_start = time.time()

    for epoch in range(epochs):  # 每轮迭代训练
        time_epoch_start = time.time()
        model.train()  # 开启训练模式
        running_loss = 0.0
        for inputs, labels in tqdm.tqdm(
            trainloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"
        ):  # 每个批次迭代训练
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()  # 累计损失

        scheduler.step()  # 更新学习率
        lr_now = optimizer.param_groups[0]["lr"]

        time_epoch_end = time.time()
        print(
            f"[{epoch+1}/{epochs}]",
            f"Loss: {running_loss / len(trainloader):.4f}",
            f"LR: {lr_now:.6f}",
            f"Epoch time: {time_epoch_end - time_epoch_start:.2f}s",
            f"Total time: {time_epoch_end - time_start:.2f}s",
        )
        save_model(model, epoch)  # 保存阶段性模型
    time_end = time.time()
    print(f"Training finished. Total time: {time_end - time_start:.2f}s")


def save_model(model: AlexNet, index: int | None = None):
    """
    保存模型

    :param model: 待保存的模型
    :param index: 模型编号
    """
    torch.save(model.state_dict(), model_file(index))


def load_model(model: AlexNet, index: int | None = None):
    """
    加载模型

    :param model: 待加载的模型
    :param index: 模型编号
    """
    model.load_state_dict(torch.load(model_file(index)))


def test_model(model: AlexNet, testloader: DataLoader, name="Model"):
    """
    测试模型

    :param model: 待测试的模型
    :param testloader: 测试集加载器
    :param name: 模型名称
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"{name} Accuracy on test set: {100 * correct / total:.2f}%")


def predict(model: AlexNet, classes: list[str], img_path: str):
    """
    使用模型预测图片类别

    :param model: 待预测的模型
    :param classes: 类别列表
    :param img_path: 图片路径
    """

    transform = make_transform(random=False)

    img = Image.open(img_path)
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        _, pred = torch.max(outputs, 1)
    print(
        f"Predicted class: {classes[pred.item()]}",
        f"Image: {img_path}",
    )


def predicts(model: AlexNet, classes: list[str], img_dir: str = IMAGE_DIR):
    """
    使用模型预测图片类别

    对一个文件夹所有图片进行预测
    """
    for file in os.listdir(img_dir):
        file = os.path.join(img_dir, file)
        predict(model, file, classes)


def save_images(
    classes: list[str],
    dataset_dir=DATA_DIR,
    image_dir=IMAGE_DIR,
    label_cnt=2,
):
    """
    从数据集中选取指定数量的图片保存到指定目录

    :param dataset_dir: 数据集目录
    :param image_dir: 图片保存目录
    :param label_cnt: 每个类别保存数量
    """
    dataset = CIFAR10(root=dataset_dir, train=False)
    labels: dict[str, int] = {}

    for i, (img, label) in enumerate(dataset):
        cnt = labels.get(classes[label], 0)
        if cnt >= label_cnt:
            continue
        img_path = os.path.join(image_dir, f"{classes[label]}-{cnt}.jpg")
        img.save(img_path)
        labels[classes[label]] = cnt + 1


def main():
    """
    主函数, 通过修改代码开关, 完成`训练`/`测试`/`预测`流程
    """
    classes: list[str] = make_dataset().classes  # CIFA-10的类别
    print("CIFA-10 classes:", classes)
    print("Device:", device)

    epochs = 20  # 训练轮数

    # 构造模型
    model = make_model(len(classes))

    # 训练模型
    def train_func(load=False):
        if load:
            load_model(model)  # 加载已有模型
        train_model(model, make_dataloader(True), epochs=epochs)
        save_model(model)

    # train_func()

    # 测试模型
    def test_func(test_final: bool, test_all: bool):
        testloader = make_dataloader(False)
        tests: list[int | None] = []  # 待测试的模型
        if test_final:
            tests.append(None)
        if test_all:
            tests.extend(reversed(range(epochs)))
        for i in tests:
            try:
                load_model(model, i)  # 加载模型
            except FileNotFoundError:
                continue
            test_model(model, testloader, name=f"Model-{i}")  # 测试模型

    # test_func(True,True)

    # 预测图片类别
    def predict_func():
        # 从测试集选取一张图片保存为文件
        save_images(classes)

        # 预测图片类别
        load_model(model)  # 加载默认模型
        predicts(model, classes)

    predict_func()


if __name__ == "__main__":
    main()
