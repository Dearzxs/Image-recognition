import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Utils import *

batch_size = 100
path = './'

# MNIST dataset: root表示选择数据的根目录; train=True表示训练集; transform=None表示不考虑使用任何数据预处理; download=True表示从网络上download图片,
# 若已下载则不会下载

train_datasets = datasets.MNIST(root=path, train=True, transform=None, download=True)
test_datasets = datasets.MNIST(root=path, train=False, transform=None,  download=True)

# 加载数据 shuffle=True表示将数据打乱
train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

# 对训练数据处理
x_train = train_loader.dataset.data.numpy()
# 归一化处理
mean_image = getXmean(x_train)
x_train = centralized(x_train, mean_image)
y_train = train_loader.dataset.targets.numpy()
# 对测试数据处理，取前num_test个测试数据
num_test = 200
x_test = test_loader.dataset.data[:num_test].numpy()
mean_image = getXmean(x_test)
x_test = centralized(x_test, mean_image)
y_test = test_loader.dataset.targets[:num_test].numpy()

print("train_data:", x_train.shape)
print("train_label:", len(y_train))
print("test_data:", x_test.shape)
print("test_labels:", len(y_test))

classifier = Knn()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(5, 'E', x_test)
num_correct = np.sum(y_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# 利用KNN计算识别率
# for k in range(1, 6, 2):  # 不同K值计算识别率
#     classifier = Knn()
#     classifier.fit(x_train, y_train)
#     y_pred = classifier.predict(5, 'E', x_test)
#     num_correct = np.sum(y_pred == y_test)
#     accuracy = float(num_correct) / num_test
#     print('Got %d / %d correct when k= %d => accuracy: %f' % (num_correct, num_test, k, accuracy))
