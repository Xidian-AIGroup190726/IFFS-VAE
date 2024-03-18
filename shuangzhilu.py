import numpy as np
import torch
import torch.nn as nn
from libtiff import TIFF
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import os
import torch.optim as optim
from resnet18 import Net
from sklearn.metrics import confusion_matrix
import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


EPOCH = 30     
BATCH_SIZE = 64    
LR = 0.001    
Train_Rate = 0.2     
os.environ['CUDA_VISIBLE_DEVICES'] = '1'        

cfg = {
        'Categories_Number': 12, # 类别数
    }

# 读取图片
ms4_tif = TIFF.open('./ms4.tif', mode='r')
ms4_np = ms4_tif.read_image()
print('原始ms4图的形状：', np.shape(ms4_np))

pan_tif = TIFF.open('./pan.tif', mode='r')
pan_np = pan_tif.read_image()
print('原始pan图的形状：', np.shape(pan_np))

label_np = np.load("./label6.npy")

# label_np = np.transpose(label_mat['label'])
# label_np = np.load('./image/label.npy')
print('label数组形状：', np.shape(label_np))

# ground_truth = cv2.imread('./data/groundtruth.bmp')

# ms4与pan图补零
Ms4_patch_size = 16          # ms4截块的边长
Interpolation = cv2.BORDER_REFLECT_101
# cv2.BORDER_REPLICATE ： 进行复制的补零操作；
# cv2.BORDER_REFLECT  ： 进行翻转的补零操作：gfedcba|abcdefgh|hgfedcb；
# cv2.BORDER_REFLECT_101  ： 进行翻转的补零操作：gfedcb|abcdefgh|gfedcb；
# cv2.BORDER_WRAP ： 进行上下边缘调换的外包复制操作： bcdegh|abcdefgh|abcdefg；

top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的ms4图的形状：', np.shape(ms4_np))

Pan_patch_size = Ms4_patch_size * 4         # pan裁块的边长
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的pan图的形状：', np.shape(pan_np))

# 按类别比例拆分数据集
# label_np = label_np.astype(np.uint8)
label_np = label_np - 1       
label_element, element_count = np.unique(label_np, return_counts=True)   # 返回类别标签与各个类别所占的数量
print('类标：', label_element)
print('各类样本数：', element_count)
Categories_Number = len(label_element) - 1      # 数据的类别数
print('标注的类别数：', Categories_Number)
label_row, label_column = np.shape(label_np)

'''归一化图片'''


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


ground_xy = np.array([[]] * Categories_Number).tolist()
ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)

count = 0
for row in range(label_row):            # 行
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np[row][column] != 255:
            ground_xy[int(label_np[row][column])].append([row, column])

# 标签内打乱
for categories in range(Categories_Number):
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
    np.random.shuffle(shuffle_array)

    ground_xy[categories] = ground_xy[categories][shuffle_array]
shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]

ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    # print('aaa', categories_number)
    for i in range(categories_number):
        if i < int(categories_number * Train_Rate):
            ground_xy_train.append(ground_xy[categories][i])
        else:
            ground_xy_test.append(ground_xy[categories][i])
    label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
    label_test = label_test + [categories for x in range(int(categories_number - int(categories_number * Train_Rate)))]

print('asasas')
label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_train = np.array(ground_xy_train)
ground_xy_test = np.array(ground_xy_test)

# 训练数据与测试数据，数据内打乱
shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]

label_train = torch.from_numpy(label_train).type(torch.LongTensor)

label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

print('训练样本数：', len(label_train))
print('测试样本数：', len(label_test))

# 数据归一化
ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)

pan = np.expand_dims(pan, axis=0)   # 二维数据进网络前要加一维
ms4 = np.array(ms4).transpose(2, 0, 1)       # 整理通道

# 转换类型
ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)


class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)            # 计算不可以在切片中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms : x_ms + self.cut_ms_size,
                   y_ms : y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan : x_pan + self.cut_pan_size,
                    y_pan : y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]
        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)


class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)             # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms : x_ms + self.cut_ms_size,
                   y_ms : y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan: x_pan + self.cut_pan_size,
                   y_pan: y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)
train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)
# print("上色数据", next(iter(all_data)))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_data_loader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# print(next(iter(all_data_loader)))




# 定义优化器

model = Net(cfg).to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.95)

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.995

# 定义训练方法
def train_model(model, train_loader, optimizer, epoch):
    model.train()
    correct = 0.0
    for step, (ms, pan, label, _) in enumerate(train_loader):
        ms, pan, label = ms.to(device), pan.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(ms, pan)
        pred_train = output.max(1, keepdim = True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
        loss = F.cross_entropy(output, label.long())
        # 定义反向传播
        loss.backward()
        # 定义优化
        optimizer.step()
        if step % 100 == 0:
            print("Train Epoch：{} \t Loss：{:.6f} \t step：{}".format(epoch, loss.item(), step))
    print("Train Accuracy：{:.6f}".format(correct * 100.0 / len(train_loader.dataset)))

# 定义测试方法
def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, data1, target, _ in test_loader:
            data, data1, target = data.to(device), data1.to(device), target.to(device)
            output = model(data, data1)
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).long()).sum().item()
        test_loss = test_loss / len(test_loader.dataset)
        print("test-average Loss：{:.4f}, Accuracy：{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))
# 调用训练和测试
for epoch in range(1, EPOCH + 1):
    train_model(model, train_loader, optimizer,epoch)
    adjust_learning_rate(optimizer, epoch)
    test_model(model, test_loader)


# 计算Kappa指标
def con_mat():
    cnn.to(device)
    l = 0
    y_pred = []
    for step, (ms, pan, label, _) in enumerate(test_loader):
        l += 1
        ms = ms.to(device)
        pan = pan.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = cnn(ms, pan)
        pred_y = output.max(1, keepdim=True)[1]
        if l == 1:
            y_pred = pred_y.cpu().numpy()
        else:
            y_pred = np.concatenate((y_pred, pred_y.cpu().numpy()), axis = 0)

    con_mat = confusion_matrix(y_true=label_test, y_pred=y_pred)
    print("混淆矩阵：", con_mat)

    # 计算性能参数
    all_acr = 0
    p = 0
    column = np.sum(con_mat, axis=0)     # 列求和
    line = np.sum(con_mat, axis=1)       # 行求和
    for i, clas in enumerate(con_mat):
        precise = clas[i]
        all_acr = precise + all_acr
        acr = precise / column[i]
        recall = precise / line[i]
        f1 = 2 * acr * recall / (acr + recall)
        temp = column[i] * line[i]
        p = p + temp
        # print('PRECISION：', acr, '||RECALL：', recall, '||F1：', f1)    # 查准率 # 查全率 # F1
        print("第 %d 类：|| 准确率： %.7f || 召回率： %.7f || F1： %.7f " % (i, acr, recall, f1))
    OA = np.trace(con_mat) / np.sum(con_mat)
    print('OA：', OA)

    AA = np.mean(con_mat.diagonal() / np.sum(con_mat, axis=1))   # axis=1每行求和
    print('AA：', AA)

    Pc = np.sum(np.sum(con_mat, axis=0) * np.sum(con_mat, axis=1)) / (np.sum(con_mat)) ** 2
    Kappa = (OA - Pc) / (1 - Pc)
    print('Kappa：', Kappa)






# clour_model(cnn, colour_loader)
# colour_end = time.time()
