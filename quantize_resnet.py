import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torchvision.transforms as transforms
import math
from modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import time
from torchsummary import summary
import matplotlib.pyplot as plt


import torch.nn as nn
import torchvision.transforms as transforms
import math

#参数设置
# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()


NUM_BITS = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD = 8


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)


def init_model(model):
    for m in model.modules():
        if isinstance(m, QConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = QConv2d(inplanes, planes, kernel_size=1, bias=False,
                             num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)
        self.bn1 = RangeBN(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, num_bits=NUM_BITS,
                             num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)
        self.bn2 = RangeBN(planes)
        self.conv3 = QConv2d(planes, planes * 4, kernel_size=1, bias=False,
                             num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)
        self.bn3 = RangeBN(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QConv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                        num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = QConv2d(1, 64, kernel_size=3, stride=1, padding=1,
                             bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = QLinear(512 * block.expansion, num_classes, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)

        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 1e-2},
            {'epoch': 60, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 90, 'lr': 1e-4}
        ]


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.conv1 = QConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                             bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = QLinear(64, num_classes, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)

        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 81, 'lr': 1e-2},
            {'epoch': 122, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 164, 'lr': 1e-4}
        ]


def resnet18_quantized():
            return ResNet_imagenet(num_classes=10,
                                   block=BasicBlock, layers=[2, 2, 2, 2])

#参数设置
# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多

# 超参数设置
EPOCH = 4   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.01        #学习率
Loss_list = [2.362] #绘制曲线
Accuracy_list = [6.250]
# #定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
# "cuda" if torch.cuda.is_available() else
#
# #准备数据集
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
#     transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# trainset = torchvision.datasets.CIFAR10(root='C:\data', train=True, download=True, transform=transform_train) #训练数据集
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
#
# testset = torchvision.datasets.CIFAR10(root='C:\data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# # Cifar-10的标签
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# 数据集的下载器
# train_dataset = datasets.MNIST(
#     root='./data', train=True, transform=data_tf, download=True)
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,  # this is training data
    transform=data_tf,  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=True,
)


# #plot one example
loaded = train_data.train_data.numpy() #将数据由张量变为数组
trX = loaded.reshape((60000*28*28, 1))
trX = np.array(trX)
# 将数据比特化
Xp = np.unpackbits(trX, axis=1)     # 进行二进制转换
# ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××
# 生成系数矩阵
beta = np.zeros((60000*784, 8), dtype=np.uint8)
beta = np.array(beta)

for i in range(0, 8):
    beta[:, i] = 2**(8-i-1)
# ××××××××××××××××××××××××××××××××××××××××××××××××××××××××
# 系数矩阵与二值化矩阵点乘
Xp_beta = np.multiply(Xp, beta)
alpha = np.load('/home/alexrich/MNIST/coef7.npy')
trX_recov = np.dot(Xp_beta, alpha)
trX_recov = trX_recov.reshape(60000, 28, 28)
trX_recov = trX_recov.astype(np.uint8)
trX_recov = torch.from_numpy(trX_recov)
train_data.train_data = trX_recov


test_data = datasets.MNIST(root='./mnist', train=False, transform=data_tf)
#plot one example
loaded = test_data.test_data.numpy() #将数据由张量变为数组
trX = loaded.reshape((10000*28*28, 1))
trX = np.array(trX)
# 将数据比特化
Xp = np.unpackbits(trX, axis=1)     # 进行二进制转换
# ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××
# 生成系数矩阵
beta = np.zeros((10000*784, 8), dtype=np.uint8)
beta = np.array(beta)

for i in range(0, 8):
    beta[:, i] = 2**(8-i-1)
# ××××××××××××××××××××××××××××××××××××××××××××××××××××××××
# 系数矩阵与二值化矩阵点乘
Xp_beta = np.multiply(Xp, beta)
alpha = np.load('/home/alexrich/MNIST/coef7.npy')
trX_recov = np.dot(Xp_beta, alpha)
trX_recov = trX_recov.reshape(10000, 28, 28)
trX_recov = trX_recov.astype(np.uint8)
trX_recov = torch.from_numpy(trX_recov)
test_data.test_data = trX_recov
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

#模型定义
net = resnet18_quantized().to(device)
# summary(net,(1,28,28))

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)#优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

#训练
if __name__ == "__main__":
    best_acc = 0  # 初始化best test accuracyplanes
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                '''
                asas=1
                if epoch % asas ==0:

                   fe='/home/cc/Desktop/dj/model1/model/incption--{}'.format(epoch)
                   torch.save(net.state_dict(),fe)
                '''
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(train_loader, 0): #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，
                    # 准备数据
                    length = len(train_loader)
                    # if (i+1+epoch * length)>=2391:
                    #     LR=LR*0.1
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct += predicted.eq(labels.data).cpu().sum()
                    total += labels.size(0)
                    correct = float(correct)
                    acc = 100.*(correct/total)
                    if (i+1+epoch * length)%150 == 0:
                        Loss_list.append(sum_loss/(i+1))
                        Accuracy_list.append(acc)
                    '''
                    qw=100. * correct / total
                    ee=qw.cpu().numpy()
                    train_a.writelines('epoch:'+str(epoch + 1)+'     '+'iter:'+str(i + 1 + epoch * length)+'   '+'Loss:'+str(sum_loss / (i + 1))+'   '+'acc:'+str(ee)+'%'+'\n')
                    '''
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), acc))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), acc))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    start = time.time()
                    for data in test_loader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).cpu().sum()
                        correct = float(correct)
                    end = time.time()
                    duration = end - start
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    print('推断时间：%.3f' % (duration))
                    acc = 100. * (correct / total)

                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            np.save("./RES7.npy", Accuracy_list)
            x1 = range(1,3752,150)
            x2 = range(1,3752,150)
            y1 = Accuracy_list
            y2 = Loss_list
            plt.subplot(1,2,1)
            plt.plot(x1, y1,'o-',markersize =7)
            plt.title('Acc_train vs. Iteration',fontsize=15)
            plt.ylabel('Acc_train', fontsize=15)
            plt.xlabel('Iteration', fontsize=15)
            plt.yticks(size=14)
            plt.xticks(size=14)
            plt.subplot(1, 2, 2)
            plt.plot(x2, y2, 'o-',markersize =7)
            plt.title('Loss vs. Iteration',fontsize=15)
            plt.ylabel('Loss', fontsize=15)
            plt.xlabel('Iteration', fontsize=15)
            plt.yticks(size=14)
            plt.xticks(size=14)
            plt.show()
            plt.savefig("accuracy_loss.jpg")

            print("Training Finished, TotalEPOCH=%d" % EPOCH)
#
