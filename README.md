# LiGe-DeepLearning-PKUCourse
Jupyter notebook 深度学习技术与应用（2023春-李戈老师课程）

# 作业一：多层神经网络的训练

## 作业要求

**请根据自己的计算环境情况和兴趣，选择以下两个数据集之一，完成如下的实验：**

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)

**1. 构造一个多层的神经网络（注意，不要使用卷积神经网络，本题目要求使用多层神经网络），并在上述数据集任务上进行训练，并汇报一个“使用了你认为最优的超参数配置的神经网络”的学习曲线；要求如下：**

（1）自己手动完成反向传播算法部分的编写；
（2）该网络应为一个“纯净”的多层神经网络，不使用正则化方法、率优化算法等；

**2. 在上述“你认为最优配置的神经网络”的基础上，**

（1）分别汇报“增加一个隐藏层”和“减小一个隐藏层”情况下的学习曲线；
（2）分别汇报使用BGD和SGD进行训练的学习曲线；
（3）分别汇报使用两种以上参数初始化方法下的学习曲线；
（4）分别汇报使用两种以上学习率优化算法下的学习曲线；
（5）分别汇报使用两种以上正则化方法下的学习曲线；

**最终提交：包含6个子文件夹的一个zip文件，其中的子文件夹应包含：**

（1）对应上述6种情况之一的一份源代码；
（2）对应上述源代码的学习曲线的一个.png文件；

# 作业二：图像分类模型的对抗攻击和对抗训练

- 作业要求：见[homework-2](./homework-2/homework-2.pdf)文件的说明，对应GitHub连接为[ADVERSARIAL ATTACK & ADVERSARIAL TRAINING](https://github.com/LC-John/Fashion-MNIST)
- 项目内容简介：训练[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)数据集上的分类模型；进行定向白盒攻击（I-FGSM）；进行定向黑盒攻击（样本迁移、MCMC）；简单对抗训练
- 若样本的真实标签为 label ，攻击方向为使分类器错判为 (label + 1) % 10

# 作业三：Street View House Number Recognition 街景字符编码识别

**数据集：**
* [SVHN: Street View House Number](http://ufldl.stanford.edu/housenumbers/)，来源于谷歌街景门牌号码
* 本次作业的目标数据集是其中的 Format 1 (Full Numbers: train.tar.gz, test.tar.gz , extra.tar.gz). 其中，train.tar.gz 为训练数据集，test.tar.gz为测试数据集。注：extra.tar.gz是附加数据集，建议不使用。
* 在train.tar.gz与test.tar.gz中，分别包含：
    （1）一些.png格式的图片，每张图片包含一个门牌号；
    （2）一个digitStruct.mat文件，包含每张图片所对应的门牌号，以及每个门牌号数字的位置信息；
    （3）一个see_bboxes.m文件，用于辅助Matlab环境下的处理，请忽略之。

**要求：**
1. 设计一个网络，用train.tar.gz中的数据进行训练，并用test.tar.gz中的数据进行测试；
2. 在测试的过程中，不允许使用test.tar.gz/digitStruct.mat文件中的位置信息作为输入，即必须在“忽略测试数据集中给出的位置信息”的前提下，识别出test.tar.gz中每张图片中的门牌号；
3. 撰写一个PPT，汇报如下信息：
    （1）所设计的网络的结构和超参数信息；
    （2）网络的训练方法和优化方法；
    （3）体现训练过程的“训练曲线”；
    （4）识别准确率；

# 作业四：Code Generation with Deep Learning 代码生成

Code Generation是一个以自然语言为输入，输出一个代码片段的任务。要求该输出的代码片段能够完成自然语言输入所描述的编程任务。在通常情况下，自然语言输入的长度单位是一个语句，而相应的程序输出可以是一行代码、多行代码或一个完整的方法体。

CONCODE是一个较为经典的Code Generation任务的数据集。

本次作业的要求是：以CONCODE数据集为训练集和测试集，完成一个支持程序代码生成的深度神经网络。

**一、任务数据集：**

本次作业的数据集选用CodeXGlue数据集中与代码生成相关的子数据集CONCODE，数据相关的格式、基本状况可以参考如下的链接：

https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code

**二、结果汇报**

请提供你的【程序源代码】及【模型训练介绍PPT】，其中PPT应包含以下内容：

（1）请提供你所采用的模型结构的图示及相关说明；

（2）请提供你的模型在验证数据集和测试数据集上的结果，衡量指标采用：Exact Match 和 BLEU

（3）请提供能够体现你的训练过程的Learn Curve及相关说明。

