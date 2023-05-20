# Street View House Number Recognition 街景字符编码识别

## 作业说明

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

## 解题思路

**致谢**：
- 本部分参考了此文献的内容[《零基础入门CV赛事—街景字符编码识别—task1赛题理解》](https://blog.csdn.net/Libaididi/article/details/106185983)
- 阿里云天池学习上有相应的赛题：[零基础入门CV - 街景字符编码识别](https://tianchi.aliyun.com/competition/entrance/531795/information)

### 定长字符识别
- 将赛题抽象为一个定长字符识别问题，在赛题数据集中大部分图像中字符个数为2-4个，最多的字符 个数为6个。
- 因此可以对于所有的图像都抽象为6个字符的识别问题，字符23填充为23XXXX，字符231填充为231XXX。
- 经过填充之后，原始赛题变为6个字符的分类问题。在每个字符的分类中会进行11个类别的分类，假如分类为填充字符，则表明该字符为空。
- 可以参考Google的一篇文献：[Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://arxiv.org/abs/1312.6082)，GitHub上有已经实现的Pytorch框架下的代码[SVHNClassifier-PyTorch](https://github.com/potterhsu/SVHNClassifier-PyTorch)

### 不定长字符识别
- 在字符识别研究中，有特定的方法来解决此种不定长的字符识别问题，比较典型的有CRNN字符识别模型。
- CRNN的结构是CNN-RNN-CTC，CNN提取特征，RNN预测出不太好的序列，CTC把不太准确的序列翻译成正确的词。

### 检测再识别
- 因为数据集里给出了每个数字的位置框，可以选择目标检测的方法，可以采用Faster R-CNN、SSD、YOLO等方法，先检测出图像中的字符框，再对字符框进行识别。
### 可以参考的资料和代码Repo：
- [GitHub：SVHNClassifier-PyTorch](https://github.com/potterhsu/SVHNClassifier-PyTorch) 定长字符识别
- [简书：零基础入门CV赛事- 街景字符编码识别](https://www.jianshu.com/p/60d3cee6ccbf) 定长字符识别
- [知乎：真*零基础入门CV--街景字符识别（阿里天池学习赛）](https://zhuanlan.zhihu.com/p/359572604) 定长字符识别
- [天池：Datawhale 零基础入门CV赛事-Baseline](https://tianchi.aliyun.com/notebook/108342) 定长字符识别
- [天池：零基础基于yolov5进行街景字符编码识别](https://tianchi.aliyun.com/forum/post/328865) YOLO
- [CSDN：阿里天池街景字符编码YOLO5方案](https://blog.csdn.net/qq_44694861/article/details/124523492) YOLO
- [CSDN：用YOLOV4实现街景字符编码识别](https://blog.csdn.net/m0_46478164/article/details/106305143) YOLO
- [GitHub：Street-View-House-Numbers-Detection](https://github.com/chia56028/Street-View-House-Numbers-Detection) YOLO

## 本项目实现
- 具体内容见[PPT](./train-yolov5s.pptx)文件中的展示说明

### 数据集情况

- 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
- Format-1: 
    - 33402 images for Train and 13068 images for Test. 
    - Each image has several digits('label', 'left', 'top', 'width', 'height'). All files are png.
<div align=center>
    <img src='https://github.com/AkexStar/LiGe-DeepLearning-PKUCourse/assets/55226358/fcc444e8-df8f-4c22-ac6b-75a09cfe5b0e' width='50%'>
</div>

- 类别上标签’9’最少，’1’最多，标签类别分布不均匀
- 图像上的数字目标框位置在纵轴上较为居中，在横轴上在中间两侧分布
- 数字目标框基本为窄矩形，符合数字字体的特点
- 数字目标框的长宽比近似于4.5：1

<div align=center>
    <img src='https://github.com/AkexStar/LiGe-DeepLearning-PKUCourse/assets/55226358/7fe0d98a-7cd1-4756-a517-bd14457f4027' width='40%'>   
    <img src='https://github.com/AkexStar/LiGe-DeepLearning-PKUCourse/assets/55226358/fde5cd01-0597-41a7-a2a8-4a99b656532e' width='40%'>
</div>

