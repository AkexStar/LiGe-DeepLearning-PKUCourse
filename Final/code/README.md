# 基于卷积条件神经过程的空间预测实验

## 代码文件结构

项目代码参考了GitHub上的开源Repo：[pytorch-convnp](https://github.com/makora9143/pytorch-convcnp)

- npf：神经过程的基础类
- utils：数据读取和图像显示类
- result：存放训练好的模型文件
- data：示例数据
- requirements.txt：项目所需要的各种python包
- figures：实验结果图像
- **myexperiment.ipynb：项目实验内容**

## 实验复现说明

- 先安装requirements中的各种环境
- 打开Jupyter notebook运行myexperiment.ipynb
  - 每个实验步骤均直接在其中说明，包括导入数据、查看数据、构建元数据集、根据制定参数构建模型、根据制定参数训练模型、可视化模型预测结果
