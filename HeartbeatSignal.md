# 心跳信号分类预测

## 数据预处理

* 有严重的数据不平衡问题
  * 采用了oversample
  * 有SMOTE与Mahakil两种方式可选，Mahakil方式较优但差别不大
* 任务简单、样本特征少
  * 简单模型效果较好
  * 易过拟合
  * 数据预处理很重要
  * 应用DFT引入频域信息
    * 有提升，但不明显

## 模型介绍

`full_connnect.py`：简单的全连接神经网络

`simple_CNN.py`：简单的CNN

> 444分

`top_Net1.py`：复现了冠军方案中的Net1部分

* 多尺寸感受域
* dropout
* 投票机制

> 312分

`complex_CNN.py`：深层CNN

* 采用了dropout、Batch Normal技术
* 拼接了3\*3和5\*5的感受域

> 230分

`Res_CNN.py`：ResNet

* 在`complex_CNN`的基础上，引入了残差块
