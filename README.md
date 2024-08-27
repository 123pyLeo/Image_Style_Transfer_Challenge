# 模型说明文档

本项目基于finlay-liu的大模型图像风格迁移挑战赛baseline，采用U-Net模型和ResNet18编码器，结合ImageNet预训练权重，通过SSIM和MSE Loss的联合损失函数进行优化。训练分为两个阶段：初始训练使用预热学习率，再训练冻结编码器并降低学习率。该模型在排行榜上获得第三名的成绩。

# 解决方案及算法介绍

## 文件结构

- project/
  - `README.md`            项目说明文件
  - `requirements.txt`     Python依赖列表
  - `prediction_result/`   存放模型预测结果的文件夹
  - `xfdata/`              包含项目使用的所有原始数据集
  - `code/`                源代码文件夹
    - `train.py`           模型训练代码
    - `test.py`            测试代码
    - `train.sh`           模型训练脚本
    - `test.sh`            测试脚本
  - `user_data/`           用户数据文件夹
    - `model_data/`        存放训练好的模型权重
      - `new_model_weights.pth`    训练好的最终权重
    - `tmp_data/`          存放临时代码生成的权重文件夹

### 注意：

#### 关于文件夹

注意： 除了code文件夹之外，其他文件夹在初始状态下为空。您需要根据项目的进展自行创建和填充这些文件夹

#### 关于数据

要获取本次大赛的数据集，请按照以下步骤操作：

1.访问以下网址：https://challenge.xfyun.cn/topic/info?type=image-style-transfer&option=ssgy

2.点击报名比赛，报名成功后，您将能够在赛题数据那下载所需的数据集

#### 关于权重

作者提供了使用GPU 3080训练得到的权重文件，您可以通过以下步骤将其应用到您的项目中：

1.下载作者提供的权重文件，并将其放置在项目的model_data文件夹下

链接：https://pan.baidu.com/s/1vsFvEX4sQzxZ66x1AAKQKQ 
提取码：1234

2.为了在测试中能直接使用权重，您需要修改test.py文件。找到第64行，将以下代码：

```bash
model.load_state_dict(torch.load('../user_data/tmp_data/new_model_weights.pth'))
```

将文件路径改成../user_data/model_data/new_model_weights.pth


## 环境设置

请确保您的Python环境是版本3.10。为了顺利进行开发，您需要安装以下包和库：

- opencv-python==4.10.0.82
- numpy==1.26.4
- matplotlib==3.8.1
- torch==2.0.1
- torchvision==0.15.2
- albumentations==1.4.13
- segmentation_models_pytorch==0.3.3
- six==1.16.0

安装依赖：

```bash
pip install -r requirements.txt
```

请确保在使用之前安装所有必要的包。


## 使用说明

### 训练模型

1. 首先进入该项目文件夹：

    ```bash
    cd project
    ```

2. 确保脚本具有可执行权限：

    在终端中，导航到包含 `train.sh` 脚本的目录，然后运行以下命令来赋予脚本可执行权限：
    ```bash
    chmod +x train.sh
    ```

3. 运行脚本：

    现在可以直接运行脚本。在终端中，导航到包含 `train.sh` 脚本的目录，然后运行以下命令：
    ```bash
    ./train.sh
    ```
    最后生成的权重文件将保存到tmp_data文件夹下

#### 注意：
在 Linux 环境下， .sh 脚本文件可以直接通过命令行运行。

如果您使用的是 Windows 系统，终端无法直接执行 .sh 文件。您可以通过以下两种方式之一来运行：

1.使用 GIT 工具来执行 .sh脚本。
2.直接在终端输入以下命令来启动Python脚本：
  ```bash
    python train.py
  ```

### 预测文件

预测文件和训练模型同理，将train改成test即可


## 训练细节

本项目代码参考了finlay-liu开源的大模型图像风格迁移挑战赛baseline，同时在此基础上做出了进一步调整和改进。

### 数据增强

使用Albumentations库进行数据增强。由于在训练的过程中发现加入翻转旋转等数据增强方法分数有一定程度下滑，最后选择的增强操作为图像大小调整为512x512像素。

### 模型

使用了Segmentation Models PyTorch库中的U-Net模型，选择ResNet18作为编码器，并加载了ImageNet的预训练权重。

### 损失函数

#### 结构相似性损失（SSIM Loss）

SSIM是一种衡量两幅图像结构相似性的指标。具体实现步骤如下：

1. 定义高斯窗口函数，用于计算图像的局部统计特性。
2. 计算图像块的均值、方差和协方差。
3. 结合均值、方差和协方差计算SSIM值。

#### 联合损失函数

我们设计了一个联合损失函数，结合了均方误差损失和结构相似性损失。联合损失函数的表达式为：

$$
\text{CombinedLoss} = \alpha \cdot \text{MSELoss} + (1 - \alpha) \cdot \text{SSIMLoss}
$$

其中，$\alpha$ 是一个权重，用于平衡两种损失的贡献。

### 训练过程

训练过程分为两个阶段：

#### 初始训练

1. 初始化模型并加载预训练权重。
2. 使用Adam优化器，并设置初始学习率为1e-3。
3. 采用联合损失函数进行训练。
4. 训练过程中，前5个epoch进行学习率预热（warmup），逐步增加学习率到初始值。
5. 训练13个epoch，并保存模型权重。

#### 再训练

1. 加载之前保存的模型权重。
2. 冻结编码器部分的权重，重新设置优化器，学习率降低为1e-4。
3. 重新训练10个epoch。


## 致谢

再次感谢以下开源项目提供的baseline，它们为本项目的研究和开发提供了重要的参考和基础：

- [Baseline 大模型图像风格迁移挑战赛](https://github.com/datawhalechina/competition-baseline/blob/master/competition/%E7%A7%91%E5%A4%A7%E8%AE%AF%E9%A3%9EAI%E5%BC%80%E5%8F%91%E8%80%85%E5%A4%A7%E8%B5%9B2024/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%BE%E5%83%8F%E9%A3%8E%E6%A0%BC%E8%BF%81%E7%A7%BB%E6%8C%91%E6%88%98%E8%B5%9B_label.ipynb)

我希望我们的工作也能为开源社区做出贡献，并激励更多的研究和发展。

