# DDPM论文用jittor框架的复现（用MNIST训练）
---

## 一、创建环境
运行以下命令：
```bash
conda create -n ddpm_jt python=3.8
conda activate ddpm_jt
pip install -r requirements.txt
cd Jittor_DDPM
```
即可创建成功该项目的可运行环境，下面可在该环境下进行采样和训练。

---


## 二、运行项目

在ddpm_jt环境下，输入如下命令进行训练和采样

### 训练

本项目采用MNIST数据集进行训练，并保存训练好的模型权重,运行以下命令可实现训练并保存模型权重。
```bash
python train.py
```
### 生成

采用上述训练的模型权重进行图像生成，可生成8*8图像网格以及16*16渐进式采样。
```bash
python sample.py
```
---
## 三、曲线对齐

### 3.1 loss曲线

  训练Jittor版本和Pytorch版本的DDPM过程中，本项目记录了二者训练过程每个step的loss曲线，设置batch_size=64,训练10个epoch，每个epoch训练938个step，训练曲线如下：
  
#### 整体loss曲线

![step_loss_jittor](Curve/jittor_loss_curve_all_steps.png)

![step_loss_torch](Curve/torch_loss_curve_all_steps.png)

#### 每200个step取一个点的loss曲线
![step_loss_jittor](Curve/torch_loss_curve_sampled_from_200.png)

![step_loss_torch](Curve/jittor_loss_curve_sampled_from_200.png)

观察loss曲线，我们可以得出如下结论：
- 二者的收敛值较为相近。
- 需要让二者收敛的step值也较为相近。

**详细记录的训练log如下：**

- [jittor_loss_step.txt](Jittor_DDPM/training_logs/loss_step.txt)  
- [torch_loss_step.txt](Pytorch_DDPM/DiffusionModels/training_logs_pytorch/loss_step.txt)

---
### 3.2 训练时间对齐曲线

本项目对pytorch和jittor的实现版本的每个epoch的训练时间进行记录，对比图如下：

![epoch_training_time_comparison.png)](Curve/epoch_training_time_comparison.png)

可以看到：
- Pytorch版本的训练时间明显低于Jittor版本，每个epoch大约少25秒左右的时间。
- Jittor版本的训练时间相较于Pytorch明显更短.

**详细记录的训练时间log如下：**

- [jittor_train_time.txt](Jittor_DDPM/training_logs/train_time.txt)
- [torch_train_time.txt](Jittor_DDPM/training_logs/train_time.txt)  
### 3.3 采样时间对比曲线

最后，本项目对pytorch和jittor的实现版本的每个epoch的训练时间进行记录，对比图如下：

![epoch_training_time_comparison.png)](Curve/sampling_time_conmaprison.png)

在这里发现了一个比较反常的现象：
- Pytorch版本的训练时间明显低于Jittor版本，每个epoch大约少25秒左右的时间。
- Jittor版本的训练时间相较于Pytorch明显更短.







