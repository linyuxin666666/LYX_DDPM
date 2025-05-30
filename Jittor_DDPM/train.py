import jittor as jt
from jittor.dataset.mnist import MNIST
from jittor import transform
from dataset import train_loader, model, optimizer, gaussian_diffusion, timesteps
import time 
import matplotlib.pyplot as plt 



# train
epochs = 10
batch_size=64
for epoch in range(epochs):
    epoch_start_time = time.time() # 获取epoch开始时间
    for step, (images, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()

        batch_size = images.shape[0]
      
        # sample t uniformally for every example in the batch
        t =jt.randint(0, timesteps, (batch_size,)).long()

        loss = gaussian_diffusion.train_losses(model, images, t)


        if step % 200 == 0:
            print("Epoch:", epoch, "Loss:", loss.item())

        optimizer.step(loss)  # 梯度清零、反向传播、参数更新一步完成

    # --- 每个epoch的统计数据记录结束 ---
# 保存训练好的模型
model_save_path = "ddpm_jittor_mnist.ckpt"
jt.save(model.state_dict(), model_save_path)
print(f"模型已成功保存到: {model_save_path}")
