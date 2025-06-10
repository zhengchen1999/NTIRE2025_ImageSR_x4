"""
融合网络改进版本，加入验证集和PSNR指标
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import math

from .team27_DAT import DAT
from .team27_RFDN import RFDN
from .team27_SwinIR import SwinIR

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, crop_size=128, scale=4, is_train=True):
        """
        Args:
            lr_dir (str): 低分辨率图像目录
            hr_dir (str): 高分辨率图像目录
            crop_size (int): 训练时随机裁剪的大小
            scale (int): 超分辨率的放大倍数
            is_train (bool): 是否为训练模式
        """
        super(DIV2KDataset, self).__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.crop_size = crop_size
        self.scale = scale
        self.is_train = is_train
        
        # 获取所有图像文件名
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))

        # 确保LR和HR图像对数量匹配
        assert len(self.lr_images) == len(self.hr_images), "LR和HR图像数量不匹配"
        
        # 基础变换
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        # 读取图像
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        if self.is_train:
            # 验证图像尺寸比例
            lr_w, lr_h = lr_img.size
            hr_w, hr_h = hr_img.size
            
            assert hr_w == lr_w * self.scale and hr_h == lr_h * self.scale, \
                f"图像{idx}的LR和HR尺寸不符合scale比例"
            
            # 确保有足够的裁剪空间
            if lr_w < self.crop_size or lr_h < self.crop_size:
                # 如果图像太小，将其放大到可裁剪大小
                scale_factor = max(self.crop_size / lr_w, self.crop_size / lr_h)
                new_lr_w = int(lr_w * scale_factor)
                new_lr_h = int(lr_h * scale_factor)
                lr_img = lr_img.resize((new_lr_w, new_lr_h), Image.BICUBIC)
                hr_img = hr_img.resize((new_lr_w * self.scale, new_lr_h * self.scale), Image.BICUBIC)
                lr_w, lr_h = lr_img.size
            
            # 安全的随机裁剪
            max_lr_x = max(0, lr_w - self.crop_size)
            max_lr_y = max(0, lr_h - self.crop_size)
            x = random.randint(0, max_lr_x)
            y = random.randint(0, max_lr_y)
            
            # 执行裁剪
            lr_crop = lr_img.crop((x, y, x + self.crop_size, y + self.crop_size))
            
            # 对HR图像进行对应位置的裁剪
            hr_x = x * self.scale
            hr_y = y * self.scale
            hr_crop_size = self.crop_size * self.scale
            hr_crop = hr_img.crop((hr_x, hr_y, hr_x + hr_crop_size, hr_y + hr_crop_size))
            
            # 数据增强
            if random.random() < 0.5:
                lr_crop = lr_crop.transpose(Image.FLIP_LEFT_RIGHT)
                hr_crop = hr_crop.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                lr_crop = lr_crop.transpose(Image.FLIP_TOP_BOTTOM)
                hr_crop = hr_crop.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                lr_crop = lr_crop.rotate(angle)
                hr_crop = hr_crop.rotate(angle)
            
            # 转换为tensor
            lr_tensor = self.to_tensor(lr_crop)
            hr_tensor = self.to_tensor(hr_crop)
        else:
            # 验证模式不裁剪，只转换为tensor
            lr_tensor = self.to_tensor(lr_img)
            hr_tensor = self.to_tensor(hr_img)
        
        return lr_tensor, hr_tensor

class AttentionWeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(9, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        
        self.attention = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Softmax(dim=1)  # 使用Softmax确保权重和为1
        )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))
        weights = self.attention(feat)
        return weights

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        return self.alpha * l1 + self.beta * mse

def get_random_indices(total_size, sample_size=32):
    """随机抽取指定数量的索引"""
    indices = torch.randperm(total_size)[:sample_size]
    return indices.tolist()

def main():
    # 创建保存目录
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    print("正在加载基础模型...")
    dat_model = DAT().to(device).eval()
    swinir_model = SwinIR().to(device).eval()
    rfdn_model = RFDN().to(device).eval()

    # 加载预训练权重
    dat_model.load_state_dict(torch.load("./model_zoo/team27_dat.pth"))
    swinir_model.load_state_dict(torch.load("./model_zoo/team27_swinir.pth"))
    rfdn_model.load_state_dict(torch.load("./model_zoo/team27_rfdn.pth"))

    # 初始化权重网络和优化器
    weight_net = AttentionWeightNet().to(device)

    optimizer = optim.Adam(weight_net.parameters(), lr=1e-4)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    criterion = CombinedLoss()

    # 数据加载
    print("正在加载数据集...")
    train_dataset = DIV2KDataset(
        lr_dir='./data/DIV2K_train_LR',
        hr_dir='./data/DIV2K_train_HR',
        is_train=True
    )
    val_dataset = DIV2KDataset(
        lr_dir='./data/DIV2K_valid_LR',
        hr_dir='./data/DIV2K_valid_HR',
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=72, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 训练循环
    print("开始训练...")
    num_epochs = 100
    best_psnr = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        weight_net.train()
        epoch_loss = 0
        start_time = time.time()
        
        for i, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            
            # 获取三个模型的输出
            with torch.no_grad():
                out_dat = dat_model(lr)
                out_swinir = swinir_model(lr)
                out_rfdn = rfdn_model(lr)
            
            # 拼接输出并预测权重
            combined = torch.cat([out_dat, out_swinir, out_rfdn], dim=1)
            weights = weight_net(combined)
            
            # 加权融合
            w1, w2, w3 = weights.chunk(3, dim=1)
            fused = w1 * out_dat + w2 * out_swinir + w3 * out_rfdn
            
            # 计算损失
            loss = criterion(fused, hr)
            epoch_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        # 验证阶段
        weight_net.eval()
        val_psnr = 0
        
        # 获取验证集总数并随机抽样
        total_val_samples = len(val_loader.dataset)
        sampled_indices = get_random_indices(total_val_samples, sample_size=28)  # 随机抽取28张图片
        
        with torch.no_grad():
            for idx, (lr, hr) in enumerate(val_loader):
                # 只处理被抽样的图片
                if idx not in sampled_indices:
                    continue
                    
                lr, hr = lr.to(device), hr.to(device)
                
                out_dat = dat_model(lr)
                out_swinir = swinir_model(lr)
                out_rfdn = rfdn_model(lr)
                
                combined = torch.cat([out_dat, out_swinir, out_rfdn], dim=1)
                weights = weight_net(combined)
                
                w1, w2, w3 = weights.chunk(3, dim=1)
                fused = w1 * out_dat + w2 * out_swinir + w3 * out_rfdn
                
                val_psnr += calculate_psnr(fused, hr)
        
        # 使用实际验证的图片数量计算平均PSNR
        avg_val_psnr = val_psnr / len(sampled_indices)
        scheduler.step(avg_val_psnr)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, "
              f"Val PSNR: {avg_val_psnr:.2f}, Time: {time.time()-start_time:.2f}s")

        # 每25个epoch保存一次模型
        if (epoch + 1) % 25 == 0:
            save_path = os.path.join(save_dir, f'fusion_weights_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'weight_net_state_dict': weight_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'psnr': avg_val_psnr,
            }, save_path)
            print(f"模型已保存到: {save_path}")
        
        # 保存最佳模型
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            best_save_path = os.path.join(save_dir, 'fusion_weights_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'weight_net_state_dict': weight_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'psnr': best_psnr,
            }, best_save_path)
            print(f"最佳模型已保存到: {best_save_path}，PSNR: {best_psnr:.2f}")

if __name__ == '__main__':
    main()

