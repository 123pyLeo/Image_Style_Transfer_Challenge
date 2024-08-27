import cv2
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision
from torchvision import transforms as T
import albumentations as A
from tqdm import tqdm
import segmentation_models_pytorch as smp
from math import exp
import random

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load images
train_img = glob.glob('../xfdata/train/*_input.jpg')
train_mask = glob.glob('../xfdata/train/*_target.jpg')
train_img.sort()
train_mask.sort()

# Dataset class
class FoodDataset(D.Dataset):
    def __init__(self, images, masks, transform):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        mask = cv2.imread(self.masks[index])
        mask = np.transpose(mask, (2, 0, 1))
        return self.as_tensor(image), mask / 255.0

    def __len__(self):
        return len(self.images)

# Data transforms
trfm = A.Compose([A.Resize(512, 512)])
train_ds = FoodDataset(train_img[:-200], train_mask[:-200], transform=trfm)
val_ds = FoodDataset(train_img[-200:], train_mask[-200:], transform=trfm)
train_loader = D.DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=3)
val_loader = D.DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=3)

# Model
model = smp.Unet(
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3
)

# Loss functions
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, height, width) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, self.window_size, self.size_average)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.alpha = alpha

    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        ssim_loss = self.ssim(output, target)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss

# Validation function
@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to('cuda'), target.to('cuda').float()
        output = model(image)
        loss = loss_fn(output, target)
        losses.append(loss.item())
    avg_loss = np.array(losses).mean()
    return avg_loss

# Training
model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
combined_loss_fn = CombinedLoss(alpha=0.5)  # Adjust alpha as needed

EPOCHES = 13
WARMUP_EPOCHES = 5  # Number of warmup epochs
initial_lr = 1e-3
warmup_lr = initial_lr / 10  # Starting learning rate for warmup

train_losses = []
val_losses = []

for epoch in range(1, EPOCHES + 1):
    # Adjust learning rate for warmup
    if epoch <= WARMUP_EPOCHES:
        lr = warmup_lr + (initial_lr - warmup_lr) * (epoch / WARMUP_EPOCHES)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = initial_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    losses = []
    start_time = time.time()
    model.train()
    for image, target in tqdm(train_loader):
        image, target = image.to('cuda'), target.to('cuda').float()
        optimizer.zero_grad()
        output = model(image)
        loss = combined_loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    avg_train_loss = np.mean(losses)
    avg_loss = validation(model, val_loader, combined_loss_fn)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_loss)
    
    print(f"Epoch {epoch}/{EPOCHES}")
    print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_loss:.4f}")
    print(f"Time: {(time.time() - start_time) / 60:.2f} minutes\n")

# Save the model weights
torch.save(model.state_dict(), '../user_data/tmp_data/model_weights.pth')

# Initialize model
model = smp.Unet(
    encoder_name="resnet18",  # using the same encoder
    encoder_weights=None,     # not using ImageNet pretrained weights, as we load custom pretrained weights
    in_channels=3,            # input channels
    classes=3                 # number of classes
)

# Load trained weights
model.load_state_dict(torch.load('../user_data/tmp_data/model_weights.pth'))
model.to('cuda')  

# Optionally: freeze encoder weights, only train decoder part
for param in model.encoder.parameters():
    param.requires_grad = False

# Set optimizer with lower learning rate
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Training loop
EPOCHES = 10  # set number of epochs
for epoch in range(1, EPOCHES + 1):
    model.train()
    losses = []
    for image, target in tqdm(train_loader):
        image, target = image.to('cuda'), target.to('cuda').float()
        optimizer.zero_grad()
        output = model(image)
        loss = combined_loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    avg_train_loss = np.mean(losses)
    avg_loss = validation(model, val_loader, combined_loss_fn)
    
    print(f"Epoch {epoch}/{EPOCHES}")
    print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_loss:.4f}")

# Optionally save new weights
torch.save(model.state_dict(), '../user_data/tmp_data/new_model_weights.pth')
