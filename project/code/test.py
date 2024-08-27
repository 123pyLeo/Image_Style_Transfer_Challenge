import cv2
import glob
import numpy as np
import torch
import random  
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.utils.data as D
from torchvision import transforms as T
import albumentations as A
import segmentation_models_pytorch as smp
import os
import shutil

# Load images
test_img = glob.glob('../xfdata/test/*_input.jpg')
test_img.sort()

# Dataset class
class FoodDataset(D.Dataset):
    def __init__(self, images, masks=None, transform=None):
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
        if self.masks is not None:
            mask = cv2.imread(self.masks[index])
            mask = np.transpose(mask, (2, 0, 1))
            return self.as_tensor(image), mask / 255.0
        else:
            return self.as_tensor(image)

    def __len__(self):
        return len(self.images)

# Data transforms
trfm = A.Compose([A.Resize(512, 512)])

# Test dataset and loader
test_ds = FoodDataset(test_img, transform=trfm)
test_loader = D.DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=3)

# Model
model = smp.Unet(
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,           # No pre-trained weights for testing
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3
)

# Load the trained model weights
model.load_state_dict(torch.load('../user_data/tmp_data/new_model_weights.pth'))
model.to('cuda')
model.eval()

# Create the prediction_result directory if it doesn't exist
os.makedirs('../prediction_result', exist_ok=True)

# Create the submit directory inside prediction_result if it doesn't exist
submit_dir = '../prediction_result/submit'
os.makedirs(submit_dir, exist_ok=True)

# Test and save predictions
idx = 0
with torch.no_grad():
    for image in test_loader:
        image = image.to('cuda')
        outputs = model(image)
        outputs = (outputs.data.cpu().numpy() * 255).astype(np.uint8)
        for output in outputs:
            output = np.transpose(output, (1, 2, 0))
            output_path = os.path.join(submit_dir, test_img[idx].split('/')[-1].replace('input', 'target'))
            cv2.imwrite(output_path, output)
            idx += 1

# Zip the submission directory while preserving the directory structure
shutil.make_archive('../prediction_result/submit', 'zip', root_dir='../prediction_result', base_dir='submit')