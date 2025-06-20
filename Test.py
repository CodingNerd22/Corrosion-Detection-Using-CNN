import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Dataset class ---
class CorrosionDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_label.jpg'))

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)  # grayscale mask

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        image = to_tensor(image).float()
        mask = (mask > 0).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask

# --- CNN Segmentation Model ---
class CNN_Segmenter(nn.Module):
    def __init__(self):
        super(CNN_Segmenter, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2), nn.ReLU(),
            nn.Conv2d(8, 1, 1)  # output 1-channel mask
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)

# --- Training Function ---
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# --- Main Execution ---
if __name__ == "__main__":
    img_dir = "/Users/user/VIIT/BTech/SEM_8/CV/MiniProject/ry392rp8cj-1/HiRes/raw"
    mask_dir = "/Users/user/VIIT/BTech/SEM_8/CV/MiniProject/ry392rp8cj-1/HiRes/labeled"

    dataset = CorrosionDataset(img_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Segmenter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(100):
        loss = train(model, loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), "cnn_segmenter.pth")

import matplotlib.pyplot as plt

# Load model for inference
model = CNN_Segmenter().to(device)
model.load_state_dict(torch.load("cnn_segmenter.pth", map_location=device))
model.eval()

# Load dataset again if needed
dataset = CorrosionDataset(img_dir, mask_dir)

# Visualize a few predictions
for i in range(3):  # visualize first 3 samples
    image, true_mask = dataset[i]
    with torch.no_grad():
        pred_mask = model(image.unsqueeze(0).to(device)).squeeze().cpu().numpy()

    # Display
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask.squeeze(), cmap="gray")
    plt.title("Ground Truth Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask > 0.5, cmap="gray")
    plt.title("Predicted Mask")
    plt.tight_layout()
    plt.show()
