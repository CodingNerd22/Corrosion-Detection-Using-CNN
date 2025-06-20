import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor

# --- CNN Segmenter class (same as before) ---
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
            nn.Conv2d(8, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)

# --- Inference Function ---
def predict_and_show(image_path, model, device):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image '{image_path}' not found or unreadable.")
        return

    image = cv2.resize(image, (256, 256))
    tensor = to_tensor(image).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_mask = model(tensor).squeeze().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.resize(cv2.imread(image_path), (256, 256)), cv2.COLOR_BGR2RGB))
    plt.title("Input Image")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask > 0.5, cmap="gray")
    plt.title("Predicted Mask")
    plt.tight_layout()
    plt.show()

# --- Main Script ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Segmenter().to(device)
    model.load_state_dict(torch.load("cnn_segmenter.pth", map_location=device))
    model.eval()

    while True:
        image_path = input("Enter the path to the image you want to test (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            break
        predict_and_show(image_path, model, device)
