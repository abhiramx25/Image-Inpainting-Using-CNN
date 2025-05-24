import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
from model import InpaintingCNN
from utils import InpaintingDataset

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(InpaintingDataset(train_data), batch_size=64, shuffle=True)

model = InpaintingCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    for corrupted, original, mask in train_loader:
        output = model(corrupted)
        loss = criterion(output * mask, original * mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/inpainting_cnn.pth')
print("Model trained and saved.")
