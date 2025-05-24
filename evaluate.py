import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import InpaintingCNN
from utils import InpaintingDataset, evaluate_metrics, visualize_results

# Load dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = DataLoader(InpaintingDataset(data), batch_size=1, shuffle=True)

# Load model
model = InpaintingCNN()
model.load_state_dict(torch.load('models/inpainting_cnn.pth'))
model.eval()

# Run evaluation
visualize_results(model, loader, num_images=5)
evaluate_metrics(model, loader, num_samples=10)
