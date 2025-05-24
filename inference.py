import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import InpaintingCNN

model = InpaintingCNN()
model.load_state_dict(torch.load('models/inpainting_cnn.pth'))
model.eval()

def inference_on_custom_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    mask = torch.zeros_like(image)
    mask[:, :, 8:24, 8:24] = 1
    corrupted_image = image * (1 - mask)

    with torch.no_grad():
        output = model(corrupted_image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(corrupted_image[0].permute(1, 2, 0))
    axes[0].set_title('Corrupted Image')
    axes[1].imshow(output[0].permute(1, 2, 0))
    axes[1].set_title('Reconstructed Image')
    axes[2].imshow(image[0].permute(1, 2, 0))
    axes[2].set_title('Original Image')
    plt.show()

inference_on_custom_image('catpic.jpg')
