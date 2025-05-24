import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

class InpaintingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        mask = torch.zeros_like(image)
        mask[:, 8:24, 8:24] = 1
        corrupted_image = image * (1 - mask)
        return corrupted_image, image, mask

def visualize_results(model, dataloader, num_images=5):
    model.eval()
    with torch.no_grad():
        for i, (corrupted, original, _) in enumerate(dataloader):
            if i >= num_images:
                break
            output = model(corrupted)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(corrupted[0].permute(1, 2, 0))
            axes[0].set_title('Corrupted Image')
            axes[1].imshow(output[0].permute(1, 2, 0))
            axes[1].set_title('Reconstructed Image')
            axes[2].imshow(original[0].permute(1, 2, 0))
            axes[2].set_title('Original Image')
            plt.show()

def evaluate_metrics(model, dataloader, num_samples=10):
    psnr_total, ssim_total = 0, 0
    model.eval()
    with torch.no_grad():
        for i, (corrupted, original, _) in enumerate(dataloader):
            if i >= num_samples:
                break
            output = model(corrupted)
            reconstructed = output[0].permute(1, 2, 0).cpu().numpy()
            original_img = original[0].permute(1, 2, 0).cpu().numpy()
            psnr_total += psnr(original_img, reconstructed, data_range=1.0)
            ssim_total += ssim(original_img, reconstructed, channel_axis=-1, win_size=7, data_range=1.0)

    print(f"Average PSNR: {psnr_total / num_samples:.4f}")
    print(f"Average SSIM: {ssim_total / num_samples:.4f}")
