import os
from torchvision.utils import save_image

def save_generated_images(fake_images, output_dir="generated_images", count=0):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(fake_images):
        save_path = os.path.join(output_dir, f"gen_{count+i}.png")
        save_image(img, save_path, normalize=True)
