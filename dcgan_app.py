import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset_loader import get_dataloader
from utils.save_images import save_generated_images
import os, zipfile, shutil
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import sys
sys.path.append('./models')

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧠 DCGAN - Synthetic Image Generator")

# Training parameters (always visible)
epochs = st.slider("Epochs", 1, 100, 5, key="epochs_slider")
batch_size = st.slider("Batch size", 16, 256, 64, key="batch_size_slider")
num_generate = st.number_input("How many synthetic images do you need?", 1000, 50000, 10000, step=1000, key="num_generate_input")
image_size = st.slider("Image Size", 32, 128, 64, key="image_size_slider")

# ----------------------------
# Prepare dataset folder
# ----------------------------


epochs = st.slider("Epochs", 1, 100, 5)
batch_size = st.slider("Batch size", 16, 256, 64)
num_generate = st.number_input("How many synthetic images do you need?", 1000, 50000, 10000, step=1000)
image_size = 64

# Use fixed CIFAR-10 path
# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (ZIP of images)", type=["zip"])
data_dir = uploaded_file if uploaded_file is not None else ""

# ----------------------------
# Training Setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 100
lr = 0.0002

if st.button("Train and Generate"):
    loader = get_dataloader(data_dir, image_size=image_size, batch_size=batch_size)

    gen = Generator(z_dim=z_dim).to(device)
    disc = Discriminator().to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    st.write("Training started...")
    progress = st.progress(0)

    step = 0
    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            noise = torch.randn(real.size(0), z_dim, 1, 1).to(device)
            fake = gen(noise)

            # Train Discriminator
            disc_real = disc(real).reshape(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2

            opt_disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            # Train Generator
            output = disc(fake).reshape(-1)
            lossG = criterion(output, torch.ones_like(output))
            opt_gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            step += 1

        progress.progress((epoch+1)/epochs)
        st.write(f"Epoch [{epoch+1}/{epochs}] Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")

    st.success("Training completed! ✅ Now generating synthetic images...")

    # ----------------------------
    # Generate Synthetic Images
    # ----------------------------
    gen.eval()
    count = 0
    out_dir = "generated_images/"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    while count < num_generate:
        with torch.no_grad():
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise).cpu()
            save_generated_images(fake, count=count)
            count += batch_size

    # Show preview
    st.subheader("Generated Image Samples")
    with torch.no_grad():
        noise = torch.randn(16, z_dim, 1, 1).to(device)
        fake = gen(noise).cpu()
        grid = make_grid(fake, nrow=4, normalize=True)
        plt.figure(figsize=(6,6))
        plt.imshow(grid.permute(1, 2, 0).detach().numpy())
        plt.axis("off")
        st.pyplot(plt)

    # Zip the dataset
    zip_path = "generated_images.zip"
    shutil.make_archive("generated_images", 'zip', out_dir)

    with open(zip_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Generated Dataset (ZIP)",
            data=f,
            file_name="generated_images.zip",
            mime="application/zip"
        )

    st.success(f"✅ Done! {num_generate} synthetic images saved and ready to download.")
