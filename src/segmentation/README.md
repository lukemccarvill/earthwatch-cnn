# Segmentation — Model Download & Usage

This folder contains the code for running the trained UNet segmentation model.

Because the trained model file (`UNet_best_bs8.keras`, ~260 MB) is too large to store in GitHub, we provide a script to download it from Google Drive.

## Download the model

In the terminal, from this `segmentation/` directory, run:

```bash
bash download_model.sh
