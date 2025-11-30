# Segmentation — Model Download & Usage

This folder contains the code for running the trained UNet segmentation model.

Because the trained model file (`UNet_best_bs8.keras`, ~260 MB) is too large to store in GitHub, we provide a script to download it from Google Drive.

## Download the model

In the terminal, from this `segmentation/` directory, run:

```bash
bash download_model.sh
```
You may need to run
```
pip install gdown
```

## Test the model using the example notebook
```
loading_segmentation_model.ipynb
```
This notebook demonstrates how to load the pretrained UNet model for water segmentation and also includes definitions for the
required custom layers (Encoder and Decoder). \
You can try the model on four example images stored in `our_images/`.

To use the notebook, open it in VS Code, Jupyter, or Colab, and make sure that UNET_best_bs8.keras exists in the `segmentation/` folder. Then run all cells. 

To make sure you have installed the required modules, create a virtual environment and run
```
pip install -r requirements.txt
```
in the `segmentation/` folder

## Acknowledgements

This model was trained using the **ADE20K** dataset and the **river_segs** subset from the *water_v2* dataset, available at:  
https://www.kaggle.com/datasets/gvclsu/water-segmentation-dataset

The model architecture and training workflow were developed by adapting methods from the following Kaggle notebook:  
https://www.kaggle.com/code/utkarshsaxenadn/water-body-segmentation-unet-model

The images supplied are from Earthwatch's Great UK Water Blitz survey:\
https://www.freshwaterwatch.org/pages/great-uk-waterblitz-results

