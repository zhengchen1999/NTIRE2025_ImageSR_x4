## Installation

1. Install PyTorch by following the official installation guide from the [PyTorch website](https://pytorch.org/).

2. Install the necessary dependencies by following the instructions in our repository.


```python
pip install -r requirements.txt
python setup.py develop
```

```python
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
```
<hr/>

## To Inference
```python
python inference/inference_swinfir.py
```

## Testing
1. **Prepare Your Dataset:**

    - Place all your images in the datasets folder.

    - Ensure the folder structure matches the expected format for the SwinFIR model.

2. **Update Configuration File:**

    - Locate the configuration file at options/test/SwinFIR/SwinFIR_SRx4.yml.

    - Open the file in a text editor and modify the dataset-related options (e.g., name, dataroot_gt or dataroot_lq) to point to your dataset's location and settings.

3. **Run the Model:**

    - After updating the configuration, you can proceed to test the model with your dataset. The pretrained model is in 'experiments/pretrained_models/net_g_latest.pth'

```python
python swinfir/test.py -opt options/test/SwinFIR/SwinFIR_SRx4.yml
```


## Training
1. **Prepare Your Dataset:**

    - Place all your images in the datasets folder.

    - Ensure the folder structure matches the expected format for the SwinFIR model.

2. **Update Configuration File:**

    - Locate the configuration file at options/train/SwinFIR/train_SwinFIR_SRx4_from_scratch.yml.

    - Open the file in a text editor and modify the dataset-related options (e.g., name, dataroot_gt or dataroot_lq) to point to your dataset's location and settings.

3. **Run the Model:**

    - After updating the configuration, you can proceed to train or run the model with your dataset.

```python
python swinfir/train.py -opt options/train/SwinFIR/train_SwinFIR_SRx4_from_scratch.yml
```

## Results

**SWinSTASR Results (X4):**
|  Dataset   |   PSNR    |    SSIM    |   LPIPS  |
|     :--    |     :-:   |      :-:   |     :-:  |
|    Set5    |  32.7260  |   0.9013   |  0.1726  |
|    Set14   |  29.0286  |   0.7924   |  0.2741  |
|  DIV2K_valid  |  31.0018  |   0.8501   |  0.2425  |

