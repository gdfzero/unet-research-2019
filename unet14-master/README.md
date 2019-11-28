####UNet14

This version uses new unet model (keras implementation with batch normalization) and does image augmentation without cropping.

Need the following environment to run the code.

```{python}
(base) [npovey@ka ~]$ conda create -n keras-gpu python=3.6 numpy scipy keras-gpu
(base) [npovey@ka unet4]$ conda activate keras-gpu
(keras-gpu) [npovey@ka unet4]$ pip install pandas
(keras-gpu) [npovey@ka unet4]$ pip install Pillow
(keras-gpu) [npovey@ka unet4]$ pip install matplotlib
(keras-gpu) [npovey@ka unet4]$ python3 main.py 

```

other useful commands

```{
export CUDA_VISIBLE_DEVICES=0
(keras-gpu) [npovey@ka unet14]$ kill -KILL 43377
(keras-gpu) [npovey@ka unet14]$ screen -S unet148
(press ctlr+A, D to detach from screen)
[detached from 50566.unet148]
```

##### View a Tensorboard

```py
tensorboard --logdir=./logs
http://0.0.0.0:6006
```





##### UNet14 (final model)

| Low Dose Image | DnCnn                               | UNet14                              |
| -------------- | ----------------------------------- | ----------------------------------- |
| sparseview_60  | Avg PSNR: 32.30    Avg SSIM: 0.8560 | Avg PSNR: 33.12	Avg SSIM: 0.8823 |
| sparseview_90  | Avg PSNR: 35.13    Avg SSIM: 0.8892 | Avg PSNR: 35.45	Avg SSIM: 0.9043 |
| sparseview_180 | Avg PSNR: 39.77    Avg SSIM: 0.9341 | Avg PSNR: 39.28	Avg SSIM: 0.9303 |
| ldct_7e4       |                                     | Avg PSNR: 41.75	Avg SSIM: 0.9426 |
| ldct_1e5       |                                     | Avg PSNR: 42.16	Avg SSIM: 0.9444 |
| ldct_2e5       |                                     | Avg PSNR: 42.70	Avg SSIM: 0.9466 |



##### UNet6 results (no cropping)

| Low Dose Image | DnCnn                               | UNet6                               |
| -------------- | ----------------------------------- | ----------------------------------- |
| sparseview_60  | Avg PSNR: 32.30    Avg SSIM: 0.8560 | Avg PSNR: 33.26	Avg SSIM: 0.8851 |
| sparseview_90  | Avg PSNR: 35.13    Avg SSIM: 0.8892 | Avg PSNR: 35.53	Avg SSIM: 0.9042 |
| sparseview_180 | Avg PSNR: 39.77    Avg SSIM: 0.9341 | Avg PSNR: 39.44	Avg SSIM: 0.9315 |



