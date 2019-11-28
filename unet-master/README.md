### U-Net-Denoiser
Solves CT image denoising in TensorFlow

Given input and target datasets both of 512 * 512 CT images, this U-Net learns the pattern of streaking noise and removes it to produce a cleaned output CT image.

Images are saved in the output_samples directory.

##### Original git hub project

This project was adopted from https://github.com/kkweon/UNet-in-Tensorflow

The main inspiration was taken from the paper: 

##### U-Net: Convolutional Networks for Biomedical Image Segmentation

- read a paper (taken from <https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/>)



image below is taken from the paper above

**Fig.1.**  U-Net: Convolutional Networks for Biomedical Image Segmentation.



![UNET](/diagrams/UNET.png)



**Fig.2.** Our adopted U-Net architecture for image denoising. 



![unet.io](/diagrams/unet.io.png)



##### How to set up the environment

```{python}
(base) [npovey@ka data]$ conda create --name myenv python=3.6
(base) [npovey@ka data]$ conda activate myenv
(myenv) [npovey@ka data]$ pip install --upgrade tensorflow
(myenv) [npovey@ka U-Net-Denoiser-master]$ pip install pandas
(myenv) [npovey@ka U-Net-Denoiser-master]$ pip install Pillow
(myenv) [npovey@ka U-Net-Denoiser-master]$ pip install tensorflow-gpu
(myenv) [npovey@ka U-Net-Denoiser-master]$ python3 unet.py

```



#### Results

| Low dose image | Avg PSNR | Avg SSIM |
| -------------- | -------- | -------- |
| sparseview_60  | 31.18    | 0.8497   |
| sparseview_90  | 34.38    | 0.8888   |
| sparseview_180 | 39.15    | 0.9279   |

##### 
