####  Unet2 vs. UNet

| Low Dose Image | Unet2 (extra conv2D layer)          | Unet                                |
| -------------- | ----------------------------------- | ----------------------------------- |
| sparseview_60  | Avg PSNR: 31.71	Avg SSIM: 0.8460 | Avg PSNR: 31.18	Avg SSIM: 0.8497 |
| sparseview_90  | Avg PSNR: 34.13	Avg SSIM: 0.8835 | Avg PSNR: 34.38	Avg SSIM: 0.8888 |
| sparseview_180 |                                     | Avg PSNR: 39.15	Avg SSIM: 0.9279 |
| ldct_1e5       |                                     | Avg PSNR: 41.86	Avg SSIM: 0.9419 |
| ldct_2e5       |                                     | Avg PSNR: 42.35	Avg SSIM: 0.9441 |
| ldct_7e4       |                                     | Avg PSNR: 41.32	Avg SSIM: 0.9360 |



#### DnCNN vs. UNet Results

| Low Dose Image | DnCnn                               | Unet                                |
| -------------- | ----------------------------------- | ----------------------------------- |
| sparseview_60  | Avg PSNR: 32.30    Avg SSIM: 0.8560 | Avg PSNR: 31.18	Avg SSIM: 0.8497 |
| sparseview_90  | Avg PSNR: 35.13    Avg SSIM: 0.8892 | Avg PSNR: 34.38	Avg SSIM: 0.8888 |
| sparseview_180 | Avg PSNR: 39.77    Avg SSIM: 0.9341 | Avg PSNR: 39.15	Avg SSIM: 0.9279 |
| ldct_1e5       |                                     | Avg PSNR: 41.86	Avg SSIM: 0.9419 |
| ldct_2e5       |                                     | Avg PSNR: 42.35	Avg SSIM: 0.9441 |
| ldct_7e4       |                                     | Avg PSNR: 41.32	Avg SSIM: 0.9360 |

