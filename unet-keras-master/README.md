##### Unet-keras

```{python}
(base) [npovey@ka ~]$ conda create -n keras-gpu python=3.6 numpy scipy keras-gpu
(base) [npovey@ka unet4]$ conda activate keras-gpu
(keras-gpu) [npovey@ka unet4]$ pip install pandas
(keras-gpu) [npovey@ka unet4]$ pip install Pillow
(keras-gpu) [npovey@ka unet4]$ pip install matplotlib
(keras-gpu) [npovey@ka unet4]$ python3 main.py 
export CUDA_VISIBLE_DEVICES=0




```

This model was trained on .png images. It did not produce good results. Check unet-keras3 for the best results. Unet-keras3 is the same mode but it was trained on .flt images.

| sparseview_60 | PSNR Python        |
| ------------- | ------------------ |
| Epochs 50     | 35.99036362055831  |
| Epochs 100    | 36.06231315481163  |
| Epochs 150    | 36.2879197179926   |
| Epochs 200    | 36.25833039011886  |
| Epochs 250    | 36.188440508405655 |

##### Original results

| Low Dose Image | DnCnn                               | Unet                                |
| -------------- | ----------------------------------- | ----------------------------------- |
| sparseview_60  | Avg PSNR: 32.30    Avg SSIM: 0.8560 | Avg PSNR: 31.18	Avg SSIM: 0.8497 |
| sparseview_90  | Avg PSNR: 35.13    Avg SSIM: 0.8892 | Avg PSNR: 34.38	Avg SSIM: 0.8888 |
| sparseview_180 | Avg PSNR: 39.77    Avg SSIM: 0.9341 | Avg PSNR: 39.15	Avg SSIM: 0.9279 |



##### UNet6 results (no cropping)

| Low Dose Image | Python          | Matlab                              |
| -------------- | --------------- | ----------------------------------- |
| sparseview_60  | Avg PSNR: 33.26 | Avg PSNR: 33.26	Avg SSIM: 0.8851 |
| sparseview_90  | Avg PSNR: 35.53 | Avg PSNR: 35.53	Avg SSIM: 0.9042 |
| sparseview_180 | 6/38.71         |                                     |

##### 

##### UNet7 results (with cropping)

| Low Dose Image | Python             | Matlab                              |
| -------------- | ------------------ | ----------------------------------- |
| sparseview_60  | Avg PSNR: 33.37    | Avg PSNR: 33.37	Avg SSIM: 0.8856 |
| sparseview_90  | Avg PSNR: 27/35.26 | Avg PSNR: 	Avg SSIM:             |
| sparseview_180 |                    |                                     |

##### 







```{python}
1440/1440 [==============================] - 26s 18ms/step - loss: 0.0012 - tf_psnr: 29.2770 - val_loss: 8.8125e-04 - val_tf_psnr: 30.5608
Epoch 2/50
1440/1440 [==============================] - 17s 12ms/step - loss: 7.4568e-04 - tf_psnr: 31.2859 - val_loss: 6.8143e-04 - val_tf_psnr: 31.6780
Epoch 3/50
1440/1440 [==============================] - 17s 12ms/step - loss: 6.1616e-04 - tf_psnr: 32.1118 - val_loss: 5.7494e-04 - val_tf_psnr: 32.4133
Epoch 4/50
1440/1440 [==============================] - 17s 12ms/step - loss: 5.3717e-04 - tf_psnr: 32.7053 - val_loss: 5.0076e-04 - val_tf_psnr: 33.0111
Epoch 5/50
1440/1440 [==============================] - 17s 12ms/step - loss: 4.6965e-04 - tf_psnr: 33.2892 - val_loss: 4.3061e-04 - val_tf_psnr: 33.6666
Epoch 6/50
1440/1440 [==============================] - 17s 12ms/step - loss: 4.4271e-04 - tf_psnr: 33.5541 - val_loss: 3.9444e-04 - val_tf_psnr: 34.0446
Epoch 7/50
1440/1440 [==============================] - 17s 12ms/step - loss: 3.7583e-04 - tf_psnr: 34.2569 - val_loss: 3.5085e-04 - val_tf_psnr: 34.5530
Epoch 8/50
1440/1440 [==============================] - 17s 12ms/step - loss: 3.4563e-04 - tf_psnr: 34.6252 - val_loss: 3.1431e-04 - val_tf_psnr: 35.0275
Epoch 9/50
1440/1440 [==============================] - 17s 12ms/step - loss: 3.1293e-04 - tf_psnr: 35.0538 - val_loss: 3.0286e-04 - val_tf_psnr: 35.1879
Epoch 10/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.9809e-04 - tf_psnr: 35.2622 - val_loss: 3.0839e-04 - val_tf_psnr: 35.1091
Epoch 11/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.8720e-04 - tf_psnr: 35.4254 - val_loss: 2.8397e-04 - val_tf_psnr: 35.4679
Epoch 12/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.6623e-04 - tf_psnr: 35.7547 - val_loss: 2.6388e-04 - val_tf_psnr: 35.7867
Epoch 13/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.7095e-04 - tf_psnr: 35.7024 - val_loss: 2.5941e-04 - val_tf_psnr: 35.8610
Epoch 14/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.4864e-04 - tf_psnr: 36.0504 - val_loss: 2.5639e-04 - val_tf_psnr: 35.9126
Epoch 15/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.4385e-04 - tf_psnr: 36.1354 - val_loss: 2.6162e-04 - val_tf_psnr: 35.8244
Epoch 16/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.3560e-04 - tf_psnr: 36.2862 - val_loss: 2.6566e-04 - val_tf_psnr: 35.7579
Epoch 17/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.3247e-04 - tf_psnr: 36.3458 - val_loss: 2.4211e-04 - val_tf_psnr: 36.1616
Epoch 18/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.3773e-04 - tf_psnr: 36.2574 - val_loss: 2.3976e-04 - val_tf_psnr: 36.2035
Epoch 19/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.2130e-04 - tf_psnr: 36.5615 - val_loss: 2.3583e-04 - val_tf_psnr: 36.2766
Epoch 20/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.1512e-04 - tf_psnr: 36.6845 - val_loss: 2.3310e-04 - val_tf_psnr: 36.3277
Epoch 21/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.1422e-04 - tf_psnr: 36.7020 - val_loss: 2.3594e-04 - val_tf_psnr: 36.2747
Epoch 22/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.1021e-04 - tf_psnr: 36.7823 - val_loss: 2.2962e-04 - val_tf_psnr: 36.3930
Epoch 23/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.0855e-04 - tf_psnr: 36.8203 - val_loss: 2.3000e-04 - val_tf_psnr: 36.3851
Epoch 24/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.0710e-04 - tf_psnr: 36.8478 - val_loss: 2.3206e-04 - val_tf_psnr: 36.3467
Epoch 25/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.0112e-04 - tf_psnr: 36.9712 - val_loss: 2.2959e-04 - val_tf_psnr: 36.3950
Epoch 26/50
1440/1440 [==============================] - 17s 12ms/step - loss: 2.0081e-04 - tf_psnr: 36.9808 - val_loss: 2.3045e-04 - val_tf_psnr: 36.3792
Epoch 27/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.9334e-04 - tf_psnr: 37.1437 - val_loss: 2.2315e-04 - val_tf_psnr: 36.5210
Epoch 28/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.9469e-04 - tf_psnr: 37.1132 - val_loss: 2.4160e-04 - val_tf_psnr: 36.1742
Epoch 29/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.8946e-04 - tf_psnr: 37.2331 - val_loss: 2.3195e-04 - val_tf_psnr: 36.3487
Epoch 30/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.9237e-04 - tf_psnr: 37.1684 - val_loss: 2.2767e-04 - val_tf_psnr: 36.4308
Epoch 31/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.8562e-04 - tf_psnr: 37.3200 - val_loss: 2.2328e-04 - val_tf_psnr: 36.5170
Epoch 32/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.8731e-04 - tf_psnr: 37.2847 - val_loss: 2.4394e-04 - val_tf_psnr: 36.1303
Epoch 33/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.8365e-04 - tf_psnr: 37.3708 - val_loss: 2.1820e-04 - val_tf_psnr: 36.6157
Epoch 34/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.8046e-04 - tf_psnr: 37.4442 - val_loss: 2.2418e-04 - val_tf_psnr: 36.4974
Epoch 35/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.8254e-04 - tf_psnr: 37.4055 - val_loss: 2.5327e-04 - val_tf_psnr: 35.9665
Epoch 36/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.8105e-04 - tf_psnr: 37.4327 - val_loss: 2.2277e-04 - val_tf_psnr: 36.5253
Epoch 37/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.7512e-04 - tf_psnr: 37.5746 - val_loss: 2.1955e-04 - val_tf_psnr: 36.5915
Epoch 38/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.7174e-04 - tf_psnr: 37.6567 - val_loss: 2.1948e-04 - val_tf_psnr: 36.5923
Epoch 39/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.7485e-04 - tf_psnr: 37.5831 - val_loss: 2.1282e-04 - val_tf_psnr: 36.7264
Epoch 40/50
1440/1440 [==============================] - 18s 12ms/step - loss: 1.7152e-04 - tf_psnr: 37.6651 - val_loss: 2.1858e-04 - val_tf_psnr: 36.6132
Epoch 41/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.7194e-04 - tf_psnr: 37.6559 - val_loss: 2.2546e-04 - val_tf_psnr: 36.4749
Epoch 42/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.6906e-04 - tf_psnr: 37.7303 - val_loss: 2.2177e-04 - val_tf_psnr: 36.5475
Epoch 43/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.6852e-04 - tf_psnr: 37.7414 - val_loss: 2.1800e-04 - val_tf_psnr: 36.6216
Epoch 44/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.6410e-04 - tf_psnr: 37.8569 - val_loss: 2.1498e-04 - val_tf_psnr: 36.6883
Epoch 45/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.7336e-04 - tf_psnr: 37.6294 - val_loss: 2.1862e-04 - val_tf_psnr: 36.6148
Epoch 46/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.6226e-04 - tf_psnr: 37.9042 - val_loss: 2.1894e-04 - val_tf_psnr: 36.6021
Epoch 47/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.6069e-04 - tf_psnr: 37.9466 - val_loss: 2.1682e-04 - val_tf_psnr: 36.6471
Epoch 48/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.7219e-04 - tf_psnr: 37.6688 - val_loss: 2.2474e-04 - val_tf_psnr: 36.4928
Epoch 49/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5915e-04 - tf_psnr: 37.9895 - val_loss: 2.1272e-04 - val_tf_psnr: 36.7296
Epoch 50/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5723e-04 - tf_psnr: 38.0406 - val_loss: 2.3581e-04 - val_tf_psnr: 36.2880
psnr 35.71889802242022
Train on 1440 samples, validate on 160 samples
Epoch 1/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5987e-04 - tf_psnr: 37.9708 - val_loss: 2.1663e-04 - val_tf_psnr: 36.6527
Epoch 2/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5604e-04 - tf_psnr: 38.0751 - val_loss: 2.4421e-04 - val_tf_psnr: 36.1263
Epoch 3/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5891e-04 - tf_psnr: 37.9981 - val_loss: 2.1747e-04 - val_tf_psnr: 36.6324
Epoch 4/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5260e-04 - tf_psnr: 38.1689 - val_loss: 2.1705e-04 - val_tf_psnr: 36.6417
Epoch 5/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5913e-04 - tf_psnr: 37.9946 - val_loss: 2.1169e-04 - val_tf_psnr: 36.7498
Epoch 6/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5320e-04 - tf_psnr: 38.1539 - val_loss: 2.1581e-04 - val_tf_psnr: 36.6716
Epoch 7/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5409e-04 - tf_psnr: 38.1342 - val_loss: 2.1653e-04 - val_tf_psnr: 36.6525
Epoch 8/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5150e-04 - tf_psnr: 38.2032 - val_loss: 2.1909e-04 - val_tf_psnr: 36.6024
Epoch 9/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5485e-04 - tf_psnr: 38.1169 - val_loss: 2.1430e-04 - val_tf_psnr: 36.6997
Epoch 10/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4811e-04 - tf_psnr: 38.3010 - val_loss: 2.1114e-04 - val_tf_psnr: 36.7691
Epoch 11/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.5307e-04 - tf_psnr: 38.1669 - val_loss: 2.1499e-04 - val_tf_psnr: 36.6847
Epoch 12/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4792e-04 - tf_psnr: 38.3053 - val_loss: 2.1777e-04 - val_tf_psnr: 36.6261
Epoch 13/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4765e-04 - tf_psnr: 38.3134 - val_loss: 2.1566e-04 - val_tf_psnr: 36.6738
Epoch 14/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4951e-04 - tf_psnr: 38.2708 - val_loss: 2.1713e-04 - val_tf_psnr: 36.6473
Epoch 15/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4505e-04 - tf_psnr: 38.3945 - val_loss: 2.1429e-04 - val_tf_psnr: 36.6988
Epoch 16/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4874e-04 - tf_psnr: 38.3072 - val_loss: 2.2382e-04 - val_tf_psnr: 36.5120
Epoch 17/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4823e-04 - tf_psnr: 38.3009 - val_loss: 2.1312e-04 - val_tf_psnr: 36.7265
Epoch 18/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4124e-04 - tf_psnr: 38.5081 - val_loss: 2.1849e-04 - val_tf_psnr: 36.6159
Epoch 19/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4334e-04 - tf_psnr: 38.4431 - val_loss: 2.1573e-04 - val_tf_psnr: 36.6722
Epoch 20/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4544e-04 - tf_psnr: 38.3848 - val_loss: 2.1678e-04 - val_tf_psnr: 36.6508
Epoch 21/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3984e-04 - tf_psnr: 38.5489 - val_loss: 2.1701e-04 - val_tf_psnr: 36.6447
Epoch 22/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4594e-04 - tf_psnr: 38.3654 - val_loss: 2.1538e-04 - val_tf_psnr: 36.6777
Epoch 23/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3793e-04 - tf_psnr: 38.6100 - val_loss: 2.1244e-04 - val_tf_psnr: 36.7389
Epoch 24/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.4608e-04 - tf_psnr: 38.3784 - val_loss: 2.1590e-04 - val_tf_psnr: 36.6663
Epoch 25/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3790e-04 - tf_psnr: 38.6099 - val_loss: 2.1416e-04 - val_tf_psnr: 36.7080
Epoch 26/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3781e-04 - tf_psnr: 38.6143 - val_loss: 2.1840e-04 - val_tf_psnr: 36.6132
Epoch 27/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3827e-04 - tf_psnr: 38.5989 - val_loss: 2.1542e-04 - val_tf_psnr: 36.6790
Epoch 28/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3818e-04 - tf_psnr: 38.6057 - val_loss: 2.1394e-04 - val_tf_psnr: 36.7099
Epoch 29/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3793e-04 - tf_psnr: 38.6105 - val_loss: 2.3521e-04 - val_tf_psnr: 36.2887
Epoch 30/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3800e-04 - tf_psnr: 38.6110 - val_loss: 2.1607e-04 - val_tf_psnr: 36.6690
Epoch 31/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3582e-04 - tf_psnr: 38.6771 - val_loss: 2.1434e-04 - val_tf_psnr: 36.6997
Epoch 32/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3669e-04 - tf_psnr: 38.6488 - val_loss: 2.0876e-04 - val_tf_psnr: 36.8198
Epoch 33/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3494e-04 - tf_psnr: 38.7057 - val_loss: 2.1979e-04 - val_tf_psnr: 36.5890
Epoch 34/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3471e-04 - tf_psnr: 38.7119 - val_loss: 2.1727e-04 - val_tf_psnr: 36.6443
Epoch 35/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3628e-04 - tf_psnr: 38.6645 - val_loss: 2.1143e-04 - val_tf_psnr: 36.7603
Epoch 36/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3319e-04 - tf_psnr: 38.7627 - val_loss: 2.2681e-04 - val_tf_psnr: 36.4559
Epoch 37/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3435e-04 - tf_psnr: 38.7253 - val_loss: 2.1205e-04 - val_tf_psnr: 36.7504
Epoch 38/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3266e-04 - tf_psnr: 38.7788 - val_loss: 2.2713e-04 - val_tf_psnr: 36.4429
Epoch 39/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3424e-04 - tf_psnr: 38.7282 - val_loss: 2.1236e-04 - val_tf_psnr: 36.7389
Epoch 40/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3153e-04 - tf_psnr: 38.8147 - val_loss: 2.1763e-04 - val_tf_psnr: 36.6348
Epoch 41/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3074e-04 - tf_psnr: 38.8426 - val_loss: 2.1284e-04 - val_tf_psnr: 36.7374
Epoch 42/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3195e-04 - tf_psnr: 38.8025 - val_loss: 2.1149e-04 - val_tf_psnr: 36.7645
Epoch 43/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2944e-04 - tf_psnr: 38.8868 - val_loss: 2.3501e-04 - val_tf_psnr: 36.2963
Epoch 44/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3498e-04 - tf_psnr: 38.7168 - val_loss: 2.1704e-04 - val_tf_psnr: 36.6464
Epoch 45/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2957e-04 - tf_psnr: 38.8854 - val_loss: 2.1347e-04 - val_tf_psnr: 36.7242
Epoch 46/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2722e-04 - tf_psnr: 38.9611 - val_loss: 2.1192e-04 - val_tf_psnr: 36.7546
Epoch 47/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3049e-04 - tf_psnr: 38.8578 - val_loss: 2.1778e-04 - val_tf_psnr: 36.6286
Epoch 48/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2659e-04 - tf_psnr: 38.9813 - val_loss: 2.1603e-04 - val_tf_psnr: 36.6762
Epoch 49/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2917e-04 - tf_psnr: 38.8940 - val_loss: 2.1669e-04 - val_tf_psnr: 36.6568
Epoch 50/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.3018e-04 - tf_psnr: 38.8689 - val_loss: 2.2073e-04 - val_tf_psnr: 36.5760
psnr 36.123826697716204
Train on 1440 samples, validate on 160 samples
Epoch 1/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2751e-04 - tf_psnr: 38.9503 - val_loss: 2.1324e-04 - val_tf_psnr: 36.7252
Epoch 2/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2347e-04 - tf_psnr: 39.0885 - val_loss: 2.1242e-04 - val_tf_psnr: 36.7462
Epoch 3/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2631e-04 - tf_psnr: 38.9917 - val_loss: 2.1258e-04 - val_tf_psnr: 36.7400
Epoch 4/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2871e-04 - tf_psnr: 38.9122 - val_loss: 2.1959e-04 - val_tf_psnr: 36.6046
Epoch 5/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2397e-04 - tf_psnr: 39.0732 - val_loss: 2.1120e-04 - val_tf_psnr: 36.7692
Epoch 6/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2728e-04 - tf_psnr: 38.9647 - val_loss: 2.1518e-04 - val_tf_psnr: 36.6911
Epoch 7/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2393e-04 - tf_psnr: 39.0753 - val_loss: 2.1479e-04 - val_tf_psnr: 36.6931
Epoch 8/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2522e-04 - tf_psnr: 39.0357 - val_loss: 2.1052e-04 - val_tf_psnr: 36.7857
Epoch 9/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2344e-04 - tf_psnr: 39.0938 - val_loss: 2.1384e-04 - val_tf_psnr: 36.7174
Epoch 10/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2295e-04 - tf_psnr: 39.1087 - val_loss: 2.1778e-04 - val_tf_psnr: 36.6405
Epoch 11/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2380e-04 - tf_psnr: 39.0827 - val_loss: 2.1019e-04 - val_tf_psnr: 36.7913
Epoch 12/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2271e-04 - tf_psnr: 39.1168 - val_loss: 2.1631e-04 - val_tf_psnr: 36.6766
Epoch 13/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2084e-04 - tf_psnr: 39.1834 - val_loss: 2.1344e-04 - val_tf_psnr: 36.7196
Epoch 14/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2211e-04 - tf_psnr: 39.1395 - val_loss: 2.1273e-04 - val_tf_psnr: 36.7425
Epoch 15/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2506e-04 - tf_psnr: 39.0516 - val_loss: 2.0985e-04 - val_tf_psnr: 36.7965
Epoch 16/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2129e-04 - tf_psnr: 39.1683 - val_loss: 2.1445e-04 - val_tf_psnr: 36.7002
Epoch 17/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2030e-04 - tf_psnr: 39.2037 - val_loss: 2.1309e-04 - val_tf_psnr: 36.7320
Epoch 18/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2196e-04 - tf_psnr: 39.1452 - val_loss: 2.1588e-04 - val_tf_psnr: 36.6760
Epoch 19/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1905e-04 - tf_psnr: 39.2488 - val_loss: 2.1420e-04 - val_tf_psnr: 36.7169
Epoch 20/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2477e-04 - tf_psnr: 39.0541 - val_loss: 2.1761e-04 - val_tf_psnr: 36.6350
Epoch 21/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2009e-04 - tf_psnr: 39.2131 - val_loss: 2.1228e-04 - val_tf_psnr: 36.7468
Epoch 22/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1697e-04 - tf_psnr: 39.3229 - val_loss: 2.1444e-04 - val_tf_psnr: 36.7093
Epoch 23/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2036e-04 - tf_psnr: 39.2048 - val_loss: 2.1345e-04 - val_tf_psnr: 36.7287
Epoch 24/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1699e-04 - tf_psnr: 39.3230 - val_loss: 2.1203e-04 - val_tf_psnr: 36.7546
Epoch 25/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.2394e-04 - tf_psnr: 39.0962 - val_loss: 2.1240e-04 - val_tf_psnr: 36.7488
Epoch 26/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1519e-04 - tf_psnr: 39.3921 - val_loss: 2.1274e-04 - val_tf_psnr: 36.7396
Epoch 27/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1854e-04 - tf_psnr: 39.2670 - val_loss: 2.2100e-04 - val_tf_psnr: 36.5753
Epoch 28/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1679e-04 - tf_psnr: 39.3300 - val_loss: 2.1113e-04 - val_tf_psnr: 36.7718
Epoch 29/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1900e-04 - tf_psnr: 39.2545 - val_loss: 2.1269e-04 - val_tf_psnr: 36.7489
Epoch 30/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1754e-04 - tf_psnr: 39.3062 - val_loss: 2.1435e-04 - val_tf_psnr: 36.7045
Epoch 31/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1873e-04 - tf_psnr: 39.2663 - val_loss: 2.1000e-04 - val_tf_psnr: 36.8053
Epoch 32/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1511e-04 - tf_psnr: 39.3936 - val_loss: 2.0835e-04 - val_tf_psnr: 36.8310
Epoch 33/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1734e-04 - tf_psnr: 39.3166 - val_loss: 2.1313e-04 - val_tf_psnr: 36.7339
Epoch 34/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1485e-04 - tf_psnr: 39.4049 - val_loss: 2.1582e-04 - val_tf_psnr: 36.6812
Epoch 35/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1615e-04 - tf_psnr: 39.3587 - val_loss: 2.1739e-04 - val_tf_psnr: 36.6434
Epoch 36/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1657e-04 - tf_psnr: 39.3396 - val_loss: 2.1378e-04 - val_tf_psnr: 36.7323
Epoch 37/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1643e-04 - tf_psnr: 39.3465 - val_loss: 2.0981e-04 - val_tf_psnr: 36.8055
Epoch 38/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1833e-04 - tf_psnr: 39.2899 - val_loss: 2.1199e-04 - val_tf_psnr: 36.7585
Epoch 39/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1275e-04 - tf_psnr: 39.4834 - val_loss: 2.1071e-04 - val_tf_psnr: 36.7897
Epoch 40/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1306e-04 - tf_psnr: 39.4719 - val_loss: 2.1264e-04 - val_tf_psnr: 36.7447
Epoch 41/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1569e-04 - tf_psnr: 39.3804 - val_loss: 2.2790e-04 - val_tf_psnr: 36.4299
Epoch 42/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1390e-04 - tf_psnr: 39.4399 - val_loss: 2.1401e-04 - val_tf_psnr: 36.7133
Epoch 43/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1548e-04 - tf_psnr: 39.3818 - val_loss: 2.1844e-04 - val_tf_psnr: 36.6246
Epoch 44/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1312e-04 - tf_psnr: 39.4694 - val_loss: 2.1196e-04 - val_tf_psnr: 36.7631
Epoch 45/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1269e-04 - tf_psnr: 39.4858 - val_loss: 2.1217e-04 - val_tf_psnr: 36.7545
Epoch 46/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1245e-04 - tf_psnr: 39.4969 - val_loss: 2.1479e-04 - val_tf_psnr: 36.7032
Epoch 47/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1451e-04 - tf_psnr: 39.4173 - val_loss: 2.1528e-04 - val_tf_psnr: 36.6854
Epoch 48/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1187e-04 - tf_psnr: 39.5192 - val_loss: 2.1672e-04 - val_tf_psnr: 36.6579
Epoch 49/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1320e-04 - tf_psnr: 39.4724 - val_loss: 2.1192e-04 - val_tf_psnr: 36.7600
Epoch 50/50
1440/1440 [==============================] - 17s 12ms/step - loss: 1.1527e-04 - tf_psnr: 39.3925 - val_loss: 2.2225e-04 - val_tf_psnr: 36.5460
psnr 36.16387545238426
354/354 [==============================] - 2s 5ms/step
Test loss: 0.00020370286419766687
Test accuracy: 37.04067025480971
```





##### Trained float images



```{python}
1350/1350 [==============================] - 29s 22ms/step - loss: 0.0012 - tf_psnr: 29.5450 - val_loss: 6.9311e-04 - val_tf_psnr: 31.6106
Epoch 2/50
1350/1350 [==============================] - 18s 13ms/step - loss: 6.8025e-04 - tf_psnr: 31.6818 - val_loss: 5.8849e-04 - val_tf_psnr: 32.3262
Epoch 3/50
1350/1350 [==============================] - 18s 13ms/step - loss: 5.8334e-04 - tf_psnr: 32.3481 - val_loss: 5.0349e-04 - val_tf_psnr: 33.0031
Epoch 4/50
1350/1350 [==============================] - 18s 13ms/step - loss: 5.2933e-04 - tf_psnr: 32.7718 - val_loss: 4.7503e-04 - val_tf_psnr: 33.2606
Epoch 5/50
1350/1350 [==============================] - 17s 13ms/step - loss: 4.7174e-04 - tf_psnr: 33.2709 - val_loss: 4.2912e-04 - val_tf_psnr: 33.6957
Epoch 6/50
1350/1350 [==============================] - 18s 13ms/step - loss: 4.3303e-04 - tf_psnr: 33.6447 - val_loss: 3.6366e-04 - val_tf_psnr: 34.4147
Epoch 7/50
1350/1350 [==============================] - 17s 12ms/step - loss: 4.0449e-04 - tf_psnr: 33.9485 - val_loss: 3.3047e-04 - val_tf_psnr: 34.8236
Epoch 8/50
1350/1350 [==============================] - 18s 13ms/step - loss: 3.5842e-04 - tf_psnr: 34.4644 - val_loss: 3.2752e-04 - val_tf_psnr: 34.8545
Epoch 9/50
1350/1350 [==============================] - 18s 13ms/step - loss: 3.3187e-04 - tf_psnr: 34.7955 - val_loss: 2.8402e-04 - val_tf_psnr: 35.4775
Epoch 10/50
1350/1350 [==============================] - 18s 13ms/step - loss: 3.0879e-04 - tf_psnr: 35.1119 - val_loss: 2.8270e-04 - val_tf_psnr: 35.4975
Epoch 11/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.9607e-04 - tf_psnr: 35.2959 - val_loss: 2.6317e-04 - val_tf_psnr: 35.8098
Epoch 12/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.7993e-04 - tf_psnr: 35.5378 - val_loss: 2.5951e-04 - val_tf_psnr: 35.8734
Epoch 13/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.6728e-04 - tf_psnr: 35.7371 - val_loss: 2.4537e-04 - val_tf_psnr: 36.1142
Epoch 14/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.6946e-04 - tf_psnr: 35.7102 - val_loss: 2.5423e-04 - val_tf_psnr: 35.9632
Epoch 15/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.5356e-04 - tf_psnr: 35.9685 - val_loss: 2.3793e-04 - val_tf_psnr: 36.2515
Epoch 16/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.4553e-04 - tf_psnr: 36.1055 - val_loss: 2.3725e-04 - val_tf_psnr: 36.2733
Epoch 17/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.4074e-04 - tf_psnr: 36.1959 - val_loss: 2.3015e-04 - val_tf_psnr: 36.3979
Epoch 18/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.3555e-04 - tf_psnr: 36.2872 - val_loss: 2.4941e-04 - val_tf_psnr: 36.0514
Epoch 19/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.3635e-04 - tf_psnr: 36.2746 - val_loss: 2.2825e-04 - val_tf_psnr: 36.4387
Epoch 20/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.2550e-04 - tf_psnr: 36.4802 - val_loss: 2.6534e-04 - val_tf_psnr: 35.7861
Epoch 21/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.2291e-04 - tf_psnr: 36.5278 - val_loss: 2.2270e-04 - val_tf_psnr: 36.5518
Epoch 22/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.2077e-04 - tf_psnr: 36.5758 - val_loss: 2.1992e-04 - val_tf_psnr: 36.6051
Epoch 23/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.1443e-04 - tf_psnr: 36.6954 - val_loss: 2.2565e-04 - val_tf_psnr: 36.4991
Epoch 24/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.1010e-04 - tf_psnr: 36.7839 - val_loss: 2.4279e-04 - val_tf_psnr: 36.1669
Epoch 25/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.0852e-04 - tf_psnr: 36.8210 - val_loss: 2.1930e-04 - val_tf_psnr: 36.6352
Epoch 26/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.1110e-04 - tf_psnr: 36.7674 - val_loss: 2.4489e-04 - val_tf_psnr: 36.1321
Epoch 27/50
1350/1350 [==============================] - 18s 13ms/step - loss: 2.0412e-04 - tf_psnr: 36.9120 - val_loss: 2.1934e-04 - val_tf_psnr: 36.6192
Epoch 28/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.9410e-04 - tf_psnr: 37.1264 - val_loss: 2.1807e-04 - val_tf_psnr: 36.6542
Epoch 29/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.9612e-04 - tf_psnr: 37.0849 - val_loss: 2.1604e-04 - val_tf_psnr: 36.6912
Epoch 30/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.9307e-04 - tf_psnr: 37.1498 - val_loss: 2.4123e-04 - val_tf_psnr: 36.2039
Epoch 31/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.9573e-04 - tf_psnr: 37.0998 - val_loss: 2.1342e-04 - val_tf_psnr: 36.7604
Epoch 32/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.8933e-04 - tf_psnr: 37.2356 - val_loss: 2.1234e-04 - val_tf_psnr: 36.7830
Epoch 33/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.9240e-04 - tf_psnr: 37.1715 - val_loss: 2.0450e-04 - val_tf_psnr: 36.9297
Epoch 34/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.8272e-04 - tf_psnr: 37.3914 - val_loss: 2.1495e-04 - val_tf_psnr: 36.7183
Epoch 35/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.8423e-04 - tf_psnr: 37.3577 - val_loss: 2.0819e-04 - val_tf_psnr: 36.8683
Epoch 36/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.8081e-04 - tf_psnr: 37.4406 - val_loss: 2.3388e-04 - val_tf_psnr: 36.3473
Epoch 37/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.8335e-04 - tf_psnr: 37.3787 - val_loss: 2.0632e-04 - val_tf_psnr: 36.9163
Epoch 38/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.7598e-04 - tf_psnr: 37.5539 - val_loss: 2.0472e-04 - val_tf_psnr: 36.9507
Epoch 39/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.7844e-04 - tf_psnr: 37.4962 - val_loss: 2.0413e-04 - val_tf_psnr: 36.9644
Epoch 40/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.7511e-04 - tf_psnr: 37.5736 - val_loss: 2.1379e-04 - val_tf_psnr: 36.7591
Epoch 41/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.7835e-04 - tf_psnr: 37.4996 - val_loss: 2.0489e-04 - val_tf_psnr: 36.9650
Epoch 42/50
1350/1350 [==============================] - 17s 13ms/step - loss: 1.7590e-04 - tf_psnr: 37.5609 - val_loss: 2.0922e-04 - val_tf_psnr: 36.8491
Epoch 43/50
1350/1350 [==============================] - 17s 12ms/step - loss: 1.7170e-04 - tf_psnr: 37.6614 - val_loss: 2.0576e-04 - val_tf_psnr: 36.9469
Epoch 44/50
1350/1350 [==============================] - 17s 12ms/step - loss: 1.6984e-04 - tf_psnr: 37.7078 - val_loss: 2.0076e-04 - val_tf_psnr: 37.0371
Epoch 45/50
1350/1350 [==============================] - 17s 12ms/step - loss: 1.6956e-04 - tf_psnr: 37.7129 - val_loss: 2.0133e-04 - val_tf_psnr: 37.0291
Epoch 46/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.6571e-04 - tf_psnr: 37.8155 - val_loss: 2.1692e-04 - val_tf_psnr: 36.7165
Epoch 47/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.6757e-04 - tf_psnr: 37.7682 - val_loss: 2.1020e-04 - val_tf_psnr: 36.8542
Epoch 48/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.6205e-04 - tf_psnr: 37.9075 - val_loss: 2.0192e-04 - val_tf_psnr: 37.0212
Epoch 49/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.6630e-04 - tf_psnr: 37.7988 - val_loss: 1.9630e-04 - val_tf_psnr: 37.1473
Epoch 50/50
1350/1350 [==============================] - 18s 13ms/step - loss: 1.6020e-04 - tf_psnr: 37.9664 - val_loss: 2.0186e-04 - val_tf_psnr: 37.0369
psnr 36.96107470788776
10/10 [==============================] - 0s 4ms/step
Test loss: 0.00015911785885691643
Test accuracy: 37.982810974121094
 

```











##### trained png images



```{python}
Epoch 1/100
2019-08-22 01:40:52.487628: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
2019-08-22 01:40:52.543083: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2019-08-22 01:40:52.547940: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56467350eaf0 executing computations on platform Host. Devices:
2019-08-22 01:40:52.548040: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-08-22 01:40:52.566784: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcuda.so.1
2019-08-22 01:40:53.563618: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 0 with properties: 
name: Quadro RTX 5000 major: 7 minor: 5 memoryClockRate(GHz): 1.815
pciBusID: 0000:67:00.0
2019-08-22 01:40:53.564478: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 1 with properties: 
name: Quadro RTX 5000 major: 7 minor: 5 memoryClockRate(GHz): 1.815
pciBusID: 0000:68:00.0
2019-08-22 01:40:53.568994: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.1
2019-08-22 01:40:53.614243: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcublas.so.10
2019-08-22 01:40:53.642595: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcufft.so.10
2019-08-22 01:40:53.650065: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcurand.so.10
2019-08-22 01:40:53.700690: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcusolver.so.10
2019-08-22 01:40:53.708451: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcusparse.so.10
2019-08-22 01:40:53.793929: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
2019-08-22 01:40:53.803597: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1763] Adding visible gpu devices: 0, 1
2019-08-22 01:40:53.804386: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.1
2019-08-22 01:40:53.809476: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1181] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-08-22 01:40:53.809547: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1187]      0 1 
2019-08-22 01:40:53.810245: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 0:   N Y 
2019-08-22 01:40:53.810273: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 1:   Y N 
2019-08-22 01:40:53.818527: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15170 MB memory) -> physical GPU (device: 0, name: Quadro RTX 5000, pci bus id: 0000:67:00.0, compute capability: 7.5)
2019-08-22 01:40:53.821714: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 26 MB memory) -> physical GPU (device: 1, name: Quadro RTX 5000, pci bus id: 0000:68:00.0, compute capability: 7.5)
2019-08-22 01:40:53.825061: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564674ca03b0 executing computations on platform CUDA. Devices:
2019-08-22 01:40:53.825097: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Quadro RTX 5000, Compute Capability 7.5
2019-08-22 01:40:53.825111: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (1): Quadro RTX 5000, Compute Capability 7.5
2019-08-22 01:40:58.471132: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
2019-08-22 01:41:04.504589: W tensorflow/core/common_runtime/bfc_allocator.cc:237] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.94GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-08-22 01:41:04.505389: W tensorflow/core/common_runtime/bfc_allocator.cc:237] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.94GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2700/2700 [==============================] - 48s 18ms/step - loss: 0.0064 - tf_psnr: 22.0118 - val_loss: 0.0049 - val_tf_psnr: 23.1323
Epoch 2/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0044 - tf_psnr: 23.5656 - val_loss: 0.0040 - val_tf_psnr: 23.9485
Epoch 3/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0039 - tf_psnr: 24.1127 - val_loss: 0.0038 - val_tf_psnr: 24.2119
Epoch 4/100
2700/2700 [==============================] - 31s 12ms/step - loss: 0.0037 - tf_psnr: 24.3295 - val_loss: 0.0035 - val_tf_psnr: 24.5737
Epoch 5/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0036 - tf_psnr: 24.4459 - val_loss: 0.0033 - val_tf_psnr: 24.7716
Epoch 6/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0034 - tf_psnr: 24.7435 - val_loss: 0.0035 - val_tf_psnr: 24.6063
Epoch 7/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0032 - tf_psnr: 25.0003 - val_loss: 0.0030 - val_tf_psnr: 25.2390
Epoch 8/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0030 - tf_psnr: 25.2242 - val_loss: 0.0029 - val_tf_psnr: 25.3176
Epoch 9/100
2700/2700 [==============================] - 31s 12ms/step - loss: 0.0031 - tf_psnr: 25.1435 - val_loss: 0.0029 - val_tf_psnr: 25.3507
Epoch 10/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0028 - tf_psnr: 25.5369 - val_loss: 0.0027 - val_tf_psnr: 25.7279
Epoch 11/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0027 - tf_psnr: 25.7543 - val_loss: 0.0026 - val_tf_psnr: 25.8985
Epoch 12/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0026 - tf_psnr: 25.8440 - val_loss: 0.0026 - val_tf_psnr: 25.8570
Epoch 13/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0026 - tf_psnr: 25.8594 - val_loss: 0.0025 - val_tf_psnr: 26.0663
Epoch 14/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0026 - tf_psnr: 25.8106 - val_loss: 0.0024 - val_tf_psnr: 26.1181
Epoch 15/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0025 - tf_psnr: 26.1029 - val_loss: 0.0024 - val_tf_psnr: 26.2512
Epoch 16/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0025 - tf_psnr: 26.0934 - val_loss: 0.0024 - val_tf_psnr: 26.2688
Epoch 17/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0025 - tf_psnr: 26.0309 - val_loss: 0.0024 - val_tf_psnr: 26.1729
Epoch 18/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0024 - tf_psnr: 26.2264 - val_loss: 0.0024 - val_tf_psnr: 26.1619
Epoch 19/100
2700/2700 [==============================] - 31s 12ms/step - loss: 0.0024 - tf_psnr: 26.2673 - val_loss: 0.0023 - val_tf_psnr: 26.3117
Epoch 20/100
2700/2700 [==============================] - 30s 11ms/step - loss: 0.0023 - tf_psnr: 26.3559 - val_loss: 0.0023 - val_tf_psnr: 26.4076
Epoch 21/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0023 - tf_psnr: 26.4251 - val_loss: 0.0023 - val_tf_psnr: 26.4182
Epoch 22/100
2700/2700 [==============================] - 31s 12ms/step - loss: 0.0023 - tf_psnr: 26.4529 - val_loss: 0.0023 - val_tf_psnr: 26.4866
Epoch 23/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0022 - tf_psnr: 26.5495 - val_loss: 0.0023 - val_tf_psnr: 26.4757
Epoch 24/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0023 - tf_psnr: 26.4486 - val_loss: 0.0023 - val_tf_psnr: 26.4694
Epoch 25/100
2700/2700 [==============================] - 30s 11ms/step - loss: 0.0023 - tf_psnr: 26.4731 - val_loss: 0.0024 - val_tf_psnr: 26.2541
Epoch 26/100
2700/2700 [==============================] - 30s 11ms/step - loss: 0.0022 - tf_psnr: 26.5764 - val_loss: 0.0023 - val_tf_psnr: 26.4051
Epoch 27/100
2700/2700 [==============================] - 30s 11ms/step - loss: 0.0022 - tf_psnr: 26.5705 - val_loss: 0.0022 - val_tf_psnr: 26.5410
Epoch 28/100
2700/2700 [==============================] - 30s 11ms/step - loss: 0.0022 - tf_psnr: 26.5281 - val_loss: 0.0022 - val_tf_psnr: 26.5433
Epoch 29/100
2700/2700 [==============================] - 30s 11ms/step - loss: 0.0021 - tf_psnr: 26.7041 - val_loss: 0.0022 - val_tf_psnr: 26.5834
Epoch 30/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0022 - tf_psnr: 26.6548 - val_loss: 0.0022 - val_tf_psnr: 26.5221
Epoch 31/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0021 - tf_psnr: 26.8334 - val_loss: 0.0023 - val_tf_psnr: 26.4691
Epoch 32/100
2700/2700 [==============================] - 30s 11ms/step - loss: 0.0021 - tf_psnr: 26.7403 - val_loss: 0.0024 - val_tf_psnr: 26.2582
Epoch 33/100
2700/2700 [==============================] - 30s 11ms/step - loss: 0.0019 - tf_psnr: 27.1384 - val_loss: 0.0023 - val_tf_psnr: 26.4431
Epoch 34/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0022 - tf_psnr: 26.5589 - val_loss: 0.0023 - val_tf_psnr: 26.3949
Epoch 35/100
2700/2700 [==============================] - 31s 12ms/step - loss: 0.0020 - tf_psnr: 27.0532 - val_loss: 0.0021 - val_tf_psnr: 26.7128
Epoch 36/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0020 - tf_psnr: 27.0136 - val_loss: 0.0022 - val_tf_psnr: 26.5635
Epoch 37/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0019 - tf_psnr: 27.3193 - val_loss: 0.0022 - val_tf_psnr: 26.5001
Epoch 38/100
2700/2700 [==============================] - 30s 11ms/step - loss: 0.0019 - tf_psnr: 27.3426 - val_loss: 0.0021 - val_tf_psnr: 26.7769
Epoch 39/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0017 - tf_psnr: 27.7765 - val_loss: 0.0022 - val_tf_psnr: 26.7084
Epoch 40/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0017 - tf_psnr: 27.6518 - val_loss: 0.0025 - val_tf_psnr: 26.1224
Epoch 41/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0016 - tf_psnr: 27.9195 - val_loss: 0.0022 - val_tf_psnr: 26.6352
Epoch 42/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0016 - tf_psnr: 28.0706 - val_loss: 0.0022 - val_tf_psnr: 26.5463
Epoch 43/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0015 - tf_psnr: 28.2384 - val_loss: 0.0020 - val_tf_psnr: 26.9653
Epoch 44/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0015 - tf_psnr: 28.2041 - val_loss: 0.0021 - val_tf_psnr: 26.7669
Epoch 45/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0014 - tf_psnr: 28.5764 - val_loss: 0.0021 - val_tf_psnr: 26.8642
Epoch 46/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0013 - tf_psnr: 28.8400 - val_loss: 0.0021 - val_tf_psnr: 26.8953
Epoch 47/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0013 - tf_psnr: 28.8467 - val_loss: 0.0025 - val_tf_psnr: 25.9712
Epoch 48/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0013 - tf_psnr: 28.7308 - val_loss: 0.0021 - val_tf_psnr: 26.8608
Epoch 49/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0012 - tf_psnr: 29.0588 - val_loss: 0.0021 - val_tf_psnr: 26.9193
Epoch 50/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0013 - tf_psnr: 29.0412 - val_loss: 0.0023 - val_tf_psnr: 26.4970
Epoch 51/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0012 - tf_psnr: 29.1216 - val_loss: 0.0020 - val_tf_psnr: 27.0407
Epoch 52/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0012 - tf_psnr: 29.3167 - val_loss: 0.0021 - val_tf_psnr: 26.7919
Epoch 53/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0011 - tf_psnr: 29.6700 - val_loss: 0.0022 - val_tf_psnr: 26.6840
Epoch 54/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0011 - tf_psnr: 29.6593 - val_loss: 0.0023 - val_tf_psnr: 26.4680
Epoch 55/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0011 - tf_psnr: 29.7550 - val_loss: 0.0021 - val_tf_psnr: 26.7731
Epoch 56/100
2700/2700 [==============================] - 30s 11ms/step - loss: 0.0010 - tf_psnr: 29.8264 - val_loss: 0.0021 - val_tf_psnr: 26.8239
Epoch 57/100
2700/2700 [==============================] - 31s 11ms/step - loss: 0.0010 - tf_psnr: 29.9547 - val_loss: 0.0021 - val_tf_psnr: 26.7650
Epoch 58/100
2700/2700 [==============================] - 31s 11ms/step - loss: 9.9302e-04 - tf_psnr: 30.0384 - val_loss: 0.0021 - val_tf_psnr: 26.8227
Epoch 59/100
2700/2700 [==============================] - 31s 11ms/step - loss: 9.6026e-04 - tf_psnr: 30.1797 - val_loss: 0.0022 - val_tf_psnr: 26.6250
Epoch 60/100
2700/2700 [==============================] - 31s 11ms/step - loss: 9.6869e-04 - tf_psnr: 30.1445 - val_loss: 0.0022 - val_tf_psnr: 26.6995
Epoch 61/100
2700/2700 [==============================] - 31s 11ms/step - loss: 9.4573e-04 - tf_psnr: 30.2468 - val_loss: 0.0020 - val_tf_psnr: 26.9710
Epoch 62/100
2700/2700 [==============================] - 31s 11ms/step - loss: 9.6182e-04 - tf_psnr: 30.1810 - val_loss: 0.0021 - val_tf_psnr: 26.7425
Epoch 63/100
2700/2700 [==============================] - 31s 11ms/step - loss: 9.1989e-04 - tf_psnr: 30.3717 - val_loss: 0.0021 - val_tf_psnr: 26.8452
Epoch 64/100
2700/2700 [==============================] - 31s 11ms/step - loss: 9.0354e-04 - tf_psnr: 30.4459 - val_loss: 0.0020 - val_tf_psnr: 26.9479
Epoch 65/100
2700/2700 [==============================] - 31s 11ms/step - loss: 9.0101e-04 - tf_psnr: 30.4583 - val_loss: 0.0021 - val_tf_psnr: 26.7685
Epoch 66/100
2700/2700 [==============================] - 31s 11ms/step - loss: 8.9369e-04 - tf_psnr: 30.4943 - val_loss: 0.0021 - val_tf_psnr: 26.7915
Epoch 67/100
2700/2700 [==============================] - 31s 11ms/step - loss: 8.9971e-04 - tf_psnr: 30.4673 - val_loss: 0.0021 - val_tf_psnr: 26.8177
Epoch 68/100
2700/2700 [==============================] - 31s 11ms/step - loss: 8.3728e-04 - tf_psnr: 30.7748 - val_loss: 0.0020 - val_tf_psnr: 26.9311
Epoch 69/100
2700/2700 [==============================] - 31s 11ms/step - loss: 8.3305e-04 - tf_psnr: 30.7953 - val_loss: 0.0020 - val_tf_psnr: 26.9406
Epoch 70/100
2700/2700 [==============================] - 31s 11ms/step - loss: 8.3845e-04 - tf_psnr: 30.7697 - val_loss: 0.0021 - val_tf_psnr: 26.8968
Epoch 71/100
2700/2700 [==============================] - 31s 11ms/step - loss: 8.2143e-04 - tf_psnr: 30.8589 - val_loss: 0.0021 - val_tf_psnr: 26.7859
Epoch 72/100
2700/2700 [==============================] - 30s 11ms/step - loss: 7.8939e-04 - tf_psnr: 31.0297 - val_loss: 0.0021 - val_tf_psnr: 26.8237
Epoch 73/100
2700/2700 [==============================] - 31s 11ms/step - loss: 8.2659e-04 - tf_psnr: 30.8371 - val_loss: 0.0021 - val_tf_psnr: 26.8848
Epoch 74/100
2700/2700 [==============================] - 30s 11ms/step - loss: 8.0248e-04 - tf_psnr: 30.9600 - val_loss: 0.0021 - val_tf_psnr: 26.8950
Epoch 75/100
2700/2700 [==============================] - 31s 12ms/step - loss: 8.3818e-04 - tf_psnr: 30.7748 - val_loss: 0.0021 - val_tf_psnr: 26.8816
Epoch 76/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.9555e-04 - tf_psnr: 30.9961 - val_loss: 0.0021 - val_tf_psnr: 26.8834
Epoch 77/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.6054e-04 - tf_psnr: 31.1912 - val_loss: 0.0021 - val_tf_psnr: 26.7851
Epoch 78/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.8768e-04 - tf_psnr: 31.0464 - val_loss: 0.0021 - val_tf_psnr: 26.8443
Epoch 79/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.5366e-04 - tf_psnr: 31.2311 - val_loss: 0.0021 - val_tf_psnr: 26.7792
Epoch 80/100
2700/2700 [==============================] - 31s 12ms/step - loss: 7.5715e-04 - tf_psnr: 31.2124 - val_loss: 0.0021 - val_tf_psnr: 26.8229
Epoch 81/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.3435e-04 - tf_psnr: 31.3451 - val_loss: 0.0021 - val_tf_psnr: 26.7221
Epoch 82/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.5901e-04 - tf_psnr: 31.2029 - val_loss: 0.0021 - val_tf_psnr: 26.8736
Epoch 83/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.3917e-04 - tf_psnr: 31.3188 - val_loss: 0.0021 - val_tf_psnr: 26.8991
Epoch 84/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.6220e-04 - tf_psnr: 31.1868 - val_loss: 0.0020 - val_tf_psnr: 26.9088
Epoch 85/100
2700/2700 [==============================] - 32s 12ms/step - loss: 7.3385e-04 - tf_psnr: 31.3496 - val_loss: 0.0020 - val_tf_psnr: 27.0047
Epoch 86/100
2700/2700 [==============================] - 31s 12ms/step - loss: 7.4628e-04 - tf_psnr: 31.2754 - val_loss: 0.0022 - val_tf_psnr: 26.6472
Epoch 87/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.3709e-04 - tf_psnr: 31.3335 - val_loss: 0.0021 - val_tf_psnr: 26.7758
Epoch 88/100
2700/2700 [==============================] - 32s 12ms/step - loss: 7.7044e-04 - tf_psnr: 31.1417 - val_loss: 0.0020 - val_tf_psnr: 26.9855
Epoch 89/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.1064e-04 - tf_psnr: 31.4860 - val_loss: 0.0021 - val_tf_psnr: 26.8372
Epoch 90/100
2700/2700 [==============================] - 31s 11ms/step - loss: 6.9347e-04 - tf_psnr: 31.5913 - val_loss: 0.0021 - val_tf_psnr: 26.8810
Epoch 91/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.2104e-04 - tf_psnr: 31.4267 - val_loss: 0.0021 - val_tf_psnr: 26.9048
Epoch 92/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.0965e-04 - tf_psnr: 31.4929 - val_loss: 0.0021 - val_tf_psnr: 26.9068
Epoch 93/100
2700/2700 [==============================] - 31s 11ms/step - loss: 6.9115e-04 - tf_psnr: 31.6068 - val_loss: 0.0021 - val_tf_psnr: 26.8183
Epoch 94/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.0160e-04 - tf_psnr: 31.5455 - val_loss: 0.0021 - val_tf_psnr: 26.7418
Epoch 95/100
2700/2700 [==============================] - 31s 11ms/step - loss: 6.8787e-04 - tf_psnr: 31.6282 - val_loss: 0.0021 - val_tf_psnr: 26.9095
Epoch 96/100
2700/2700 [==============================] - 31s 11ms/step - loss: 6.7453e-04 - tf_psnr: 31.7115 - val_loss: 0.0021 - val_tf_psnr: 26.9154
Epoch 97/100
2700/2700 [==============================] - 31s 11ms/step - loss: 6.7016e-04 - tf_psnr: 31.7412 - val_loss: 0.0021 - val_tf_psnr: 26.7336
Epoch 98/100
2700/2700 [==============================] - 31s 11ms/step - loss: 7.4785e-04 - tf_psnr: 31.2750 - val_loss: 0.0021 - val_tf_psnr: 26.8674
Epoch 99/100
2700/2700 [==============================] - 32s 12ms/step - loss: 7.0351e-04 - tf_psnr: 31.5305 - val_loss: 0.0021 - val_tf_psnr: 26.7554
Epoch 100/100
2700/2700 [==============================] - 31s 12ms/step - loss: 6.7164e-04 - tf_psnr: 31.7309 - val_loss: 0.0021 - val_tf_psnr: 26.7695
psnr 29.92060412603745
100/100 [==============================] - 0s 4ms/step
Test loss: 0.0010184496734291315
Test accuracy: 29.962328872680665


```



##### trained float images



```{python}

```

