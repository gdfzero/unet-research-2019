##### UNet-keras3

```{python}
(base) [npovey@ka ~]$ conda create -n keras-gpu python=3.6 numpy scipy keras-gpu
(base) [npovey@ka unet4]$ conda activate keras-gpu
(keras-gpu) [npovey@ka unet4]$ pip install pandas
(keras-gpu) [npovey@ka unet4]$ pip install Pillow
(keras-gpu) [npovey@ka unet4]$ pip install matplotlib
(keras-gpu) [npovey@ka unet4]$ python3 unet_keras3.py 
```

##### other useful commands

```{python}
export CUDA_VISIBLE_DEVICES=0
#view tensorboard
tensorboard --logdir=./logs
http://0.0.0.0:6006
```



##### Unet-keras3 results

| Low Dose Image | DnCnn                               | Unet-keras3                       |
| -------------- | ----------------------------------- | --------------------------------- |
| sparseview_60  | Avg PSNR: 32.30    Avg SSIM: 0.8560 | psnr 50 epochs: 37.71519761116282 |
| sparseview_90  | Avg PSNR: 35.13    Avg SSIM: 0.8892 | psnr 50 epochs: 39.68172444470224 |
| sparseview_180 | Avg PSNR: 39.77    Avg SSIM: 0.9341 | psnr 50 epochs:  42.8100640160669 |
| ldct_1e5       |                                     | psnr 50 epochs: 45.1429232130602  |
| ldct_2e5       |                                     |                                   |
| ldct_7e4       |                                     |                                   |



##### outputs sparseview_60

```{python}
2019-08-25 18:39:54.554071: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
3240/3240 [==============================] - 52s 16ms/step - loss: 9.4390e-04 - tf_psnr: 30.5203 - val_loss: 5.7822e-04 - val_tf_psnr: 32.4536
Epoch 2/50
3240/3240 [==============================] - 41s 13ms/step - loss: 5.8787e-04 - tf_psnr: 32.3280 - val_loss: 4.7184e-04 - val_tf_psnr: 33.3339
Epoch 3/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.6633e-04 - tf_psnr: 33.3321 - val_loss: 3.9146e-04 - val_tf_psnr: 34.1417
Epoch 4/50
3240/3240 [==============================] - 41s 13ms/step - loss: 3.7309e-04 - tf_psnr: 34.2997 - val_loss: 3.2473e-04 - val_tf_psnr: 34.9465
Epoch 5/50
3240/3240 [==============================] - 41s 13ms/step - loss: 3.1103e-04 - tf_psnr: 35.0825 - val_loss: 2.8455e-04 - val_tf_psnr: 35.5264
Epoch 6/50
3240/3240 [==============================] - 41s 13ms/step - loss: 2.7822e-04 - tf_psnr: 35.5657 - val_loss: 2.6219e-04 - val_tf_psnr: 35.8855
Epoch 7/50
3240/3240 [==============================] - 41s 13ms/step - loss: 2.5419e-04 - tf_psnr: 35.9598 - val_loss: 2.6101e-04 - val_tf_psnr: 35.8971
Epoch 8/50
3240/3240 [==============================] - 40s 12ms/step - loss: 2.4077e-04 - tf_psnr: 36.1936 - val_loss: 2.4057e-04 - val_tf_psnr: 36.2658
Epoch 9/50
3240/3240 [==============================] - 39s 12ms/step - loss: 2.2576e-04 - tf_psnr: 36.4727 - val_loss: 2.2849e-04 - val_tf_psnr: 36.4927
Epoch 10/50
3240/3240 [==============================] - 41s 13ms/step - loss: 2.1948e-04 - tf_psnr: 36.6029 - val_loss: 2.2882e-04 - val_tf_psnr: 36.4760
Epoch 11/50
3240/3240 [==============================] - 41s 13ms/step - loss: 2.0689e-04 - tf_psnr: 36.8498 - val_loss: 2.1463e-04 - val_tf_psnr: 36.7526
Epoch 12/50
3240/3240 [==============================] - 41s 13ms/step - loss: 2.0292e-04 - tf_psnr: 36.9372 - val_loss: 2.1834e-04 - val_tf_psnr: 36.6779
Epoch 13/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.9724e-04 - tf_psnr: 37.0597 - val_loss: 2.0863e-04 - val_tf_psnr: 36.8867
Epoch 14/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.9172e-04 - tf_psnr: 37.1846 - val_loss: 2.1297e-04 - val_tf_psnr: 36.7925
Epoch 15/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.8427e-04 - tf_psnr: 37.3541 - val_loss: 2.0051e-04 - val_tf_psnr: 37.0525
Epoch 16/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.8294e-04 - tf_psnr: 37.3885 - val_loss: 2.0260e-04 - val_tf_psnr: 37.0108
Epoch 17/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.7911e-04 - tf_psnr: 37.4796 - val_loss: 2.0820e-04 - val_tf_psnr: 36.8853
Epoch 18/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.8052e-04 - tf_psnr: 37.4496 - val_loss: 1.9947e-04 - val_tf_psnr: 37.0852
Epoch 19/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.6963e-04 - tf_psnr: 37.7113 - val_loss: 2.0080e-04 - val_tf_psnr: 37.0492
Epoch 20/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.6897e-04 - tf_psnr: 37.7302 - val_loss: 2.0219e-04 - val_tf_psnr: 37.0246
Epoch 21/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.6638e-04 - tf_psnr: 37.7973 - val_loss: 1.9143e-04 - val_tf_psnr: 37.2610
Epoch 22/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.6579e-04 - tf_psnr: 37.8162 - val_loss: 1.9437e-04 - val_tf_psnr: 37.1936
Epoch 23/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.6081e-04 - tf_psnr: 37.9444 - val_loss: 1.9381e-04 - val_tf_psnr: 37.2095
Epoch 24/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.6276e-04 - tf_psnr: 37.8967 - val_loss: 1.8655e-04 - val_tf_psnr: 37.3686
Epoch 25/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.5842e-04 - tf_psnr: 38.0112 - val_loss: 1.8806e-04 - val_tf_psnr: 37.3348
Epoch 26/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.5537e-04 - tf_psnr: 38.0931 - val_loss: 2.0602e-04 - val_tf_psnr: 36.9254
Epoch 27/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.5633e-04 - tf_psnr: 38.0727 - val_loss: 1.9422e-04 - val_tf_psnr: 37.1864
Epoch 28/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.5262e-04 - tf_psnr: 38.1744 - val_loss: 1.8584e-04 - val_tf_psnr: 37.3926
Epoch 29/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.5139e-04 - tf_psnr: 38.2074 - val_loss: 1.9586e-04 - val_tf_psnr: 37.1432
Epoch 30/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.5229e-04 - tf_psnr: 38.1882 - val_loss: 1.8527e-04 - val_tf_psnr: 37.4023
Epoch 31/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.4772e-04 - tf_psnr: 38.3133 - val_loss: 1.8545e-04 - val_tf_psnr: 37.4022
Epoch 32/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.4641e-04 - tf_psnr: 38.3513 - val_loss: 1.8412e-04 - val_tf_psnr: 37.4293
Epoch 33/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.4672e-04 - tf_psnr: 38.3449 - val_loss: 1.8407e-04 - val_tf_psnr: 37.4305
Epoch 34/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.4567e-04 - tf_psnr: 38.3755 - val_loss: 1.8728e-04 - val_tf_psnr: 37.3453
Epoch 35/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.4269e-04 - tf_psnr: 38.4642 - val_loss: 1.8413e-04 - val_tf_psnr: 37.4263
Epoch 36/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.4310e-04 - tf_psnr: 38.4552 - val_loss: 1.7929e-04 - val_tf_psnr: 37.5360
Epoch 37/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.4179e-04 - tf_psnr: 38.4910 - val_loss: 1.8199e-04 - val_tf_psnr: 37.4832
Epoch 38/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.4152e-04 - tf_psnr: 38.4996 - val_loss: 1.8123e-04 - val_tf_psnr: 37.4871
Epoch 39/50
3240/3240 [==============================] - 39s 12ms/step - loss: 1.4065e-04 - tf_psnr: 38.5298 - val_loss: 1.9307e-04 - val_tf_psnr: 37.2106
Epoch 40/50
3240/3240 [==============================] - 39s 12ms/step - loss: 1.3875e-04 - tf_psnr: 38.5865 - val_loss: 1.7919e-04 - val_tf_psnr: 37.5488
Epoch 41/50
3240/3240 [==============================] - 38s 12ms/step - loss: 1.3714e-04 - tf_psnr: 38.6353 - val_loss: 1.8026e-04 - val_tf_psnr: 37.5214
Epoch 42/50
3240/3240 [==============================] - 40s 12ms/step - loss: 1.3748e-04 - tf_psnr: 38.6260 - val_loss: 1.8219e-04 - val_tf_psnr: 37.4725
Epoch 43/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.3538e-04 - tf_psnr: 38.6915 - val_loss: 1.8085e-04 - val_tf_psnr: 37.4992
Epoch 44/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.3541e-04 - tf_psnr: 38.6921 - val_loss: 1.7797e-04 - val_tf_psnr: 37.5801
Epoch 45/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.3577e-04 - tf_psnr: 38.6831 - val_loss: 1.7907e-04 - val_tf_psnr: 37.5507
Epoch 46/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.3290e-04 - tf_psnr: 38.7735 - val_loss: 1.9127e-04 - val_tf_psnr: 37.2541
Epoch 47/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.3455e-04 - tf_psnr: 38.7205 - val_loss: 1.7708e-04 - val_tf_psnr: 37.6078
Epoch 48/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.3237e-04 - tf_psnr: 38.7890 - val_loss: 1.7757e-04 - val_tf_psnr: 37.5923
Epoch 49/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.3248e-04 - tf_psnr: 38.7861 - val_loss: 1.7757e-04 - val_tf_psnr: 37.5859
Epoch 50/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.3012e-04 - tf_psnr: 38.8652 - val_loss: 1.7748e-04 - val_tf_psnr: 37.5884
psnr 50 epochs..... 37.71519761116282
354/354 [==============================] - 2s 5ms/step
Test loss: 0.00014251642549702488
Test accuracy: 38.49687442671781

```

##### Sparseview_90

```{python}
2019-08-25 19:24:14.995115: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
3240/3240 [==============================] - 48s 15ms/step - loss: 5.0589e-04 - tf_psnr: 33.1099 - val_loss: 3.3912e-04 - val_tf_psnr: 34.7722
Epoch 2/50
3240/3240 [==============================] - 41s 13ms/step - loss: 3.3080e-04 - tf_psnr: 34.8244 - val_loss: 2.6423e-04 - val_tf_psnr: 35.8497
Epoch 3/50
3240/3240 [==============================] - 41s 13ms/step - loss: 2.5759e-04 - tf_psnr: 35.9044 - val_loss: 2.1127e-04 - val_tf_psnr: 36.8342
Epoch 4/50
3240/3240 [==============================] - 41s 13ms/step - loss: 2.1235e-04 - tf_psnr: 36.7411 - val_loss: 1.8537e-04 - val_tf_psnr: 37.4072
Epoch 5/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.8733e-04 - tf_psnr: 37.2856 - val_loss: 1.6729e-04 - val_tf_psnr: 37.8511
Epoch 6/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.6734e-04 - tf_psnr: 37.7741 - val_loss: 1.5316e-04 - val_tf_psnr: 38.2433
Epoch 7/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.5948e-04 - tf_psnr: 37.9892 - val_loss: 1.4893e-04 - val_tf_psnr: 38.3675
Epoch 8/50
3240/3240 [==============================] - 40s 12ms/step - loss: 1.4931e-04 - tf_psnr: 38.2693 - val_loss: 1.4429e-04 - val_tf_psnr: 38.4915
Epoch 9/50
3240/3240 [==============================] - 40s 12ms/step - loss: 1.4129e-04 - tf_psnr: 38.5097 - val_loss: 1.3687e-04 - val_tf_psnr: 38.7248
Epoch 10/50
3240/3240 [==============================] - 39s 12ms/step - loss: 1.3763e-04 - tf_psnr: 38.6242 - val_loss: 1.3600e-04 - val_tf_psnr: 38.7505
Epoch 11/50
3240/3240 [==============================] - 40s 12ms/step - loss: 1.3280e-04 - tf_psnr: 38.7798 - val_loss: 1.3006e-04 - val_tf_psnr: 38.9485
Epoch 12/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.2978e-04 - tf_psnr: 38.8771 - val_loss: 1.3726e-04 - val_tf_psnr: 38.6950
Epoch 13/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.2578e-04 - tf_psnr: 39.0124 - val_loss: 1.3296e-04 - val_tf_psnr: 38.8519
Epoch 14/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.2238e-04 - tf_psnr: 39.1321 - val_loss: 1.2427e-04 - val_tf_psnr: 39.1403
Epoch 15/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.2231e-04 - tf_psnr: 39.1394 - val_loss: 1.3053e-04 - val_tf_psnr: 38.9203
Epoch 16/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.1847e-04 - tf_psnr: 39.2739 - val_loss: 1.2711e-04 - val_tf_psnr: 39.0477
Epoch 17/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.1600e-04 - tf_psnr: 39.3651 - val_loss: 1.2351e-04 - val_tf_psnr: 39.1724
Epoch 18/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.1384e-04 - tf_psnr: 39.4461 - val_loss: 1.2041e-04 - val_tf_psnr: 39.2835
Epoch 19/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.1477e-04 - tf_psnr: 39.4226 - val_loss: 1.1911e-04 - val_tf_psnr: 39.3215
Epoch 20/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.0955e-04 - tf_psnr: 39.6131 - val_loss: 1.1524e-04 - val_tf_psnr: 39.4691
Epoch 21/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.0863e-04 - tf_psnr: 39.6497 - val_loss: 1.1673e-04 - val_tf_psnr: 39.4157
Epoch 22/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.0715e-04 - tf_psnr: 39.7097 - val_loss: 1.1742e-04 - val_tf_psnr: 39.3848
Epoch 23/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.0680e-04 - tf_psnr: 39.7237 - val_loss: 1.1782e-04 - val_tf_psnr: 39.3632
Epoch 24/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.0394e-04 - tf_psnr: 39.8406 - val_loss: 1.1459e-04 - val_tf_psnr: 39.4973
Epoch 25/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.0332e-04 - tf_psnr: 39.8672 - val_loss: 1.1304e-04 - val_tf_psnr: 39.5537
Epoch 26/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.0321e-04 - tf_psnr: 39.8741 - val_loss: 1.1449e-04 - val_tf_psnr: 39.4928
Epoch 27/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.0075e-04 - tf_psnr: 39.9759 - val_loss: 1.1089e-04 - val_tf_psnr: 39.6386
Epoch 28/50
3240/3240 [==============================] - 42s 13ms/step - loss: 1.0027e-04 - tf_psnr: 39.9987 - val_loss: 1.1359e-04 - val_tf_psnr: 39.5323
Epoch 29/50
3240/3240 [==============================] - 41s 13ms/step - loss: 9.9717e-05 - tf_psnr: 40.0229 - val_loss: 1.1084e-04 - val_tf_psnr: 39.6428
Epoch 30/50
3240/3240 [==============================] - 42s 13ms/step - loss: 9.7935e-05 - tf_psnr: 40.1007 - val_loss: 1.1117e-04 - val_tf_psnr: 39.6304
Epoch 31/50
3240/3240 [==============================] - 41s 13ms/step - loss: 9.7548e-05 - tf_psnr: 40.1178 - val_loss: 1.0884e-04 - val_tf_psnr: 39.7247
Epoch 32/50
3240/3240 [==============================] - 42s 13ms/step - loss: 9.6987e-05 - tf_psnr: 40.1454 - val_loss: 1.1330e-04 - val_tf_psnr: 39.5520
Epoch 33/50
3240/3240 [==============================] - 42s 13ms/step - loss: 9.5434e-05 - tf_psnr: 40.2119 - val_loss: 1.1096e-04 - val_tf_psnr: 39.6396
Epoch 34/50
3240/3240 [==============================] - 42s 13ms/step - loss: 9.4679e-05 - tf_psnr: 40.2439 - val_loss: 1.0731e-04 - val_tf_psnr: 39.7931
Epoch 35/50
3240/3240 [==============================] - 42s 13ms/step - loss: 9.4573e-05 - tf_psnr: 40.2537 - val_loss: 1.0890e-04 - val_tf_psnr: 39.7198
Epoch 36/50
3240/3240 [==============================] - 41s 13ms/step - loss: 9.3548e-05 - tf_psnr: 40.2977 - val_loss: 1.0695e-04 - val_tf_psnr: 39.8018
Epoch 37/50
3240/3240 [==============================] - 42s 13ms/step - loss: 9.2701e-05 - tf_psnr: 40.3397 - val_loss: 1.0708e-04 - val_tf_psnr: 39.7987
Epoch 38/50
3240/3240 [==============================] - 42s 13ms/step - loss: 9.2376e-05 - tf_psnr: 40.3561 - val_loss: 1.0587e-04 - val_tf_psnr: 39.8527
Epoch 39/50
3240/3240 [==============================] - 41s 13ms/step - loss: 9.1537e-05 - tf_psnr: 40.3926 - val_loss: 1.0598e-04 - val_tf_psnr: 39.8424
Epoch 40/50
3240/3240 [==============================] - 40s 12ms/step - loss: 9.1146e-05 - tf_psnr: 40.4115 - val_loss: 1.0655e-04 - val_tf_psnr: 39.8172
Epoch 41/50
3240/3240 [==============================] - 39s 12ms/step - loss: 9.0193e-05 - tf_psnr: 40.4562 - val_loss: 1.0445e-04 - val_tf_psnr: 39.9070
Epoch 42/50
3240/3240 [==============================] - 40s 12ms/step - loss: 8.9794e-05 - tf_psnr: 40.4749 - val_loss: 1.1051e-04 - val_tf_psnr: 39.6513
Epoch 43/50
3240/3240 [==============================] - 40s 12ms/step - loss: 9.0376e-05 - tf_psnr: 40.4525 - val_loss: 1.0782e-04 - val_tf_psnr: 39.7711
Epoch 44/50
3240/3240 [==============================] - 41s 13ms/step - loss: 8.8629e-05 - tf_psnr: 40.5326 - val_loss: 1.0386e-04 - val_tf_psnr: 39.9357
Epoch 45/50
3240/3240 [==============================] - 41s 13ms/step - loss: 8.8580e-05 - tf_psnr: 40.5355 - val_loss: 1.1768e-04 - val_tf_psnr: 39.3816
Epoch 46/50
3240/3240 [==============================] - 41s 13ms/step - loss: 8.8494e-05 - tf_psnr: 40.5415 - val_loss: 1.0638e-04 - val_tf_psnr: 39.8262
Epoch 47/50
3240/3240 [==============================] - 41s 13ms/step - loss: 8.7659e-05 - tf_psnr: 40.5802 - val_loss: 1.0344e-04 - val_tf_psnr: 39.9559
Epoch 48/50
3240/3240 [==============================] - 41s 13ms/step - loss: 8.6965e-05 - tf_psnr: 40.6163 - val_loss: 1.0316e-04 - val_tf_psnr: 39.9677
Epoch 49/50
3240/3240 [==============================] - 41s 13ms/step - loss: 8.6648e-05 - tf_psnr: 40.6327 - val_loss: 1.0342e-04 - val_tf_psnr: 39.9584
Epoch 50/50
3240/3240 [==============================] - 41s 13ms/step - loss: 8.6395e-05 - tf_psnr: 40.6457 - val_loss: 1.0259e-04 - val_tf_psnr: 39.9919
psnr 50 epochs..... 39.68172444470224
354/354 [==============================] - 2s 5ms/step
Test loss: 9.061753574977786e-05
Test accuracy: 40.44993102079057
```

Sparseview_180

```{python}
2019-08-25 20:35:58.831855: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
3240/3240 [==============================] - 48s 15ms/step - loss: 1.5503e-04 - tf_psnr: 38.1919 - val_loss: 1.0005e-04 - val_tf_psnr: 40.1549
Epoch 2/50
3240/3240 [==============================] - 41s 13ms/step - loss: 1.0387e-04 - tf_psnr: 39.8518 - val_loss: 8.2221e-05 - val_tf_psnr: 40.9875
Epoch 3/50
3240/3240 [==============================] - 41s 13ms/step - loss: 8.9133e-05 - tf_psnr: 40.5118 - val_loss: 7.2508e-05 - val_tf_psnr: 41.5279
Epoch 4/50
3240/3240 [==============================] - 41s 13ms/step - loss: 7.9554e-05 - tf_psnr: 41.0079 - val_loss: 6.6144e-05 - val_tf_psnr: 41.9250
Epoch 5/50
3240/3240 [==============================] - 41s 13ms/step - loss: 7.1710e-05 - tf_psnr: 41.4535 - val_loss: 6.1281e-05 - val_tf_psnr: 42.2532
Epoch 6/50
3240/3240 [==============================] - 41s 13ms/step - loss: 6.6708e-05 - tf_psnr: 41.7711 - val_loss: 5.7203e-05 - val_tf_psnr: 42.5452
Epoch 7/50
3240/3240 [==============================] - 40s 12ms/step - loss: 6.3140e-05 - tf_psnr: 42.0109 - val_loss: 5.5876e-05 - val_tf_psnr: 42.6458
Epoch 8/50
3240/3240 [==============================] - 39s 12ms/step - loss: 6.0209e-05 - tf_psnr: 42.2152 - val_loss: 5.2737e-05 - val_tf_psnr: 42.9163
Epoch 9/50
3240/3240 [==============================] - 40s 12ms/step - loss: 5.8438e-05 - tf_psnr: 42.3431 - val_loss: 5.1769e-05 - val_tf_psnr: 42.9932
Epoch 10/50
3240/3240 [==============================] - 41s 13ms/step - loss: 5.6614e-05 - tf_psnr: 42.4822 - val_loss: 5.0255e-05 - val_tf_psnr: 43.1160
Epoch 11/50
3240/3240 [==============================] - 41s 13ms/step - loss: 5.5052e-05 - tf_psnr: 42.6036 - val_loss: 4.9429e-05 - val_tf_psnr: 43.1974
Epoch 12/50
3240/3240 [==============================] - 41s 13ms/step - loss: 5.3812e-05 - tf_psnr: 42.7056 - val_loss: 4.8199e-05 - val_tf_psnr: 43.3089
Epoch 13/50
3240/3240 [==============================] - 41s 13ms/step - loss: 5.3245e-05 - tf_psnr: 42.7502 - val_loss: 4.7154e-05 - val_tf_psnr: 43.4003
Epoch 14/50
3240/3240 [==============================] - 41s 13ms/step - loss: 5.1259e-05 - tf_psnr: 42.9140 - val_loss: 4.6055e-05 - val_tf_psnr: 43.5040
Epoch 15/50
3240/3240 [==============================] - 41s 13ms/step - loss: 5.0890e-05 - tf_psnr: 42.9456 - val_loss: 4.6094e-05 - val_tf_psnr: 43.4980
Epoch 16/50
3240/3240 [==============================] - 41s 13ms/step - loss: 5.0412e-05 - tf_psnr: 42.9894 - val_loss: 4.4794e-05 - val_tf_psnr: 43.6308
Epoch 17/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.9490e-05 - tf_psnr: 43.0684 - val_loss: 4.5213e-05 - val_tf_psnr: 43.5813
Epoch 18/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.8892e-05 - tf_psnr: 43.1206 - val_loss: 4.3950e-05 - val_tf_psnr: 43.7076
Epoch 19/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.8366e-05 - tf_psnr: 43.1702 - val_loss: 4.3425e-05 - val_tf_psnr: 43.7652
Epoch 20/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.7654e-05 - tf_psnr: 43.2350 - val_loss: 4.3574e-05 - val_tf_psnr: 43.7455
Epoch 21/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.7025e-05 - tf_psnr: 43.2915 - val_loss: 4.4836e-05 - val_tf_psnr: 43.6119
Epoch 22/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.6655e-05 - tf_psnr: 43.3212 - val_loss: 4.2580e-05 - val_tf_psnr: 43.8483
Epoch 23/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.6018e-05 - tf_psnr: 43.3856 - val_loss: 4.2995e-05 - val_tf_psnr: 43.7953
Epoch 24/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.5681e-05 - tf_psnr: 43.4161 - val_loss: 4.3149e-05 - val_tf_psnr: 43.7726
Epoch 25/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.5121e-05 - tf_psnr: 43.4668 - val_loss: 4.1854e-05 - val_tf_psnr: 43.9154
Epoch 26/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.4836e-05 - tf_psnr: 43.5019 - val_loss: 4.1312e-05 - val_tf_psnr: 43.9716
Epoch 27/50
3240/3240 [==============================] - 42s 13ms/step - loss: 4.4318e-05 - tf_psnr: 43.5519 - val_loss: 4.1101e-05 - val_tf_psnr: 43.9993
Epoch 28/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.4076e-05 - tf_psnr: 43.5738 - val_loss: 4.0311e-05 - val_tf_psnr: 44.0789
Epoch 29/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.3506e-05 - tf_psnr: 43.6282 - val_loss: 4.0050e-05 - val_tf_psnr: 44.1096
Epoch 30/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.3515e-05 - tf_psnr: 43.6306 - val_loss: 4.0198e-05 - val_tf_psnr: 44.0928
Epoch 31/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.2700e-05 - tf_psnr: 43.7086 - val_loss: 4.0509e-05 - val_tf_psnr: 44.0573
Epoch 32/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.2828e-05 - tf_psnr: 43.6965 - val_loss: 3.9816e-05 - val_tf_psnr: 44.1319
Epoch 33/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.2316e-05 - tf_psnr: 43.7466 - val_loss: 3.9615e-05 - val_tf_psnr: 44.1606
Epoch 34/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.2417e-05 - tf_psnr: 43.7416 - val_loss: 3.8896e-05 - val_tf_psnr: 44.2403
Epoch 35/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.1855e-05 - tf_psnr: 43.7980 - val_loss: 3.8860e-05 - val_tf_psnr: 44.2440
Epoch 36/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.1647e-05 - tf_psnr: 43.8206 - val_loss: 3.8656e-05 - val_tf_psnr: 44.2624
Epoch 37/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.1245e-05 - tf_psnr: 43.8647 - val_loss: 3.8586e-05 - val_tf_psnr: 44.2737
Epoch 38/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.1311e-05 - tf_psnr: 43.8588 - val_loss: 3.8358e-05 - val_tf_psnr: 44.2957
Epoch 39/50
3240/3240 [==============================] - 38s 12ms/step - loss: 4.1016e-05 - tf_psnr: 43.8867 - val_loss: 3.9284e-05 - val_tf_psnr: 44.1858
Epoch 40/50
3240/3240 [==============================] - 39s 12ms/step - loss: 4.0741e-05 - tf_psnr: 43.9119 - val_loss: 3.8575e-05 - val_tf_psnr: 44.2733
Epoch 41/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.0789e-05 - tf_psnr: 43.9160 - val_loss: 3.8106e-05 - val_tf_psnr: 44.3243
Epoch 42/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.0276e-05 - tf_psnr: 43.9642 - val_loss: 3.7813e-05 - val_tf_psnr: 44.3579
Epoch 43/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.0046e-05 - tf_psnr: 43.9894 - val_loss: 3.9289e-05 - val_tf_psnr: 44.1945
Epoch 44/50
3240/3240 [==============================] - 41s 13ms/step - loss: 4.0240e-05 - tf_psnr: 43.9722 - val_loss: 3.9224e-05 - val_tf_psnr: 44.1908
Epoch 45/50
3240/3240 [==============================] - 41s 13ms/step - loss: 3.9737e-05 - tf_psnr: 44.0213 - val_loss: 3.7717e-05 - val_tf_psnr: 44.3750
Epoch 46/50
3240/3240 [==============================] - 41s 13ms/step - loss: 3.9740e-05 - tf_psnr: 44.0227 - val_loss: 3.7677e-05 - val_tf_psnr: 44.3685
Epoch 47/50
3240/3240 [==============================] - 41s 13ms/step - loss: 3.9525e-05 - tf_psnr: 44.0475 - val_loss: 3.7830e-05 - val_tf_psnr: 44.3588
Epoch 48/50
3240/3240 [==============================] - 41s 13ms/step - loss: 3.9421e-05 - tf_psnr: 44.0585 - val_loss: 3.8817e-05 - val_tf_psnr: 44.2335
Epoch 49/50
3240/3240 [==============================] - 41s 13ms/step - loss: 3.9357e-05 - tf_psnr: 44.0641 - val_loss: 3.7615e-05 - val_tf_psnr: 44.3764
Epoch 50/50
3240/3240 [==============================] - 41s 13ms/step - loss: 3.9391e-05 - tf_psnr: 44.0675 - val_loss: 4.1242e-05 - val_tf_psnr: 43.9490
psnr 50 epochs..... 42.81006401606692
354/354 [==============================] - 2s 5ms/step
Test loss: 4.409387822234661e-05
Test accuracy: 43.566977829582946
```

