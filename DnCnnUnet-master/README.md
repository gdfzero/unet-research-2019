July23

#### How to run DnCnnUnet


##### Step 1: Download Anocanda from the web. 

```{python}
[npovey@da DnCnnUnet]$ scp -r npovey@k:/home/npovey/Anaconda3-2019.03-Linux-x86_64.sh .
```

##### Step 2:  bash

```{python}
[npovey@da DnCnnUnet]$ bash Anaconda3-2019.03-Linux-x86_64.sh 

```

##### Step 3: Create environment

Must sign out and sign in, otherwise will get bash: conda: command not found... 

```{python}
(base) [npovey@da DnCnnUnet]$ conda env create -f dncnnEnv1.yaml --name dncnnEnv1

```

##### Step 4: Find out what environments are installed

```{python}
(dncnnEnv1) [npovey@da DnCnnUnet]$ conda info --envs
# conda environments:
#
base                     /home/npovey/anaconda3
dncnnEnv1             *  /home/npovey/anaconda3/envs/dncnnEnv1
myenv                    /home/npovey/anaconda3/envs/myenv
python36                 /home/npovey/anaconda3/envs/python36 
```


##### Step 5:  Activate the environment

```{python}
(base) [npovey@da DnCnnUnet]$ conda activate dncnnEnv1

```

##### Step 6: Make .npy array for the normal dose data

```{python}
(dncnnEnv1) [npovey@da DnCnnUnet]$ python read_float.py --src_dir DnCnnData/images/ndct/train/ --save_dir DnCnnData/images/ndct/train --save_name patches_ndct

```

##### Step 7: Make the .npy array for the low dose data

```{python}
(dncnnEnv1) [npovey@da DnCnnUnet]$ python read_float.py --src_dir DnCnnData/sparseview/sparseview_90/train/ --save_dir DnCnnData/sparseview/sparseview_90/train --save_name patches_sparseview_90
```
##### Step 8: Use only one gpu number zero
```{python}
(dncnnEnv1) [npovey@da DnCnnUnet]$ export CUDA_VISIBLE_DEVICES=0
```

##### Step 9: Run to train

```{python}
(dncnnEnv1) [npovey@da DnCnnUnet]$ python main.py --ndct_training_data "DnCnnData/images/ndct/train/patches_ndct.npy" --ldct_training_data "DnCnnData/sparseview/sparseview_90/train/patches_sparseview_90.npy" --ndct_eval_data "DnCnnData/images/ndct/test/*.flt" --ldct_eval_data "DnCnnData/sparseview/sparseview_90/test/*.flt"

```
##### Step 10: Deactivate the current environment
```{python}
(dncnnEnv1) [npovey@da DnCnnUnet]$ conda deactivate
(base) [npovey@da DnCnnUnet]$ 
```
