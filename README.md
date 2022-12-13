# Evidential-neural-network-for-lymphoma-segmentation
Code for paper "Lymphoma segmentation from 3D PET-CT images using a deep evidential network"



# ES-UNet
This repo contains the supported pytorch code and configuration files to reproduce 3D lymphoma segmentaion results of [ENN-UNet and RBF-UNet](https://arxiv.org/abs/2201.13078). 


## Environment
Prepare an environment with python=3.7, and then run the command "pip install -r requirements.txt" for the dependencies.

## models 
Copy the models from ./models into ./monai/networks/net

## Data Preparation
- For experiments we used LYMPHOMA dataset.
- Users can can prapare their own dataset according and put the data into follow files 

- File structure
    ```
     LYMPHOMA
      |---Data
      |   |--- ct
      |   |   |--- AA001ct.nii...
      |   |--- pet
      |   |   |--- AA001pet.nii...
      |   |--- pet_mask
      |   |   |--- AA001mask.nii...  
     ES-UNet
      |---ENN-UNET
      |    |---TRAINING-ENN.py
      |    |---TRAINING-ENN_step1.py
      |    |---TRAINING-ENN_step2.py
      |---RBF-UNET
      |    |---TRAINING-RBF.py
      |    |---TRAINING-RBF_step1.py
      |    |---TRAINING-RBF_step2.py
      |---pretrained_ckpt
      |---saved_model
      ...
    ```

## Pre-Trained model 
- UNET: 
- Download UNet pre-trained weights of UNet and add it under ./pre-trained_model folder

## Pre-Trained Base Model For LYMPHOMA
- ENN-UNet: 
- Download ENN-UNet pre-trained model and add it under ./pre-trained_model folder

- RBF-UNet: 
- Download RBF-UNet pre-trained model and add it under ./pre-trained_model main folder




## Train ENN-UNet with random initialization (the same for RBF-UNet):
-  First, train a baseline UNet model (here we provide a pretained baseline UNet model in ./pre-trained_model/ )
-  Second, indicate the path to the pre-traind UNet model (TRAINING-ENN.py line 116)
-  Third, run the following code to train ENN-UNet
```bash
python TRAINING-ENN.py
```

## Train ENN-UNet with k-means initialization (the same for RBF-UNet):
- First, train a baseline UNet model (here we provide a pretained baseline UNet model in ./pre-trained_model/ )
- Second, indicate the path to the pre-traind UNet model (TRAINING-ENN_step1.py line 109)
- Third, run the following code to train ENN by fixing UNet (activate lines 265-266 in unet_enn.py)
```bash
python TRAINING-ENN_step1.py
```
- Last, run the following code finetune the whole model (ENN-UNet). ((activate lines 265-266 in unet_enn.py)
```bash
python TRAINING-ENN_step2.py
```
## PS, python TRAINING-ENN_step2.py is not necessary if TRAINING-ENN_step2.py can already output good results. But a final finetuning of the whold model usually have better results 

## Acknowledgements
This repository makes liberal use of code from [DeepOncology](https://github.com/rnoyelle/DeepOncology) for lymphoma data processing.


## Citing ES-UNet
```bash
@inproceedings{huang2021evidential,
  title={Evidential segmentation of 3D PET/CT images},
  author={Huang, L. and Ruan, S. and Decazes, P. and Den{\oe}ux, T.},
  booktitle={International Conference on Belief Functions},
  pages={159--167},
  year={2021},
  organization={Springer}
}

@article{huang2022lymphoma,
title = {Lymphoma segmentation from 3D PET-CT images using a deep evidential network},
journal = {International Journal of Approximate Reasoning},
volume = {149},
pages = {39-60},
year = {2022},
issn = {0888-613X},
doi = {https://doi.org/10.1016/j.ijar.2022.06.007},
author = {Ling Huang and Su Ruan and Pierre Decazes and Thierry Denoeux}
}


```



