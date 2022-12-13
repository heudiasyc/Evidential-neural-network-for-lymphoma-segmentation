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
      |    |---TRAINING-ENN_(step1).py
      |    |---TRAINING-ENN_(step2).py
      |---RBF-UNET
      |    |---TRAINING-RBF.py
      |    |---TRAINING-RBF_(step1).py
      |    |---TRAINING-RBF_(step2).py
      |---pretrained_ckpt
      |---saved_model
      ...
    ```

## Pre-Trained Weights of UNET
- UNET: 
- Download UNet pre-trained weights of UNet and add it under ./pre-trained_model folder

## Pre-Trained Base Model For LYMPHOMA
- ENN-UNet: 
- Download ENN-UNet pre-trained model and add it under ./pre-trained_model folder

- RBF-UNet: 
- Download RBF-UNet pre-trained model and add it under ./pre-trained_model main folder




## Train/Test with random initialization
- Random initialization: Run the train script on LYMPHOMA Dataset end-to-end. 
```bash
python TRAINING-ENN.py
```

## Train/Test with k-means initialization  

- First run TRAINING-ENN_step1.py to train the RBF/ENN layer only by fixing UNEt.
- Then run TRAINING-ENN_step2.py to finetune the whole model with the pre-trained model on step 1.  
```bash
python TRAINING-ENN_step1.py
python TRAINING-ENN_step2.py
```

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



