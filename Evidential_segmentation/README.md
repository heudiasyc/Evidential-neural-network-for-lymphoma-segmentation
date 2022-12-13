
![ES-UNet Architecture](img/architecture.png?raw=true)

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
- Download UNet pre-trained weights of UNet and put it under ./pre-trained_model folder

## Pre-Trained Base Model For LYMPHOMA
- ENN-UNet: 
- Download ENN-UNet pre-trained model and put it under ./pre-trained_model folder

- RBF-UNet: 
- Download RBF-UNet pre-trained model and put it under ./pre-trained_model main folder




## Train UNet_ENN with random initialization (same for UNet_RBF)
- To train the whole model with random initialization, first, we need to pre-trained baseline model UNet. (Here we offer a pretrained UNet baseline model in ./pre-trained_model folder)
- Change to path to the pretrained baseline model Unet in TRAINING-ENN.py (line 116)
- Run the following code to train UNet_ENN. 
```bash
python TRAINING-ENN.py
```

## Train UNet_ENN_KMEANS with k-means initialization (same for UNet_RBF_KMEANS).
- To train the whole model with k-means initialization, first, we need to pre-trained baseline model UNet. (Here we offer a pretrained UNet baseline model in ./pre-trained_model/ folder)
_ Second we use the pre-trained UNet to calculate the initialize value of prototypes by k-means algorithom. (Here we offer a initialization value of prototypes that calculted by K-means, ./Center-kmeans.txt)
- Third run TRAINING-ENN_(step1).py to train the RBF/ENN layer only by fixing UNEt. 
```bash
python TRAINING-ENN_(step1).py
```
- Last run TRAINING-ENN_(step2).py to finetune the whole model with the pre-trained model on step 1.   

```bash
python TRAINING-ENN_(step2).py
```

## Acknowledgements
This repository makes liberal use of code from [DeepOncology](https://github.com/rnoyelle/DeepOncology) for lymphoma data processing.


## Citing ES-UNet
```bash
@article{HUANG202239,
title = {Lymphoma segmentation from 3D PET-CT images using a deep evidential network},
journal = {International Journal of Approximate Reasoning},
volume = {149},
pages = {39-60},
year = {2022},
issn = {0888-613X},
doi = {https://doi.org/10.1016/j.ijar.2022.06.007},
url = {https://www.sciencedirect.com/science/article/pii/S0888613X22000962},
author = {Ling Huang and Su Ruan and Pierre Decazes and Thierry Denœux},
}

@inproceedings{huang2021evidential,
  title={Evidential segmentation of 3D PET/CT images},
  author={Huang, Ling and Ruan, Su and Decazes, Pierre and Denoeux, Thierry},
  booktitle={International Conference on Belief Functions},
  pages={159--167},
  year={2021},
  organization={Springer}
}
```



