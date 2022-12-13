
![ES-UNet Architecture](img/architecture.png?raw=true)

## Environment
Prepare an environment with python=3.7, and then run the command "pip install -r requirements.txt" for the dependencies.

## models 
Copy the models from ./models into ./monai/networks/net

## Data Preparation
- We used a 3D PET-CT LYMPHOMA dataset to test your model.
- Users can prepare their own dataset and put the data according to the following file structure:
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

## Pre-Trained baseline model: UNET
- UNET: Download the pre-trained UNet model and put it under ./pre-trained_model folder

## Pre-Trained ENN_UNet and RBF-UNet
- ENN-UNet: Download the pre-trained ENN-UNet model and put it under ./pre-trained_model folder

- RBF-UNet: Download the pre-trained RBF-UNet model and put it under ./pre-trained_model main folder




## Train ENN_UNet with random initialization (same for RBF_UNet)
- First, we need to train a baseline model UNet. (Here, we offer a pre-trained UNet baseline model in ./pre-trained_model folder)
- Second, change to the path to the pre-trained baseline model UNet in TRAINING-ENN.py (line 116)
- Third, run the following code to train ENN_UNet. 
```bash
python TRAINING-ENN.py
```

## Train ENN_UNet_KMEANS with k-means initialization (same for RBF_UNet_KMEANS).
- First, we need to train a baseline model UNet. (Here, we offer a pre-trained UNet baseline model in ./pre-trained_model/ folder)
- Second, we use the pre-trained UNet to calculate the initialized value of prototypes by the k-means algorithm. (Here, we offer an initialization value of prototypes that are calculted by K-means, ./Center-kmeans.txt)
- Third, change to the path to the pre-trained baseline model UNet in TRAINING-ENN_(step1).py  (line 109)
- Fourth, run following code to train the ENN layer only by fixing UNet (activate lines 263-264 in une_enn_kmeans.py).
```bash
python TRAINING-ENN_(step1).py
```
- Last, run following code to finetune the whole model with the pre-trained model obtained from step 1 (deactivate lines 263-264 in une_enn_kmeans.py to disable gradient update for Unet).   

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



