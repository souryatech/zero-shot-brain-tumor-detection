# Zero-Shot Brain Tumor Detection

## Installation 
Setup environment
```shell script
pip install -r requirements.txt
sudo apt-get update
sudo apt-get install libgl1-mesa-glx -y
```

## Dataset Preparation
We have compiled all of the images from these datasets and created an annotations file from predicted bounding boxes through a highly accurate traditional brain tumor detection model. This was due to a lack of access to a detection dataset with sufficient size. 
Please download all of the images from these links and place it in a folder called "BTZSD_Dataset":
https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset?select=Brain_Cancer+raw+MRI+data 
https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection 
https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection?resource=download 

## Training
To run the training script, please run the training.ipynb notebook.

