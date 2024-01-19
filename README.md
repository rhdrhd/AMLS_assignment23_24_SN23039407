# AMLS_assignment23_24

This report presents the development and evaluation of two specialized machine learning models for different classification tasks.

## Project structure
The current project structure is shown below
```
├── A
│   ├── init.py
│   ├── images
│   ├── pretrained_weights_customCNN
│   ├── classical_ml_algorithms.py
│   ├── CNN_model.py
│   ├── utils.py
│   ├── hypertuning_process.ipynb
├── B
│   ├── init.py
│   ├── DL_model.py
│   ├── utils.py
│   ├── images
│   ├── pretrained_weights
├── Datasets
├── environment.yml
├── requirements.txt
├── README.md
└── main.py
```

**main.py**: Contains the core of the project, including the training and testing options of the models for Task A and Task B.

## How to start
1. Create a new conda environment from environment.yml file.
```
conda env create -f environment.yml
```
2. Activate this conda virtual environment. 
```
conda activate aml-final
```
3. Run main.py if all the dependencies required for the current project are already installed. 

```
python main.py
```
## NOTES

The main file is defaulted to test the model due to extensive time duration it takes to train ResNet models. Train mode can be manually configured. To make sure about 100% reproducibility please load pretrained weights.

## MODEL NAME LIST
    model name can be selected from the following:
    ResNet18_28, ResNet18_32, ResNet18_224
    ResNet18_28_dropout, ResNet18_32_dropout, ResNet18_224_dropout
    ResNet50_28, ResNet50_32, ResNet50_224
    ResNet50_28_dropout, ResNet50_32_dropout, ResNet50_224_dropout