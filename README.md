## IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing	##

## LMA-YOLO Execution Pipeline

The execution of the LMA-YOLO model can be broadly divided into the following four steps:

## Step 1: Environment Setup

Configure the Python and PyTorch environment (Python ≥ 3.8, PyTorch ≥ 2.0 recommended)

Set up the YOLO-based environment (install ultralytics and required dependencies)

pip install ultralytics </br>
pip install -r requirements.txt

## Step 2: Data Preparation

Prepare the dataset images and labels (in YOLO format .txt files)

Check and configure the dataset directory structure as follows:

dataset/ </br>
├── images/ </br>
│   ├── train/</br>
│   └── val/</br>
└── labels/</br>
    ├── train/</br>
    └── val/</br>
    
## Step 3: Model Configuration and Training

Define a custom YAML configuration file (specifying model architecture and hyperparameters)
Example:

Run the training script: </br>
python train.py --cfg models/LMA_yolo.yaml --data data/dataset.yaml --epochs 200 --batch 16 --img 640 --device 0

## Step 4: Model Evaluation and Inference

Validate the performance of the trained model: </br>
python v8_predict.py --weights runs/detect/train/weights/best.pt --data data/dataset.yaml --img 640 --batch 16 --device 0
