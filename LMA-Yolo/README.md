LMA-YOLO 

An improved YOLOv8-based object detection model with multi-scale and lightweight attention mechanisms, specifically designed for remote sensing scenarios.

🔧 Highlights

✅ Introduced Efficient Dual Attention (EDA) to capture spatial + channel features.

✅ Integrated Multi-Scale Conv + SPConv to enhance scale robustness.

✅ Replaced Bottleneck with C2f_Star blocks for fewer FLOPs and better feature flow.

 Dataset: RSOD

4 classes: aircraft, oiltank, overpass, playground

Resolution: 300×300 ~ 1024×1024 aerial imagery

 Dataset Structure

datasets/ROSD/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/

 Usage Guide

🔧 1. Train the Model

To train the improved YOLO model with your custom configuration and dataset:

python train.py

You can also customize the training process with additional arguments:

python train.py \
    --model /path/to/your/yolov8-lma.yaml \
    --data /path/to/your/rosd.yaml \
    --epochs 300 \
    --batch 16 \
    --device 0

 Argument Descriptions:

Argument

Description

--model

Path to the YOLO model config file (e.g. .yaml)

--data

Dataset configuration file (includes path and class names)

--epochs

Number of training epochs (default: 200)

--batch

Batch size

--device

Training device (GPU index or 'cpu')

Training results (weights, logs, metrics) will be saved in:

runs/detect/LMA-YOLO/

 2. Run Inference (Prediction)

To run inference using a trained model:

python predict.py --weights weights/best.pt --source path/to/image.jpg

 Examples:

# Run inference on a folder of images
python predict.py --weights weights/best.pt --source datasets/ROSD/images/val

# Run on a single image using CPU
python predict.py --weights weights/best.pt --source test.jpg --device cpu

 Argument Descriptions:

Argument

Description

--weights

Path to the trained model weights (.pt file)

--source

Input image, folder of images, or video file

--imgsz

Inference image size (default: 640)

--conf

Confidence threshold for object detection (default: 0.25)

--device

Inference device (0, 1, or 'cpu')

Prediction results will be saved in:

runs/detect/predict/

The output will include:

Annotated images with bounding boxes

YOLO-format label files (optional)

Confidence values for detections (optional)

 3. Validate the Model (Optional)

To evaluate the trained model on the validation set:

python -c "from ultralytics import YOLO; YOLO('weights/best.pt').val()"

This will compute standard detection metrics like mAP@0.5 and mAP@0.5:0.95, and print a summary to the console.