import argparse
from ultralytics import YOLO

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/yolov8-lma.yaml', help='Path to model config (.yaml)')
    parser.add_argument('--data', type=str, default='data/rosd.yaml', help='Path to dataset config (.yaml)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--device', type=str, default='0', help='CUDA device (e.g., 0 or "cpu")')
    parser.add_argument('--name', type=str, default='LMA-YOLO', help='Name of training run')
    return parser.parse_args()

def main(opt):
    print(f"🚀 Starting training with config:\n{opt}")
    model = YOLO(opt.model)
    model.train(
        data=opt.data,
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=opt.device,
        name=opt.name,
        workers=12,
        optimizer='SGD',
        amp=True
    )

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
