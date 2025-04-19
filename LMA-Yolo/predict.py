import argparse
from ultralytics import YOLO

def main(opt):
    # 加载训练好的模型权重
    model = YOLO(opt.weights)

    # 执行推理
    results = model.predict(
        source=opt.source,     # 输入：图片、文件夹或视频路径
        imgsz=opt.imgsz,       # 推理尺寸
        conf=opt.conf,         # 置信度阈值
        device=opt.device,     # 使用哪块GPU（或cpu）
        save=True,             # 保存预测图像
        save_txt=True,         # 保存标签文件（YOLO格式）
        save_conf=True,        # 保存置信度
        show=False             # 是否弹窗展示（服务器一般设为 False）
    )

    print("✅ Prediction finished.")
    for r in results:
        print(r)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to model weights')
    parser.add_argument('--source', type=str, default='inference/images', help='image, folder, or video file')
    parser.add_argument('--imgsz', type=int, default=640, help='inference image size')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or cpu')
    opt = parser.parse_args()

    main(opt)
