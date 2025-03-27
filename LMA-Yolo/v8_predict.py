from ultralytics import YOLO

if __name__ == '__main__':
    # 使用yaml配置文件来创建模型
    # model = YOLO(r'D:\yolov8\ultralytics-main (1)\ultralytics-main\ultralytics\cfg\models\v8\yolov8_c2ff_mult.yaml')

    # 检查并打印是否使用了预训练权重
    # if hasattr(model, 'weights') and model.weights is not None:
    #     print("预训练权重已加载")
    # else:
    #     print("没有加载预训练权重")

    # # 开始训练模型
    # model.train(cfg=r'D:\yolov8\ultralytics-main (1)\ultralytics-main\ultralytics\cfg\default.yaml',
    #             data=r'D:\yolov8\ultralytics-main (1)\ultralytics-main\ultralytics\cfg\datasets\coco.yaml'
    #             )

    #模型验证
    # model = YOLO('ultralytics-main/runs/detect/train89/weights/best.pt')
    # model.val(cfg='ultralytics/cfg/models/v8/yolov8.yaml')

    # 模型推理
    # model = YOLO('runs/detect/yolov8n_exp/best.pt')
    # model.predict(source='dataset/images/test', save=True)

    # 模型导出
    # model = YOLO("Weight/yolov8n.pt")  # load an official model
    # model.export(format="onnx")


    # Load a model
    pth_path=r"/root/autodl-tmp/ultralytics-main/runs/detect/train95/weights/best.pt"
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO(pth_path)  # load a custom model
 
    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category