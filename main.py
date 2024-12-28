import ultralytics
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model = YOLO("yolov8n.pt").to(device)

    # def on_train_epoch_end(trainer):
    #     result = model("potholes/output/train/images/potholes0.png")
    #     result.save(save_dir = "potholes/example")
    #
    # model.add_callback("on_train_epoch_end", on_train_epoch_end)

    train_results = model.train(data="data.yaml", epochs=100, imgsz = 640)

    model.export()