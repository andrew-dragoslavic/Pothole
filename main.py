import ultralytics
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# def on_train_epoch_end(trainer):
#     result = model("potholes/output/train/images/potholes0.png")
#     result.save(save_dir = "potholes/example")
#
# model.add_callback("on_train_epoch_end", on_train_epoch_end)

train_results = model.train(data="data.yaml", epochs=50, imgsz = 640)

model.export()