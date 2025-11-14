from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO(r"C:\Users\dions\Downloads\best (3).pt")   # change to your best.pt path

# Fine-tune (continue training)
results = model.train(
    data=r"C:\Users\dions\Downloads\uncleaned_data\data.yaml",   # dataset config file
    epochs=50,                  # extra epochs to fine-tune
    imgsz=640,                  # image size
    batch=-1,                   # auto batch size (adjust if GPU OOM)
    optimizer="AdamW",          # optimizer
    lr0=0.001,                  # initial learning rate
    lrf=0.01,                   # final learning rate multiplier
    weight_decay=0.0005,        # regularization
    patience=20,                # early stopping
    project="runs/train",       # output folder
    name="pothole_finetune_v8"  # experiment name
)
