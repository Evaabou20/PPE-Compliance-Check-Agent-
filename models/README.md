# Model Documentation

## Model

This project uses a custom-trained **YOLO11** object detection model for PPE compliance detection.

## Classes

The model detects two classes:

- `helmet`
- `no_helmet`

## Training

The model was trained during the midterm portion of the project using **50 epochs**. The final PPE Compliance Agent loads the saved trained weights instead of retraining the model during every execution.

## Model File

The trained YOLO11 weights (`best.pt`) are hosted externally to keep the repository lightweight.

Download the trained model here:

[Download best.pt from Google Drive](https://drive.google.com/file/d/1kmJXMGUUwuhxe7IGcxgp0IkeUppP-7lY/view?usp=sharing)

After downloading, place the file at:

`models/best.pt`

This allows the pretrained PPE detector to be used without retraining the model.

## Reproducing the Model

To reproduce the trained model from scratch:

1. Open the notebook in the `notebooks` folder.
2. Run the dataset setup and training sections.
3. The trained weights (`best.pt`) will be generated after training.
4. Alternatively, use the pretrained `best.pt` provided above for inference and evaluation.

## Inference Configuration

- Model: YOLO11
- Image size: 640
- Confidence threshold: 0.30
- Device: GPU when available, otherwise CPU
