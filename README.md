# PPE Compliance Check Agent

## Project Overview

The **PPE Compliance Check Agent** is an AI-powered computer vision system developed to automatically monitor Personal Protective Equipment (PPE) compliance in construction and industrial environments.

The project combines a custom-trained **YOLO11** object detection model with an intelligent decision-making pipeline capable of:

- Detecting workers wearing helmets
- Identifying workers without helmets
- Classifying PPE compliance
- Generating annotated visual evidence
- Producing structured JSON reports
- Monitoring PPE compliance in videos

---

## Features

- Custom-trained YOLO11 PPE detector
- Helmet and no-helmet detection
- AI compliance reasoning
- Image and video processing
- Continuous video monitoring
- Annotated output generation
- Evaluation metrics
- JSON reports and execution traces

---

## Repository Structure

```text
PPE-Compliance-Check-Agent-/
│
├── data/
│   └── sample/
│       ├── images/
│       └── videos/
│
├── docs/
│   ├── AI_usage_log.md
│   └── architecture.md
│
├── models/
│   └── README.md
│
├── notebooks/
│   └── ITAI1378_Final_PPE_Compliance_Check_.ipynb
│
├── results/
│   ├── evaluation/
│   │   └── images/
│   ├── images/
│   ├── metrics/
│   ├── reports/
│   ├── traces/
│   └── videos/
│
└── README.md
```

---

## Technologies

- Python
- Ultralytics YOLO11
- OpenCV
- PyTorch
- NumPy
- Google Colab
- Google Drive
- GitHub

---

## Running the Project

1. Clone this repository.
2. Open `ITAI1378_Final_PPE_Compliance_Check_.ipynb` from the `notebooks` folder.
3. Install the required dependencies.
4. Mount Google Drive.
5. Update the dataset and model paths if necessary.
6. Run the notebook from beginning to end.

---

## Repository Contents

This repository includes:

- Final project notebook
- Sample images and video
- Evaluation outputs
- Monitoring videos
- Performance metrics
- Agent execution traces
- Project documentation
- Model documentation

---

## Results

The project generates:

- Annotated evaluation images
- Annotated monitoring videos
- Compliance reports
- Evaluation metrics
- Agent execution traces

Representative outputs are available in the `results` folder.

---

## Documentation

Additional documentation is available in the `docs` folder:

- `AI_usage_log.md`
- `architecture.md`

Model documentation is available in:

- `models/README.md`

---

## Author

**Eva Abou Harb**

Houston City College

AI & Robotics Program

---

## License

This repository was created for educational purposes as part of the Houston City College AI & Robotics program.

