# PPE Compliance Check Agent

## Project Overview

The **PPE Compliance Check Agent** is an AI-powered computer vision system that automatically monitors Personal Protective Equipment (PPE) compliance in construction and industrial environments. The agent uses a custom-trained **YOLO11** object detection model to identify workers wearing helmets or not wearing helmets, applies reasoning rules to determine compliance status, and generates annotated visual evidence, reports, evaluation metrics, and execution traces.

---

## Project Tier

**Tier 3 – Autonomous AI Agent**

This project goes beyond basic object detection by integrating perception, reasoning, evaluation, reporting, and monitoring into a complete autonomous workflow. The agent analyzes images and videos, determines PPE compliance, produces structured outputs, and records execution traces for evaluation.

---

## Problem & Solution

### Problem

Construction sites present significant safety risks when workers fail to wear required Personal Protective Equipment (PPE). Manual monitoring is time-consuming, inconsistent, and difficult to scale across large job sites.

### Solution

The PPE Compliance Check Agent automates helmet compliance monitoring using computer vision. The system detects helmets and missing helmets, evaluates worker compliance, produces annotated evidence, generates structured reports, and supports continuous monitoring from video streams.

---

## Project Features

- Custom-trained YOLO11 PPE detector
- Helmet and no-helmet detection
- AI compliance reasoning
- Image processing
- Video monitoring
- Automatic compliance classification
- Annotated evidence generation
- JSON report generation
- Evaluation metrics
- Agent execution traces

---

## Agent Architecture

The agent follows a multi-stage AI pipeline:

```text
Input
   │
   ▼
Image / Video Acquisition
   │
   ▼
YOLO11 Object Detection
   │
   ▼
PPE Compliance Reasoning
   │
   ▼
Decision Making
   │
   ▼
Evidence Generation
   │
   ▼
Reports • Metrics • Traces • Annotated Outputs
```

---

## Dataset

The project uses a custom PPE dataset for training, validation, and evaluation.

The GitHub repository includes representative sample images and videos that demonstrate the complete agent workflow while keeping the repository lightweight.

Sample data is available in:

```text
data/sample/
├── images/
└── videos/
```

---

## Repository Structure

```text
PPE-Compliance-Check-Agent/
│
├── data/
│   ├── README.md
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
├── README.md
└── requirements.txt
```

---

## Technologies

- Python
- Ultralytics YOLO11
- PyTorch
- OpenCV
- ONNX Runtime
- NumPy
- Matplotlib
- Google Colab
- Google Drive
- GitHub

---

## Installation & Reproducibility

The project has been verified in a fresh Google Colab environment to confirm that the core PPE Compliance Agent workflow is reproducible.

### 1. Clone the Repository

```bash
git clone https://github.com/Evaabou20/PPE-Compliance-Check-Agent-.git
cd PPE-Compliance-Check-Agent-
```

### 2. Install Dependencies

Install the pinned project dependencies:

```bash
pip install -r requirements.txt
```

### 3. Download the Trained YOLO11 Model

The trained YOLO11 weights (`best.pt`) are hosted externally to keep the repository lightweight.

[Download best.pt from Google Drive](https://drive.google.com/file/d/1kmJXMGUUwuhxe7IGcxgp0IkeUppP-7lY/view?usp=sharing)

After downloading, place the model at:

```text
models/best.pt
```

The model can also be reproduced from scratch by running the dataset setup and training sections of the final notebook.

### 4. Open the Final Notebook

The final project notebook is located at:

```text
notebooks/ITAI1378_Final_PPE_Compliance_Check_.ipynb
```

The repository includes sample inputs under:

```text
data/sample/
```

These samples allow the PPE compliance pipeline to be tested without requiring additional user-provided data.

### Verified Fresh-Clone Test

The project was successfully tested from a fresh Google Colab environment using the pinned dependencies, pretrained YOLO11 model, and sample data included in the repository.

The fresh-clone test successfully:

- Installed all dependencies from `requirements.txt`
- Downloaded and loaded the custom YOLO11 PPE model
- Processed an included sample image
- Detected PPE objects
- Applied helmet-compliance reasoning
- Produced a compliance decision
- Generated annotated evidence
- Generated a JSON report
- Generated an execution trace

For the verified compliant sample, the agent detected:

```text
Helmet detections: 2
No-helmet detections: 0
Status: COMPLIANT
```

This verifies the complete end-to-end:

**Perception → Reasoning → Action → Traceability**

workflow from a fresh environment.

---

## Evaluation & Results

The PPE Compliance Check Agent was evaluated using ten representative test scenarios containing compliant workers, PPE violations, and uncertain situations requiring manual review.

### Component-Level Evaluation

The YOLO11 model was evaluated based on:

- Object detection performance
- Inference latency
- Detection consistency
- Annotated output quality

### System-Level Evaluation

The complete agent was evaluated on its ability to:

- Process images and videos successfully
- Correctly classify PPE compliance
- Generate annotated evidence
- Produce structured reports
- Save execution traces for every scenario

The evaluation generated:

- Annotated evaluation images
- Annotated monitoring videos
- JSON compliance reports
- Performance metrics
- Agent execution traces

Representative outputs are available in the `results` folder.

---

## Example Agent Run

**Input**

```
Sample PPE image
```

↓

**Perception**

```
YOLO11 detects:
- Person
- Helmet
```

↓

**Reasoning**

```
Helmet detected.

Compliance rules satisfied.
```

↓

**Decision**

```
Worker classified as COMPLIANT.
```

↓

**Action**

```
- Annotated image generated
- JSON report saved
- Evaluation metrics updated
- Execution trace recorded
```

---

## Impact

The PPE Compliance Check Agent demonstrates how artificial intelligence can automate workplace safety inspections. By reducing manual monitoring, the system can improve consistency, increase inspection efficiency, and provide documented evidence for safety compliance reviews.

---

## Key Learnings

Throughout this project I learned:

- How to train and evaluate a custom YOLO11 object detection model.
- How to build an end-to-end computer vision agent that integrates perception, reasoning, and action.
- How to organize AI projects using a professional GitHub repository structure.
- How to generate evaluation metrics, execution traces, and project documentation.
- The importance of reproducibility, documentation, and systematic evaluation in AI development.

---

## AI Usage

AI tools were used throughout the project to assist with debugging, code explanations, documentation, and repository organization.

A detailed record of AI-assisted development is available in:

```
docs/AI_usage_log.md
```

---

## Future Improvements

Possible future enhancements include:

- Detect additional PPE equipment such as safety vests, gloves, goggles, and boots.
- Deploy the agent as a real-time web application.
- Integrate live surveillance camera monitoring.
- Expand the reasoning engine using large language models.
- Improve robustness under challenging lighting and occlusion conditions.

---

## Documentation

Additional documentation is available in:

- `docs/AI_usage_log.md`
- `docs/architecture.md`
- `data/README.md`
- `models/README.md`

---

## References

- Ultralytics YOLO11 Documentation
- PyTorch Documentation
- OpenCV Documentation
- ONNX Runtime Documentation
- Google Colab Documentation

---

## Author

**Eva Abou Harb**

Houston City College

ITAI 1378 – Computer Vision

Summer 2026

---

## License

This repository was created for educational purposes as part of the ITAI 1378 Computer Vision course at Houston City College.
