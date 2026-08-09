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
- Single-command agent execution

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
│   ├── ITAI1378_Final_PPE_Compliance_Check_.ipynb
│   └── README.md
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
├── requirements.txt
└── run_agent.py
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

The project has been verified in a fresh Google Colab environment to confirm that the complete PPE Compliance Agent workflow is reproducible.

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

### 3. Run the Complete Agent

The complete PPE Compliance Check Agent can be executed from the repository root with a single command:

```bash
python run_agent.py
```

The runner automatically:

- Loads the custom-trained YOLO11 PPE model from `models/best.pt`
- Downloads `best.pt` from Google Drive automatically if the model is not already present
- Loads the representative sample inputs from `data/sample/`
- Processes the included sample images
- Applies PPE compliance reasoning
- Generates annotated evidence and JSON reports
- Evaluates 10 representative video scenarios
- Runs continuous video monitoring
- Generates evaluation metrics
- Records a complete agent execution trace
- Saves generated outputs under `results/`

This provides a reproducible end-to-end execution path:

**Input → Perception → Reasoning → Action → Evaluation → Traceability**

### 4. Trained YOLO11 Model

The trained YOLO11 weights (`best.pt`) are hosted externally to keep the repository lightweight.

When `python run_agent.py` is executed, the runner checks for:

```text
models/best.pt
```

If the model is not present, it is downloaded automatically from Google Drive.

The model can also be downloaded manually:

[Download best.pt from Google Drive](https://drive.google.com/file/d/1kmJXMGUUwuhxe7IGcxgp0IkeUppP-7lY/view?usp=sharing)

After manual download, place the model at:

```text
models/best.pt
```

The model can also be reproduced from scratch by running the dataset setup and training sections of the final notebook.

### 5. Open the Final Notebook

The final project notebook provides the detailed development, training, evaluation, and agent workflow:

```text
notebooks/ITAI1378_Final_PPE_Compliance_Check_.ipynb
```

The notebook documents the complete project development process, while `run_agent.py` provides the reproducible single-command execution path for the final agent.

The repository includes sample inputs under:

```text
data/sample/
```

These samples allow the PPE compliance pipeline to be tested without requiring additional user-provided data.

### Verified Fresh-Clone Test

The project was successfully tested from a fresh Google Colab environment using the cloned GitHub repository, pinned dependencies, pretrained YOLO11 model, and included sample data.

The final single-command test was executed with:

```bash
python run_agent.py
```

The fresh-clone test successfully:

- Loaded the custom YOLO11 PPE model
- Detected the `no_helmet` and `helmet` classes
- Processed all 3 included sample images
- Correctly produced `COMPLIANT`, `VIOLATION`, and `REVIEW_REQUIRED` decisions
- Evaluated 10 representative video scenarios
- Successfully processed all 10 evaluation scenarios
- Produced 0 evaluation processing errors
- Achieved a 100% scenario processing success rate
- Processed the complete sample monitoring video
- Generated annotated outputs
- Generated JSON reports
- Generated evaluation metrics
- Generated a complete execution trace

The verified batch image results were:

```text
compliant_example.jpg: COMPLIANT (helmet=2, no_helmet=0)
review_required_example.jpg: REVIEW_REQUIRED (helmet=0, no_helmet=0)
violation_example.jpg: VIOLATION (helmet=2, no_helmet=1)
```

The verified system-level evaluation produced:

```text
Scenarios requested: 10
Successfully processed: 10
Processing errors: 0
Success rate: 100.00%
```

The continuous monitoring workflow processed the complete sample video and generated an annotated monitoring video.

The final execution completed with:

```text
AGENT RUN COMPLETED SUCCESSFULLY

Perception -> Reasoning -> Action -> Traceability complete.
```

This verifies the complete end-to-end agent workflow from a fresh environment using a single documented execution command.

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

### Success and Failure Analysis

The evaluation included successful detections as well as challenging cases that revealed limitations of the perception component.

#### Successful Cases

The agent successfully identified both compliant and violation scenarios. For example, **Scenario 4** detected helmet and no-helmet conditions in the same construction scene and correctly classified the overall status as `VIOLATION`.

**Scenarios 3 and 10** also demonstrated graceful handling of uncertain or irrelevant visual input. When no helmet or no-helmet evidence was detected, the reasoning system returned `REVIEW_REQUIRED` rather than forcing an unsupported compliance decision.

#### Failure Case 1 – Partial Detection in a Crowded Scene

**Scenario 5 (Frame 931)** contains multiple visible workers in a construction environment. The YOLO11 model detected two `no_helmet` instances and the agent correctly classified the scene as `VIOLATION`. However, not every visible worker received a PPE detection.

This demonstrates a limitation of the perception component when multiple workers appear with partial occlusion, varied poses, and construction structures blocking portions of the scene. Although the final safety decision was correct, incomplete detections could affect worker-level compliance analysis.

#### Failure Case 2 – Small and Distant PPE Objects

**Scenario 9 (Frame 1863)** contains workers at a significant distance from the camera. The PPE objects occupy only a small portion of the image, and one helmet detection had relatively low confidence.

This scenario demonstrates that PPE detection becomes less reliable when workers and helmets appear very small or distant within a cluttered scene. Higher-resolution input, additional training examples containing distant workers, and improved small-object detection could improve performance.

### Failure Analysis Summary

These cases show that the agent can complete the end-to-end workflow while the underlying perception model still has limitations. The most important observed challenges were:

- Partial occlusion
- Multiple workers in complex scenes
- Small PPE objects
- Long camera distance
- Visual clutter

Future model training could address these limitations by adding more diverse examples of crowded scenes, partially occluded workers, and small or distant PPE objects.

---

## Example Agent Run

**Input**

```text
Compliant PPE sample image
```

↓

**Perception**

```text
YOLO11 detects:
- 2 helmets
- 0 no-helmet detections
```

↓

**Reasoning**

```text
Helmet detections are present.
No no-helmet detections were identified.
Compliance rules are satisfied.
```

↓

**Decision**

```text
Status: COMPLIANT
```

↓

**Action**

```text
- Annotated evidence generated
- JSON report saved
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
- How to package an AI workflow into a reproducible single-command execution pipeline.
- The importance of reproducibility, documentation, and systematic evaluation in AI development.

---

## AI Usage

AI tools were used throughout the project to assist with debugging, code explanations, documentation, and repository organization.

A detailed record of AI-assisted development is available in:

```text
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
- `notebooks/README.md`

The reproducible agent entry point is:

- `run_agent.py`

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
