# PPE Compliance Check Agent

## Project Overview

The **PPE Compliance Check Agent** is an intelligent computer vision system developed to automatically monitor Personal Protective Equipment (PPE) compliance in construction and industrial environments.

The project combines a custom-trained **YOLO11** object detection model with an AI decision-making agent capable of detecting workers, identifying helmet usage, classifying compliance status, generating annotated visual evidence, producing structured JSON reports, and monitoring entire video streams.

The final system supports both single-image evaluation and continuous video monitoring while maintaining a conservative decision-making strategy by requesting manual review whenever visual evidence is insufficient.

---

## Project Objectives

The primary objectives of this project are to:

- Detect construction workers using computer vision.
- Identify whether required PPE (helmet) is present.
- Automatically classify each scenario as:
  - **COMPLIANT**
  - **VIOLATION**
  - **REVIEW_REQUIRED**
- Generate annotated evidence images.
- Produce structured evaluation reports.
- Monitor PPE compliance throughout an entire video.
- Demonstrate an AI agent pipeline integrating perception, reasoning, evaluation, and reporting.

---

## Key Features

- Custom-trained YOLO11 PPE detector
- AI agent decision-making pipeline
- Automatic compliance reasoning
- Annotated image generation
- Continuous video monitoring
- JSON evaluation reports
- Performance metrics collection
- Failure and review-case analysis
- Modular notebook implementation
