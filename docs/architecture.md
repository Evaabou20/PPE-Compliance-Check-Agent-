# PPE Compliance Check Agent Architecture

## Overview

The PPE Compliance Check Agent is an end-to-end AI system that integrates computer vision, rule-based reasoning, evidence generation, evaluation, and trace logging into a complete autonomous workflow. The agent processes images and videos, detects Personal Protective Equipment (PPE), determines compliance status, generates annotated evidence, and records every decision for later analysis.

---

## Agent Pipeline

```text
                +-----------------------+
                |  Input Ingestion      |
                | Images / Videos       |
                +----------+------------+
                           |
                           v
                +-----------------------+
                | Input Validation      |
                | Format & File Checks  |
                +----------+------------+
                           |
                           v
                +-----------------------+
                | Preprocessing         |
                | Resize / Normalize    |
                +----------+------------+
                           |
                           v
                +-----------------------+
                | YOLO11 Perception     |
                | Detect PPE Objects    |
                +----------+------------+
                           |
                           v
                +-----------------------+
                | Compliance Reasoning  |
                | Rule-Based Decisions  |
                +----------+------------+
                           |
                           v
                +-----------------------+
                | Action Generation     |
                | Reports / Images      |
                +----------+------------+
                           |
                           v
                +-----------------------+
                | Trace Logging         |
                | JSON & Metrics        |
                +----------+------------+
                           |
                           v
                +-----------------------+
                | Evaluation            |
                | Performance Analysis  |
                +----------+------------+
                           |
                           v
                +-----------------------+
                | Continuous Monitoring |
                | Image & Video Streams |
                +-----------------------+
```

---

## Pipeline Stages

### 1. Input Ingestion

The agent accepts images and videos provided by the user for PPE compliance analysis.

### 2. Input Validation

The system verifies that input files are valid before processing and prevents invalid data from interrupting execution.

### 3. Preprocessing

Input images are prepared for inference through resizing and normalization to ensure compatibility with the YOLO11 model.

### 4. Perception

A custom-trained YOLO11 object detection model identifies workers wearing helmets and workers without helmets.

### 5. Compliance Reasoning

Rule-based logic evaluates the detection results and determines whether each worker is compliant, in violation, or requires manual review.

### 6. Action Generation

The agent generates annotated images, annotated monitoring videos, JSON reports, evaluation metrics, and structured outputs.

### 7. Trace Logging

Every execution records perception results, reasoning decisions, compliance status, and generated outputs to support reproducibility and debugging.

### 8. Evaluation

The system evaluates multiple scenarios and records component-level and system-level performance metrics.

### 9. Continuous Monitoring

The complete pipeline supports continuous monitoring of PPE compliance from video streams and sequential image processing.

---

## Agent Components

| Component | Responsibility |
|----------|----------------|
| Input Manager | Receives images and videos |
| YOLO11 Detector | Detects helmets and workers |
| Reasoning Engine | Determines compliance status |
| Report Generator | Produces annotated outputs and JSON reports |
| Trace Logger | Records execution history |
| Evaluation Module | Measures system performance |

---

## Overall Workflow

The PPE Compliance Check Agent combines perception, reasoning, action, and evaluation into a complete autonomous computer vision pipeline capable of monitoring PPE compliance while producing reproducible outputs and detailed execution records.