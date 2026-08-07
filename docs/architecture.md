# PPE Compliance Check Agent Architecture

## Overview

The PPE Compliance Check Agent is an end-to-end computer vision system that combines a custom-trained YOLO11 model with rule-based reasoning, evidence generation, evaluation, and trace logging.

## Agent Pipeline

The system follows this workflow:

```text
Input Ingestion
→ Input Validation
→ Preprocessing
→ YOLO11 Perception
→ Compliance Reasoning
→ Action and Evidence Generation
→ Trace Logging
→ Evaluation and Continuous Monitoring