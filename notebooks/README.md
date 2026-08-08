# Final Project Notebook

This folder contains the final Jupyter notebook for the **PPE Compliance Check Agent**.

## Notebook

`ITAI1378_Final_PPE_Compliance_Check_.ipynb`

The notebook documents the development and implementation of the PPE Compliance Check Agent, including:

- Dataset preparation and exploration
- Custom YOLO11 PPE model training
- Model evaluation
- Image and video inference
- PPE compliance reasoning
- Annotated evidence generation
- JSON report generation
- Agent execution traces
- System-level evaluation across 10 representative scenarios
- Final end-to-end PPE Compliance Agent pipeline

## Running the Project

For complete installation and reproducibility instructions, including how to download the trained `best.pt` model, see the main repository `README.md`.

The final agent implementation is located in **Section 10** of the notebook.

## Reproducibility

The final agent uses repository-based paths and loads the pretrained model from:

`models/best.pt`

Sample inputs are provided under:

`data/sample/`

Generated artifacts are saved under:

`results/`
