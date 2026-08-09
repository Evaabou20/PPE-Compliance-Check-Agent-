from pathlib import Path
from datetime import datetime
import csv
import json
import time

import cv2
import gdown
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# PPE Compliance Check Agent
# Single-command repository runner
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "best.pt"

SAMPLE_DIR = PROJECT_DIR / "data" / "sample"
SAMPLE_IMAGES_DIR = SAMPLE_DIR / "images"
SAMPLE_VIDEOS_DIR = SAMPLE_DIR / "videos"

RESULTS_DIR = PROJECT_DIR / "results"
IMAGES_DIR = RESULTS_DIR / "images"
EVIDENCE_DIR = RESULTS_DIR / "evidence"
REPORTS_DIR = RESULTS_DIR / "reports"
TRACES_DIR = RESULTS_DIR / "traces"
METRICS_DIR = RESULTS_DIR / "metrics"
VIDEOS_DIR = RESULTS_DIR / "videos"
EVALUATION_IMAGES_DIR = RESULTS_DIR / "evaluation" / "images"

CONFIDENCE_THRESHOLD = 0.30
IMAGE_SIZE = 640
NUMBER_OF_SCENARIOS = 10
FRAME_STRIDE = 10

MODEL_DRIVE_ID = "1kmJXMGUUwuhxe7IGcxgp0IkeUppP-7lY"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# Setup
# ============================================================

def create_output_directories():
    folders = [
        RESULTS_DIR,
        IMAGES_DIR,
        EVIDENCE_DIR,
        REPORTS_DIR,
        TRACES_DIR,
        METRICS_DIR,
        VIDEOS_DIR,
        EVALUATION_IMAGES_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def ensure_model_exists():
    if MODEL_PATH.exists():
        print(f"Model found: {MODEL_PATH}")
        return

    print("Trained model was not found.")
    print("Downloading best.pt from Google Drive...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"

    downloaded = gdown.download(
        url=url,
        output=str(MODEL_PATH),
        quiet=False,
    )

    if downloaded is None or not MODEL_PATH.exists():
        raise RuntimeError(
            "Unable to download models/best.pt. "
            "See models/README.md for the manual download link."
        )

    print(f"Model downloaded successfully: {MODEL_PATH}")


def validate_inputs():
    if not SAMPLE_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Sample image directory not found:\n{SAMPLE_IMAGES_DIR}"
        )

    if not SAMPLE_VIDEOS_DIR.exists():
        raise FileNotFoundError(
            f"Sample video directory not found:\n{SAMPLE_VIDEOS_DIR}"
        )

    image_files = sorted(
        [
            path
            for path in SAMPLE_IMAGES_DIR.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    video_files = sorted(
        [
            path
            for path in SAMPLE_VIDEOS_DIR.iterdir()
            if path.suffix.lower() == ".mp4"
        ]
    )

    if not image_files:
        raise RuntimeError("No sample images were found.")

    if not video_files:
        raise RuntimeError("No sample MP4 video was found.")

    return image_files, video_files


# ============================================================
# Perception
# ============================================================

def run_perception(model, image):
    if image is None:
        raise ValueError("Input image/frame is empty.")

    start_time = time.perf_counter()

    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
    )

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    result = results[0]

    detections = []

    if result.boxes is not None:
        for detection_id, box in enumerate(result.boxes):
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = [
                float(value)
                for value in box.xyxy[0].tolist()
            ]

            detections.append(
                {
                    "detection_id": detection_id,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "confidence": round(confidence, 4),
                    "bounding_box": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

    return {
        "raw_result": result,
        "detections": detections,
        "detection_count": len(detections),
        "inference_time_ms": round(inference_time_ms, 2),
    }


# ============================================================
# Reasoning
# ============================================================

def evaluate_helmet_compliance(detections):
    helmet_detections = [
        item
        for item in detections
        if item["class_name"] == "helmet"
    ]

    no_helmet_detections = [
        item
        for item in detections
        if item["class_name"] == "no_helmet"
    ]

    helmet_count = len(helmet_detections)
    no_helmet_count = len(no_helmet_detections)

    if no_helmet_count > 0:
        status = "VIOLATION"
        severity = "HIGH"
        reason = (
            f"{no_helmet_count} no_helmet detection(s) were found. "
            "At least one worker may not be wearing required "
            "head protection."
        )
        recommended_action = (
            "Save annotated evidence and generate a safety alert."
        )

    elif helmet_count > 0:
        status = "COMPLIANT"
        severity = "NONE"
        reason = (
            f"{helmet_count} helmet detection(s) were found and "
            "no no_helmet detections were identified."
        )
        recommended_action = (
            "Save the result as compliant evidence."
        )

    else:
        status = "REVIEW_REQUIRED"
        severity = "MEDIUM"
        reason = (
            "No helmet or no_helmet detections were found. "
            "The scene cannot be classified confidently."
        )
        recommended_action = (
            "Flag the input for manual review."
        )

    return {
        "status": status,
        "severity": severity,
        "reason": reason,
        "recommended_action": recommended_action,
        "counts": {
            "total_detections": len(detections),
            "helmet": helmet_count,
            "no_helmet": no_helmet_count,
        },
    }


# ============================================================
# Action / Output
# ============================================================

def create_action_outputs(
    source_name,
    original_image,
    perception_output,
    compliance_output,
):
    raw_result = perception_output["raw_result"]

    annotated_image = raw_result.plot()

    status = compliance_output["status"]

    cv2.putText(
        annotated_image,
        f"Status: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    annotated_path = (
        IMAGES_DIR
        / f"{RUN_ID}_{source_name}_annotated.jpg"
    )

    if not cv2.imwrite(
        str(annotated_path),
        annotated_image,
    ):
        raise RuntimeError(
            f"Could not save annotated image: {annotated_path}"
        )

    evidence_path = None

    if status in {"VIOLATION", "REVIEW_REQUIRED"}:
        evidence_path = (
            EVIDENCE_DIR
            / f"{RUN_ID}_{source_name}_{status.lower()}.jpg"
        )

        if not cv2.imwrite(
            str(evidence_path),
            annotated_image,
        ):
            raise RuntimeError(
                f"Could not save evidence image: {evidence_path}"
            )

    report_path = (
        REPORTS_DIR
        / f"{RUN_ID}_{source_name}_report.json"
    )

    report = {
        "run_id": RUN_ID,
        "created_at": datetime.now().isoformat(),
        "source": source_name,
        "model": {
            "model_path": str(MODEL_PATH),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "image_size": IMAGE_SIZE,
            "device": (
                "GPU"
                if DEVICE == 0
                else "CPU"
            ),
        },
        "perception": {
            "detection_count": (
                perception_output["detection_count"]
            ),
            "inference_time_ms": (
                perception_output["inference_time_ms"]
            ),
            "detections": (
                perception_output["detections"]
            ),
        },
        "reasoning": compliance_output,
        "outputs": {
            "annotated_image": str(annotated_path),
            "evidence_image": (
                str(evidence_path)
                if evidence_path
                else None
            ),
        },
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return {
        "annotated_image": str(annotated_path),
        "evidence_image": (
            str(evidence_path)
            if evidence_path
            else None
        ),
        "report": str(report_path),
    }


# ============================================================
# Batch image processing
# ============================================================

def process_sample_images(model, image_files):
    print("\n" + "=" * 60)
    print("BATCH IMAGE PROCESSING")
    print("=" * 60)

    records = []

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"SKIPPED unreadable image: {image_path.name}")
            records.append(
                {
                    "source": image_path.name,
                    "status": "PROCESSING_ERROR",
                    "error": "Image could not be read.",
                }
            )
            continue

        perception = run_perception(
            model,
            image,
        )

        reasoning = evaluate_helmet_compliance(
            perception["detections"]
        )

        outputs = create_action_outputs(
            source_name=image_path.stem,
            original_image=image,
            perception_output=perception,
            compliance_output=reasoning,
        )

        record = {
            "source": image_path.name,
            "status": reasoning["status"],
            "helmet_count": (
                reasoning["counts"]["helmet"]
            ),
            "no_helmet_count": (
                reasoning["counts"]["no_helmet"]
            ),
            "inference_time_ms": (
                perception["inference_time_ms"]
            ),
            "outputs": outputs,
        }

        records.append(record)

        print(
            f"{image_path.name}: "
            f"{reasoning['status']} "
            f"(helmet={reasoning['counts']['helmet']}, "
            f"no_helmet={reasoning['counts']['no_helmet']})"
        )

    return records


# ============================================================
# 10-scenario system evaluation
# ============================================================

def evaluate_video_scenarios(model, video_path):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video:\n{video_path}"
        )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = capture.get(cv2.CAP_PROP_FPS)

    frame_indices = np.linspace(
        0,
        max(total_frames - 2, 0),
        NUMBER_OF_SCENARIOS,
        dtype=int,
    )

    evaluation_records = []

    start_time = time.perf_counter()

    for scenario_id, frame_index in enumerate(
        frame_indices,
        start=1,
    ):
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        success, frame = capture.read()

        record = {
            "scenario_id": scenario_id,
            "frame_index": int(frame_index),
            "timestamp_seconds": (
                round(frame_index / fps, 2)
                if fps > 0
                else None
            ),
        }

        if not success or frame is None:
            record.update(
                {
                    "processed_successfully": False,
                    "status": "PROCESSING_ERROR",
                    "error": "Frame could not be read.",
                }
            )

            evaluation_records.append(record)
            continue

        try:
            perception = run_perception(
                model,
                frame,
            )

            reasoning = evaluate_helmet_compliance(
                perception["detections"]
            )

            annotated_frame = (
                perception["raw_result"].plot()
            )

            evaluation_image_path = (
                EVALUATION_IMAGES_DIR
                / (
                    f"scenario_{scenario_id:02d}"
                    f"_frame_{frame_index}.jpg"
                )
            )

            cv2.imwrite(
                str(evaluation_image_path),
                annotated_frame,
            )

            record.update(
                {
                    "processed_successfully": True,
                    "status": reasoning["status"],
                    "severity": reasoning["severity"],
                    "detection_count": (
                        perception["detection_count"]
                    ),
                    "helmet_count": (
                        reasoning["counts"]["helmet"]
                    ),
                    "no_helmet_count": (
                        reasoning["counts"]["no_helmet"]
                    ),
                    "inference_time_ms": (
                        perception["inference_time_ms"]
                    ),
                    "annotated_image_path": str(
                        evaluation_image_path
                    ),
                    "error": None,
                }
            )

        except Exception as error:
            record.update(
                {
                    "processed_successfully": False,
                    "status": "PROCESSING_ERROR",
                    "error": str(error),
                }
            )

        evaluation_records.append(record)

    capture.release()

    total_time = time.perf_counter() - start_time

    successful = [
        item
        for item in evaluation_records
        if item.get("processed_successfully")
    ]

    failed = [
        item
        for item in evaluation_records
        if not item.get("processed_successfully")
    ]

    inference_times = [
        item["inference_time_ms"]
        for item in successful
        if item.get("inference_time_ms") is not None
    ]

    summary = {
        "run_id": RUN_ID,
        "created_at": datetime.now().isoformat(),
        "scenarios_requested": NUMBER_OF_SCENARIOS,
        "scenarios_processed_successfully": len(successful),
        "processing_errors": len(failed),
        "processing_success_rate_percent": round(
            len(successful)
            / NUMBER_OF_SCENARIOS
            * 100,
            2,
        ),
        "compliant_scenarios": len(
            [
                item
                for item in successful
                if item["status"] == "COMPLIANT"
            ]
        ),
        "violation_scenarios": len(
            [
                item
                for item in successful
                if item["status"] == "VIOLATION"
            ]
        ),
        "review_required_scenarios": len(
            [
                item
                for item in successful
                if item["status"] == "REVIEW_REQUIRED"
            ]
        ),
        "average_inference_time_ms": round(
            sum(inference_times)
            / len(inference_times),
            2,
        )
        if inference_times
        else None,
        "total_evaluation_time_seconds": round(
            total_time,
            2,
        ),
        "scenarios": evaluation_records,
    }

    json_path = (
        METRICS_DIR
        / f"{RUN_ID}_final_evaluation.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    csv_path = (
        METRICS_DIR
        / f"{RUN_ID}_evaluation_scenarios.csv"
    )

    fieldnames = [
        "scenario_id",
        "frame_index",
        "timestamp_seconds",
        "processed_successfully",
        "status",
        "severity",
        "detection_count",
        "helmet_count",
        "no_helmet_count",
        "inference_time_ms",
        "annotated_image_path",
        "error",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in evaluation_records:
            writer.writerow(
                {
                    field: record.get(field)
                    for field in fieldnames
                }
            )

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    print(
        f"Scenarios requested: "
        f"{NUMBER_OF_SCENARIOS}"
    )
    print(
        f"Successfully processed: "
        f"{len(successful)}"
    )
    print(
        f"Processing errors: "
        f"{len(failed)}"
    )
    print(
        f"Success rate: "
        f"{summary['processing_success_rate_percent']:.2f}%"
    )

    return summary


# ============================================================
# Continuous monitoring
# ============================================================

def monitor_video(model, video_path):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open monitoring video:\n{video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    output_path = (
        VIDEOS_DIR
        / f"{RUN_ID}_ppe_monitored_video.mp4"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "Unable to create monitoring output video."
        )

    frame_index = 0
    analyzed_frames = 0

    status_counts = {
        "COMPLIANT": 0,
        "VIOLATION": 0,
        "REVIEW_REQUIRED": 0,
    }

    latest_status = "NOT_ANALYZED"

    records = []

    start_time = time.perf_counter()

    while True:
        success, frame = capture.read()

        if not success:
            break

        output_frame = frame.copy()

        if frame_index % FRAME_STRIDE == 0:
            perception = run_perception(
                model,
                frame,
            )

            reasoning = evaluate_helmet_compliance(
                perception["detections"]
            )

            output_frame = (
                perception["raw_result"].plot()
            )

            latest_status = reasoning["status"]

            status_counts[latest_status] += 1
            analyzed_frames += 1

            records.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": (
                        round(frame_index / fps, 2)
                        if fps > 0
                        else None
                    ),
                    "status": latest_status,
                    "helmet_count": (
                        reasoning["counts"]["helmet"]
                    ),
                    "no_helmet_count": (
                        reasoning["counts"]["no_helmet"]
                    ),
                    "inference_time_ms": (
                        perception["inference_time_ms"]
                    ),
                }
            )

        cv2.putText(
            output_frame,
            f"PPE Status: {latest_status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(output_frame)

        frame_index += 1

    capture.release()
    writer.release()

    processing_time = (
        time.perf_counter() - start_time
    )

    report = {
        "run_id": RUN_ID,
        "created_at": datetime.now().isoformat(),
        "input_video": str(video_path),
        "frame_stride": FRAME_STRIDE,
        "frames_written": frame_index,
        "frames_analyzed": analyzed_frames,
        "status_counts": status_counts,
        "processing_time_seconds": round(
            processing_time,
            2,
        ),
        "output_video": str(output_path),
        "analyzed_frames": records,
    }

    report_path = (
        REPORTS_DIR
        / f"{RUN_ID}_video_monitoring_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("CONTINUOUS VIDEO MONITORING")
    print("=" * 60)
    print(f"Frames written: {frame_index}")
    print(f"Frames analyzed: {analyzed_frames}")
    print(f"Output video: {output_path}")

    return report


# ============================================================
# Traceability
# ============================================================

def save_agent_trace(
    batch_results,
    evaluation_summary,
    monitoring_summary,
):
    trace = {
        "run_information": {
            "run_id": RUN_ID,
            "timestamp": datetime.now().isoformat(),
            "agent": "PPE Compliance Check Agent",
            "version": "1.0",
        },
        "configuration": {
            "model": str(MODEL_PATH),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "image_size": IMAGE_SIZE,
            "device": (
                "GPU"
                if DEVICE == 0
                else "CPU"
            ),
            "classes": CLASS_NAMES,
        },
        "pipeline": {
            "input_ingestion": True,
            "preprocessing": True,
            "perception": True,
            "reasoning": True,
            "action": True,
            "traceability": True,
        },
        "batch_image_results": batch_results,
        "evaluation_summary": {
            "scenarios_requested": (
                evaluation_summary[
                    "scenarios_requested"
                ]
            ),
            "scenarios_processed_successfully": (
                evaluation_summary[
                    "scenarios_processed_successfully"
                ]
            ),
            "processing_success_rate_percent": (
                evaluation_summary[
                    "processing_success_rate_percent"
                ]
            ),
        },
        "monitoring_summary": {
            "frames_written": (
                monitoring_summary[
                    "frames_written"
                ]
            ),
            "frames_analyzed": (
                monitoring_summary[
                    "frames_analyzed"
                ]
            ),
            "status_counts": (
                monitoring_summary[
                    "status_counts"
                ]
            ),
        },
        "completed": True,
    }

    trace_path = (
        TRACES_DIR
        / f"{RUN_ID}_agent_trace.json"
    )

    with open(
        trace_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            trace,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return trace_path


# ============================================================
# Main agent loop
# ============================================================

def main():
    print("=" * 60)
    print("PPE COMPLIANCE CHECK AGENT")
    print("=" * 60)

    create_output_directories()

    ensure_model_exists()

    image_files, video_files = validate_inputs()

    print(f"Repository: {PROJECT_DIR}")
    print(f"Sample images: {len(image_files)}")
    print(f"Sample videos: {len(video_files)}")
    print(f"Model: {MODEL_PATH}")

    global DEVICE
    global CLASS_NAMES

    DEVICE = 0 if torch.cuda.is_available() else "cpu"

    print(
        f"Device: "
        f"{'GPU' if DEVICE == 0 else 'CPU'}"
    )

    model = YOLO(str(MODEL_PATH))

    CLASS_NAMES = model.names

    print(f"Classes: {CLASS_NAMES}")

    # 1. Batch image ingestion -> perception -> reasoning -> action
    batch_results = process_sample_images(
        model,
        image_files,
    )

    # 2. Ten documented evaluation scenarios
    evaluation_summary = evaluate_video_scenarios(
        model,
        video_files[0],
    )

    # 3. Continuous video monitoring
    monitoring_summary = monitor_video(
        model,
        video_files[0],
    )

    # 4. Complete run trace
    trace_path = save_agent_trace(
        batch_results,
        evaluation_summary,
        monitoring_summary,
    )

    print("\n" + "=" * 60)
    print("AGENT RUN COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Run ID: {RUN_ID}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Agent trace: {trace_path}")
    print(
        "\nPerception -> Reasoning -> "
        "Action -> Traceability complete."
    )


if __name__ == "__main__":
    main()
