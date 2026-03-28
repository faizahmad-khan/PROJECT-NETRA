"""Export YOLO models to optimized formats (ONNX/CoreML/TensorRT)."""

from __future__ import annotations

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO model for efficient edge inference."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to .pt model file",
    )
    parser.add_argument(
        "--format",
        default="onnx",
        choices=["onnx", "coreml", "engine", "openvino"],
        help="Export target format",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size (single square dimension)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Export device: auto, cpu, mps, or cuda:0",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 export when supported",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 export (mainly for TensorRT/OpenVINO)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Dataset YAML path for INT8 calibration (recommended)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading model:", args.model)
    model = YOLO(args.model)

    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "int8": args.int8,
        "device": args.device,
    }
    if args.data:
        export_kwargs["data"] = args.data

    print("Export options:", export_kwargs)
    output = model.export(**export_kwargs)

    print("\nExport complete")
    print("Output:", output)


if __name__ == "__main__":
    main()
