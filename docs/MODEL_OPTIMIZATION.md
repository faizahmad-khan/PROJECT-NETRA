# Model Optimization Guide (Edge + Laptop)

This guide helps you run NETRA efficiently on constrained devices while keeping detection quality stable.

## 1) Fastest Win: Use a Lighter Base Model

Use `yolov8n.pt` for edge deployments instead of `yolov8m.pt`.

- Better FPS
- Lower memory
- Slight drop in accuracy (usually acceptable for demos/real-time use)

## 2) Runtime Tuning (No Re-Training Needed)

Use CLI flags in `main.py`:

```bash
python main.py --traffic-model models/yolov8n.pt --device auto --imgsz 512 --skip-frames 1
```

Recommended tuning order:
1. Switch model `yolov8m -> yolov8n`
2. Reduce `--imgsz` to `512`, then `416` if needed
3. Add `--skip-frames 1` or `2` if FPS is still low

## 3) FP16 / INT8 / TensorRT

## FP16
- Works best on NVIDIA GPUs
- Smaller memory footprint
- Faster inference with minimal quality loss

## INT8
- Best for maximum speed/efficiency
- Requires calibration data for good accuracy

## TensorRT (NVIDIA)
- Optimizes model graph + kernels for NVIDIA hardware
- Strong option for Jetson devices

## 4) Export Commands

### ONNX
```bash
python src/export_model.py --model models/yolov8n.pt --format onnx --imgsz 640
```

### CoreML (Mac M-series)
```bash
python src/export_model.py --model models/yolov8n.pt --format coreml --imgsz 640
```

### TensorRT FP16 (Jetson/NVIDIA)
```bash
python src/export_model.py --model models/yolov8n.pt --format engine --half --device cuda:0 --imgsz 640
```

### TensorRT INT8 (Jetson/NVIDIA)
```bash
python src/export_model.py --model models/yolov8n.pt --format engine --int8 --device cuda:0 --imgsz 640 --data coco8.yaml
```

## 5) Device-Specific Recommendations

## MacBook M2
- Start with `--device mps`
- Use `yolov8n.pt`
- Keep FP32 at runtime unless validated otherwise
- Consider CoreML export for deployment experiments

## Jetson Nano / Xavier / Orin
- Use TensorRT engine export
- Prefer FP16 first, then INT8 after calibration
- Keep input size modest (`512` or `416`) for stable FPS

## 6) Validation Checklist

After every optimization change, verify:
1. FPS or end-to-end latency improves
2. Ambulance recall does not collapse
3. False positives stay acceptable
4. Thermals and memory usage are stable for long runs
