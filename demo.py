#!/usr/bin/env python3
"""
demo.py
-------
Single end-to-end command demonstrating the Phase 1 pipeline:

    MODEL + IMAGE(S) -> INFERENCE -> STRUCTURED, SAVED RESULTS

Usage
-----
Run with the bundled offline sample model + sample images (default):

    python demo.py

Run with real pretrained COCO weights (requires internet access):

    python demo.py --pretrained

Run against your own model checkpoint and images:

    python demo.py --model path/to/weights.pth --images path/to/images \\
                    --architecture fasterrcnn_resnet50_fpn

See README.md for the full list of options and expected output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running the demo directly from the project root without installing
# the package first.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cv_auditor import (  # noqa: E402
    PipelineConfig,
    ModelLoader,
    ImageLoader,
    InferenceEngine,
    ResultFormatter,
)
from cv_auditor.config import SUPPORTED_ARCHITECTURES  # noqa: E402
from cv_auditor.utils import get_logger  # noqa: E402

logger = get_logger("demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 CV ingestion/inference demo")
    parser.add_argument(
        "--model", default="data/sample_model/sample_model.pth",
        help="Path to a model checkpoint (.pth state_dict) or TorchScript file (.pt). "
             "Ignored if --pretrained is set without --model.",
    )
    parser.add_argument(
        "--pretrained", action="store_true",
        help="Download and use torchvision's official pretrained COCO weights "
             "instead of a local checkpoint. Requires internet access.",
    )
    parser.add_argument(
        "--architecture", default="fasterrcnn_mobilenet_v3_large_320_fpn",
        choices=SUPPORTED_ARCHITECTURES,
        help="Detection architecture to instantiate.",
    )
    parser.add_argument(
        "--images", default="data/sample_images",
        help="Path to a single image or a directory of images.",
    )
    parser.add_argument(
        "--output", default="results/inference_results.json",
        help="Where to write the structured JSON results.",
    )
    parser.add_argument(
        "--model-name", default="reference_model",
        help="Human-readable identifier recorded with every prediction.",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.0,
        help="Drop predictions below this score. Default 0.0 keeps everything, "
             "which is useful with the (untrained) bundled sample model.",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device to run inference on.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = None if args.pretrained else args.model
    config = PipelineConfig(
        model_path=model_path,
        architecture=args.architecture,
        pretrained=args.pretrained,
        model_name=args.model_name,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        input_dir=args.images,
        output_path=args.output,
    )

    logger.info("=== Phase 1 CV Ingestion & Inference Demo ===")
    logger.info("Config: %s", json.dumps(config.to_dict(), indent=2, default=str))

    # 1. Load model
    try:
        model, model_metadata = ModelLoader(config).load()
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return 1

    # 2. Load images
    try:
        images = ImageLoader(config.input_dir).load_all()
    except Exception as exc:
        logger.error("Failed to load images: %s", exc)
        return 1

    # 3. Run inference
    engine = InferenceEngine(
        model=model,
        model_name=config.model_name,
        class_names=config.class_names,
        device=next(model.parameters()).device,
        confidence_threshold=config.confidence_threshold,
    )
    image_results = engine.run_batch(images)

    # 4. Format + save results
    formatter = ResultFormatter(
        model_metadata=model_metadata.to_dict(),
        config=config.to_dict(),
    )
    for result in image_results:
        formatter.add_result(result)
    output_path = formatter.save(config.output_path)

    # 5. Summary for the user
    total_predictions = sum(len(r.predictions) for r in image_results)
    print("\n--- Summary ---")
    print(f"Model:           {model_metadata.model_name} ({model_metadata.architecture}, source={model_metadata.source})")
    print(f"Images processed: {len(image_results)}")
    print(f"Total predictions kept (>= threshold {config.confidence_threshold}): {total_predictions}")
    print(f"Results saved to: {output_path}")
    if model_metadata.source == "random_init":
        print(
            "\nNOTE: the bundled sample model has randomly initialized weights "
            "(no internet access was available to download pretrained COCO "
            "weights in this environment). Predictions above are real model "
            "output but are NOT meaningful detections. Re-run with "
            "--pretrained (with internet access) or point --model at a "
            "trained checkpoint to see meaningful results."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
