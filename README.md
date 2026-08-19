---
title: RoadScan
emoji: 🛣️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# road_model — Road Damage Detection PoC

Pretrained-first PoC for detecting road damage (cracks, potholes) on dashcam-style imagery.
Includes a CLI (`infer.py`) and a Gradio web UI (`app.py`) that deploys 1-click to Hugging Face Spaces.

> The YAML block above is **only** read by Hugging Face Spaces. It's safe to ignore locally; if you fork to your own Space, change `title`/colors as you like.

## What's here

```
road_model/
├── app.py                       Gradio UI — image + video upload, annotated output
├── infer.py                     CLI — runs detection on an image, dir, or video
├── prep_samples.py              splits the mosaic into samples/frames/
├── requirements.txt             pip deps (CPU-friendly: opencv-python-headless)
├── models/YOLOv8_Small_RDD.pt   auto-downloaded on first run (~86 MB)
├── samples/
│   ├── val_batch_labels.jpg     16-image RDD2022 mosaic w/ ground-truth boxes
│   ├── val_batch_pred.jpg       same mosaic w/ original-author predictions
│   └── frames/                  the mosaic split into 16x 480x480 frames
└── runs/poc1/                   CLI inference outputs
```

## Classes

The pretrained weights detect 4 classes (no patch/repair class):

| ID | Name               | RDD code |
|----|--------------------|----------|
| 0  | Longitudinal Crack | D00      |
| 1  | Transverse Crack   | D10      |
| 2  | Alligator Crack    | D20      |
| 3  | Potholes           | D40      |

Source: [oracl4/RoadDamageDetection](https://github.com/oracl4/RoadDamageDetection).

## Run locally

Uses the existing `person_detector/.venv` (ultralytics 8.4.51, torch 2.6 + CUDA).

```bash
PY=../person_detector/.venv/bin/python

# (one-time) split the sample mosaic into individual frames
$PY prep_samples.py

# Option A — CLI on a directory / image / video
$PY infer.py
$PY infer.py --src /path/to/your/dashcam_video.mp4 --conf 0.25

# Option B — web UI (image + video upload in browser)
$PY app.py
# then open http://127.0.0.1:7860
```

## Deploy the web UI to Hugging Face Spaces (free)

Free CPU Space, no card required. The model auto-downloads on first start, so no Git LFS needed.

1. Create a free account at https://huggingface.co/join.
2. Create a new Space: https://huggingface.co/new-space
   - **Owner:** your username
   - **Space name:** e.g. `road-damage-poc`
   - **License:** MIT (or whichever you prefer)
   - **SDK:** Gradio
   - **Hardware:** CPU basic — free
   - Visibility: public or private (your choice)
3. On your local machine, clone the empty Space repo and copy these files in:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/road-damage-poc
   cd road-damage-poc
   cp ../models_development/road_model/{app.py,requirements.txt,README.md,.gitignore} .
   git add app.py requirements.txt README.md .gitignore
   git commit -m "Initial PoC"
   git push
   ```
4. Wait ~2–4 minutes for the Space to build (you can watch the build log in the Space UI).
5. First request triggers the weights download (~30 s once). Subsequent requests are fast.

### Performance notes on free Spaces

- **CPU only.** Per-image inference: ~2–4 s. A 10-second 30 fps dashcam clip = ~10–15 minutes to process.
- For real-time-ish video, upgrade the Space to **T4 small** (paid) or run locally with CUDA.
- The Space sleeps after ~48 h of inactivity; first hit after sleep takes ~30 s to wake.

## Known PoC gaps

- **No patch/repair class** in these weights. RDD2022's Norway/India subsets include D43/D44 patch labels — would require fine-tuning.
- **No "faded markings"** — that's lane-marking-segmentation, not in scope of RDD2022.
- Weights were trained mostly on Japan + India dashcam footage; expect quality drift on different camera angles or local road materials.

## Next steps (if pretrained quality is not enough)

1. Drop your own ~20 dashcam frames into `samples/yours/` and re-run `infer.py --src samples/yours`.
2. If results are weak: fine-tune on full RDD2022 (download via `sekilab/RoadDamageDetector` or HuggingFace mirror), reusing the `person_detector/scripts/` training scaffolding.
