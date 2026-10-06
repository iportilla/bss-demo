# bss-demo

A first look at detecting and counting **Black Spot Syndrome (BSS)** spots on ocean surgeonfish (*Acanthurus tractus*) with YOLO, using the first batch of expert annotations.

*Work summary: 2026-10-06*

## Contents

| Path | What it is |
|---|---|
| `labels_bss-surgeon-1_25_2026-09-22-11-58-55/` (+ `.zip`) | Rylee's spot annotations for 24 solitary-fish photos, exported from [MakeSense](https://www.makesense.ai/) in YOLO format |
| `bss_yolo_demo.ipynb` | End-to-end demo: validate labels → download photos → visualise → train YOLO → score spot counts |
| `bsd-demo.md` | The original questions (is the format OK? should we resize?) |

## Run it

```bash
pip install ultralytics opencv-python pandas matplotlib requests pyyaml
jupyter lab bss_yolo_demo.ipynb
```

The notebook downloads the photos from iNaturalist at run time. They are **not committed** because they belong to their iNaturalist authors under mixed licenses (CC-BY, CC-BY-NC, CC-BY-NC-ND, and 8 marked all rights reserved). Training takes about 18 minutes on an Apple M4 CPU.

## What the annotations look like

- **Format:** one `.txt` per image, one line per spot: `class x_center y_center width height`, with coordinates normalised to 0–1. There's a single class (`0` = spot). MakeSense doesn't export a class-names file, so the notebook writes its own `data.yaml`.
- **Image link:** the label filenames are **iNaturalist observation IDs**. All 24 resolve to *A. tractus*, observed across the Caribbean (Bonaire, Curaçao, Belize, Mexico, Cuba, USVI, and others).
- **Volume:** 220 boxes. Counts per image range from 1 to 44 (median 3): nine images have a single spot, and three have more than 35.
- **Spot size:** spots are tiny. The median box is **1.5% × 2.0%** of the image, and the largest is 3.7% wide. At YOLO's default 640 px input, that's about 10 px per spot.
- **Photo sizes:** mixed, from 745 to 2048 px wide, with varying aspect ratios.

### Issues found in the labels

1. **Two zero-size boxes** (width or height = 0): one in `13219574.txt`, one in `3137185.txt`. They're probably stray clicks and should be deleted in MakeSense.
2. **No negatives:** every image contains spots. The model has never seen an uninfected fish or an NA fish.
3. **Ambiguous photo choice:** 3 observations (`12454226`, `14461486`, `18853302`) have more than one photo. The notebook assumes the first one was annotated. Store the exact photo ID or file with each label going forward.

## Model results

Setup: YOLO11n (COCO-pretrained), `imgsz=1280`, 40 epochs on CPU, 18 training images and 6 held-out images. The held-out images were chosen to cover the full range of spot counts. Spot count = number of detections above a confidence threshold. The threshold (0.10) was tuned on the training images.

**Detection (held-out):** precision 0.40, recall 0.29, mAP50 0.29, mAP50-95 0.11.

**Spot counts (held-out):**

| Observation | True spots | Predicted | Error |
|---|---:|---:|---:|
| 19517290 | 1 | 3 | +2 |
| 22281742 | 1 | 3 | +2 |
| 18853302 | 2 | 0 | −2 |
| 4787366 | 4 | 5 | +1 |
| 791218 | 11 | 4 | −7 |
| 3137185 | 37 | 14 | −23 |

| Metric | Result | Target (requirements doc) | Met? |
|---|---:|---:|:---:|
| MAE | 6.17 | ≤ 2 | ❌ |
| RMSE | 9.92 | ≤ 3 | ❌ |
| R² | 0.40 | ≥ 0.80 | ❌ |
| Exact count | 0% | ≥ 80% | ❌ |
| Within ±1 | 17% | ≥ 90% | ❌ |

**How to read this:** with 18 training images, missing the targets is expected. The run shows the annotations work end to end, not what the approach can achieve. The error pattern is informative, though. The model **over-counts lightly infected fish** (it confuses other dark markings with spots) and **badly under-counts heavily infected fish**. Both problems should improve with more data, negative examples, and higher effective resolution per spot.

## Answers to the questions in `bsd-demo.md`

**Is MakeSense + YOLO format + one rectangle per spot a reasonable approach?**
Yes. It's the native format for Ultralytics YOLO and converts easily to other formats. One box per spot gives detection *and* counting from a single model, and the export worked with no conversion.

**Should we resize all images to the same size before annotating?**
No. Annotate the **full-resolution originals**. YOLO coordinates are normalised, so labels stay valid at any image size, and the training pipeline resizes automatically. Downscaling first would destroy detail on spots that are already only 1–3% of the image width. Handle size at *training* time with a large `imgsz`, fish crops, or tiling. Caveat: if images are cropped after annotation, the labels have to be regenerated.

## Recommended next steps

1. **Annotate more images**, including **uninfected fish** (empty label file) and **NA fish**. Natural markings and scars are especially useful as hard negatives.
2. **Write a short spot-boxing guideline** (how to treat touching or merged spots, minimum size) and **double-annotate about 10%** of images. Annotator agreement is the human baseline behind the "as good as a human" target.
3. **Fix the two zero-size boxes**, and store the exact photo ID or file with each label.
4. **Move to a two-stage pipeline**, as the requirements doc proposes: detect or crop each fish (and assign Y/N/NA quality), then count spots on the crop. This handles multi-fish images and makes spots larger relative to the model input. Tiled inference (e.g. SAHI) is an alternative.
5. **Train on a GPU** for more epochs. Mac `mps` training crashed in our tests (Ultralytics 8.3 bug), so the notebook defaults to CPU.

## Notes from today

- An early notebook run reported **0 spots on every image**. The cause was two copies of the notebook (one headless, one in VS Code) training at the same time and writing to the same `runs/bss_demo` folder. The notebook now gives each training run its own folder.
