# bss-demo

Demo of detecting and counting Black Spot Syndrome (BSS) spots on surgeonfish (*Acanthurus tractus*) with YOLO.

- `labels_bss-surgeon-1_25_2026-09-22-11-58-55/`: expert spot annotations for 24 iNaturalist observations, exported from [MakeSense](https://www.makesense.ai/) in YOLO format. Each file is named by its iNaturalist observation ID.
- `bss_yolo_demo.ipynb`: validates the labels, downloads the matching photos from iNaturalist, visualises the boxes, trains a small YOLO11 detector, and scores spot counts against the project's targets.
- `bsd-demo.md`: the original questions about annotation format and image resizing. The answers are at the end of the notebook.

## Run

```bash
pip install ultralytics opencv-python pandas matplotlib requests pyyaml
jupyter lab bss_yolo_demo.ipynb
```

Photos are downloaded at run time and are not committed. They belong to their iNaturalist authors and are under mixed licenses, some of them all rights reserved.
