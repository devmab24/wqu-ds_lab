# U-Net Crack Segmentation

A computer vision project for pixel-level crack segmentation using a U-Net architecture with a ResNet34 encoder pretrained on ImageNet. The model learns to identify crack regions in images and produce binary segmentation masks.

> **Project status:** Baseline model trained and evaluated in Google Colab. The reported results are from the notebook's validation workflow; further testing is needed before any safety-critical or production use.

## Overview

Crack detection is useful for inspecting surfaces and infrastructure. Unlike image-level classification, semantic segmentation predicts a label for each pixel, making it possible to localize the regions identified as cracks.

This project trains a U-Net model on paired images and masks, evaluates predictions on a held-out validation split, and visualizes predicted masks and overlays.

## Results

The notebook reports the following results after 20 training epochs:

| Metric | Result |
|---|---:|
| Training loss (epoch 20) | 0.0334 |
| Validation loss (epoch 20) | 0.0322 |
| Dice score (epoch 20) | 0.8293 |
| Crack-class precision | 0.85 |
| Crack-class recall | 0.83 |
| Crack-class F1-score | 0.84 |
| Images | 800 |
| Masks | 800 |
| Train/validation split | 80% / 20% |

The mask inspection found **504 masks containing crack pixels** and **296 empty masks**. The pixel-level classification report was generated over the validation set. Pixel accuracy is not emphasized here because the background class dominates the number of pixels and can make accuracy look high even when crack segmentation needs improvement.

Metrics are specific to the dataset, split, preprocessing, threshold, and notebook run. They should not be interpreted as evidence of generalization to unseen sites or real-world inspection conditions.

## Model and Method

- **Task:** Binary semantic segmentation (crack vs. background)
- **Architecture:** U-Net
- **Encoder:** ResNet34 with ImageNet-pretrained weights
- **Framework:** PyTorch and `segmentation-models-pytorch`
- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam
- **Learning rate:** `1e-4`
- **Epochs:** 20
- **Batch size:** 8
- **Validation split:** 20%, using `random_state=42`
- **Prediction threshold:** sigmoid probability greater than `0.5`
- **Image preprocessing:** RGB conversion and scaling pixel values to `[0, 1]`
- **Mask preprocessing:** grayscale mask converted to a binary mask (`mask > 0`)
- **Augmentation in notebook:** horizontal flip, vertical flip, and 90-degree rotations

## Dataset

The notebook expects paired image and mask files stored in Google Drive. The dataset itself is not included in this repository.

Expected directory layout:

```text
cracks_with_mask_dataset/
├── images/
│   ├── image_001.png
│   └── ...
└── masks/
    ├── image_001.png
    └── ...
```

Update the paths in the notebook to match your environment. Each image must be paired with its corresponding mask, and the image and mask lists must be aligned consistently. Before training, verify filenames and visually inspect image-mask pairs.

**Dataset access/source:** Add the dataset source or access instructions here if you are permitted to share them. Do not publish private or restricted data.

## Getting Started

### Option 1: Run in Google Colab

1. Upload `Aura_X_crack_Segmentation(with_mask).ipynb` to Google Colab.
2. Make the dataset available in your Google Drive using the directory structure described above.
3. Update `IMAGE_DIR` and `MASK_DIR` in the notebook.
4. Select a GPU runtime if one is available.
5. Run the notebook cells in order to inspect the data, train the model, evaluate predictions, and visualize segmentation outputs.

### Option 2: Install dependencies locally

Use Python 3.10 or 3.11 in a virtual environment, then install the packages used by the notebook:

```bash
pip install torch torchvision segmentation-models-pytorch albumentations opencv-python scikit-learn matplotlib seaborn tqdm numpy
```

The notebook uses Google Colab's `google.colab.drive` integration. For local execution, replace the Drive-mount and Drive paths with local filesystem paths before running it.

## Training Workflow

1. Load image and mask paths.
2. Inspect sample image-mask pairs and check for empty masks.
3. Split the dataset into training and validation sets.
4. Apply paired geometric transformations to images and masks.
5. Initialize U-Net with a ResNet34 encoder.
6. Train using `BCEWithLogitsLoss` and Adam.
7. Track training loss, validation loss, and Dice score.
8. Evaluate pixel-level predictions with a confusion matrix and classification report.
9. Visualize the input image, ground-truth mask, predicted mask, and crack overlay.

## Outputs

The notebook saves model files and a metrics JSON file to a Google Drive directory configured in the notebook. The intended artifacts include:

- `unet_crack_baseline_v1.pth` — model state dictionary
- `unet_crack_baseline_full.pt` — serialized model object
- `baseline_metrics.json` — recorded baseline metrics

These artifacts are not included in this repository by default. Upload model weights only if they are appropriate to share and repository size/licensing constraints allow it. For broader compatibility, prefer distributing a state dictionary with documented model configuration rather than relying only on a serialized Python model object.

## Limitations and Next Steps

This is a baseline experiment, not a validated inspection system. Recommended next steps:

- **Verify pairing:** Match image and mask files by filename or a manifest rather than relying only on independently sorted file lists.
- **Separate augmentation:** Apply random augmentation to the training set only. In the current notebook, the same dataset class applies random flips and rotations to the validation set as well, so validation is stochastic.
- **Checkpoint the best model correctly:** The training loop saves the best checkpoint as `best_unet_model.pth`, but the later Drive-saving cell saves the current model state. Confirm that the exported file corresponds to the best validation checkpoint before reporting or deploying it.
- **Strengthen evaluation:** Add a held-out test set, report per-image metrics and aggregate metrics, and inspect false positives and false negatives.
- **Test generalization:** Evaluate on images from different surfaces, lighting conditions, cameras, and locations.
- **Improve reproducibility:** Record package versions, random seeds, dataset provenance, and the exact checkpoint used for reported results.
- **Assess practical performance:** Measure inference time and memory requirements and define an acceptable error rate for the intended application.

## Repository Structure

A suggested repository layout:

```text
unet-crack-segmentation/
├── Aura_X_crack_Segmentation(with_mask).ipynb
├── README.md
├── requirements.txt        # optional: pinned dependencies
├── assets/                 # optional: approved sample visualizations
└── models/                 # optional: shareable model weights
```

Only add sample images, dataset files, or trained weights when you have permission to redistribute them.

## Reproducibility Note

The notebook was developed in Google Colab and uses paths specific to the author's Google Drive. Re-running it requires access to the dataset and updating those paths. Exact results may vary with package versions, runtime hardware, and random initialization.

## Author

**Muhammad Ayuba Bello**  
AI/ML Engineer | Data Scientist | Computer Vision

- GitHub: Add your GitHub profile or repository URL
- LinkedIn: Add your LinkedIn profile URL

---

*This README describes the implementation and results recorded in the accompanying notebook. Add repository links, dataset attribution, and approved visual examples before making the project public.*
