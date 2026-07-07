# Watermark Forging - Team IV

This repository contains our solution for the Trustworthy Machine Learning Assignment 4: Watermark Forgery Attack.

## Best Public Leaderboard Result

Our best public leaderboard score was obtained with the following experiment:

```text
nlm_h12_pos_a016
```

Final public leaderboard score:

```text
0.476498
```

The method uses a denoising-based residual template extraction pipeline with Non-Local Means (NLM) denoising. The best configuration used NLM with denoising strength `h = 12` and base watermark injection strength `alpha = 0.16`.

## Method Summary

For each watermark group, the watermark residual is estimated from the 25 available watermarked source images.

For each source image, we compute:

```text
residual = watermarked_image - denoised_watermarked_image
```

The residuals from the 25 images in the same watermark group are aggregated using the median to obtain a group-wise watermark template. This template is then injected into the corresponding clean target images using a controlled alpha value and a detail mask.

The final best experiment uses:

```text
denoiser      = nlm_h12
base_alpha    = 0.16
min_alpha     = 0.10
max_alpha     = 0.30
mask_floor    = 0.25
mask_ceiling  = 1.00
sharpen_mode  = signed
```

## Repository Structure

```text
watermark_forging/
├── Dataset/
│   ├── watermarked_sources/
│   └── clean_targets/
├── outputs/
├── src/
│   └── residual_template/
│       ├── config_residual.py
│       ├── denoisers.py
│       ├── template_extraction.py
│       ├── forgery.py
│       ├── pipeline.py
│       ├── main_residual.py
│       ├── create_zip_residual.py
│       └── evaluate_quality_residual.py
├── task_template.py
├── submission.py
└── README.md
```

## Requirements

Install the required Python packages:

```bash
pip install numpy opencv-python pillow
```

If you want to run local LPIPS-based quality evaluation, install the additional packages required by your evaluation script:

```bash
pip install torch torchvision lpips
```

## Dataset Setup

The expected dataset structure is:

```text
Dataset/
├── watermarked_sources/
│   ├── WM_1/
│   ├── WM_2/
│   ├── WM_3/
│   ├── WM_4/
│   ├── WM_5/
│   ├── WM_6/
│   ├── WM_7/
│   └── WM_8/
└── clean_targets/
    ├── 1.png
    ├── 2.png
    └── ...
```

Each watermark group `WM_1` to `WM_8` should contain 25 watermarked source images. The `clean_targets` folder should contain 200 clean images named `1.png` to `200.png`.

The target mapping used by the code is:

```text
WM_1 -> clean target images 1.png to 25.png
WM_2 -> clean target images 26.png to 50.png
WM_3 -> clean target images 51.png to 75.png
WM_4 -> clean target images 76.png to 100.png
WM_5 -> clean target images 101.png to 125.png
WM_6 -> clean target images 126.png to 150.png
WM_7 -> clean target images 151.png to 175.png
WM_8 -> clean target images 176.png to 200.png
```

## Reproducing the Best Result

From the project root, run:

```bash
python src/residual_template/main_residual.py --experiments nlm_h12_pos_a016
```

This generates the forged images for the best experiment under the outputs directory.

Then create the final submission ZIP:

```bash
python src/residual_template/create_zip_residual.py --experiment nlm_h12_pos_a016
```

The submission file will be created at:

```text
outputs/submission.zip
```

The ZIP file contains exactly 200 PNG images named:

```text
1.png, 2.png, ..., 200.png
```

There are no subfolders inside the ZIP.

## Optional: Quality Evaluation

To compute local visual quality metrics for the best experiment, run:

```bash
python src/residual_template/evaluate_quality_residual.py --experiments nlm_h12_pos_a016
```

This reports LPIPS, quality score, PSNR, and mean absolute difference for the generated forged images.

## Final Submission

Submit the generated ZIP file:

```text
outputs/submission.zip
```

using the provided `submission.py` script after inserting the team API key and setting the submission path correctly.

## GitHub Repository

```text
https://github.com/manuviswakarmave/watermark_forging.git
```
