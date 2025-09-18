# Fourier Frequency-Domain Image Denoising (DND)

This repository demonstrates frequency-domain image filtering using the 2D Discrete Fourier Transform (DFT) for denoising and analysis on the **Darmstadt Noise Dataset (DND)**.
It includes:

* A reproducible pipeline to apply **Low-Pass**, **High-Pass**, **Band-Pass**, and **Band-Stop** filters.
* Batch processing over the DND sRGB set and saving filtered crops to `.mat`.
* Automatic generation of comparison report pages and optional charts (PSNR/SSIM vs. noisy input).
* A LaTeX report template to present theory, experiments, and results.

> ⚠️ Note: DND does not provide clean ground-truth images publicly. Metrics here compare **filtered outputs to the noisy image** to quantify structural changes, not true denoising accuracy.

---

## Repository Structure

```
.
├─ README.md
├─ .gitignore
├─ document.tex        # Full LaTeX document (Vietnamese)
├─ scripts/
│  ├─ run_denoising.py                    # One-time batch processing over DND
│  ├─ generate_full_report.py             # Stitch grid pages for all crops
│  └─ create_report.py                    # Pick examples + PSNR/SSIM charts
└─                

You can rename/move files as you wish; the code currently uses **Windows-style absolute paths** in examples. See **Paths & Folders** below.

---

## Features

* **Fourier filters** with circular masks centered in the shifted spectrum:

  * Low-Pass (LPF): smoothing / denoising (with detail loss).
  * High-Pass (HPF): edge emphasis (not a denoiser).
  * Band-Pass (BPF): mid-frequency textures.
  * Band-Stop (BSF): remove periodic bands.
* **Batch runner** over 50 images × 20 crops (1,000 patches).
* **Visualization**:

  * Full dataset report pages (`Report_Full_Page_*.png`).
  * Example grid figure and **PSNR/SSIM** bar charts.

---

## Installation

Python ≥ 3.9 recommended.

```bash
# (Optional) Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If you don’t use `requirements.txt`, install directly:

```bash
pip install numpy scipy h5py opencv-python matplotlib scikit-image
```

---

## Dataset: DND (sRGB)

1. Download the **DND sRGB** dataset from the official source and extract it.
2. Ensure the folder contains `images_srgb/*.mat` and `info.mat`.

> This project reads `.mat` files with `h5py`; no further conversion is needed.

---

## Paths & Folders

The scripts currently show **Windows examples** (e.g., `F:\dnd_2017`).
Change these variables in each script to match your environment:

* `scripts/run_denoising.py`

  * `data_folder = r'F:\dnd_2017'`
  * Output folders in `filters_to_run` (e.g., `F:\output_low_pass`)

* `scripts/generate_full_report.py`

  * `data_folder = r'F:\dnd_2017'`
  * Output read paths in `filter_outputs`
  * Report pages saved to `F:\Report_Full_Page_*.png` (change as needed)

* `scripts/create_report.py`

  * `data_folder = r'F:\dnd_2017'`
  * Output read paths in `filter_outputs`
  * Chart/report save paths (e.g., `F:\metrics_*.png`, `F:\Fourier_Filters_Report_4_Examples.png`)

> Tip: You can refactor these into a single `config.yaml` or use environment variables later.

---

## Usage

### 1) Batch Process (run once)

This applies all filters and writes `.mat` results for each crop.

```bash
python scripts/run_denoising.py
```

Expected outputs:

```
F:\output_low_pass\0001_01.mat, ..., 0050_20.mat
F:\output_high_pass\...
F:\output_band_pass\...
F:\output_band_stop\...
```

### 2) Generate Full Report Pages (all patches)

```bash
python scripts/generate_full_report.py
```

This stitches comparison grids into page images:

```
F:\Report_Full_Page_1.png
F:\Report_Full_Page_2.png
...
```

### 3) Custom Report + Metrics (select examples)

```bash
python scripts/create_report.py
```

Produces:

* PSNR/SSIM bar chart(s) for a chosen `(img_id, box_id)`.
* A single comparative grid image for your selected examples.

---

## Notes on Metrics

* **PSNR** and **SSIM** are computed **against the original noisy crop**, because DND ground-truth is not accessible.
* Interpreting results:

  * **LPF**: typically larger deviation (lower SSIM to noisy image) but visually denoises; edges/details get smoothed.
  * **HPF/BPF**: emphasize structure/edges; appear darker; not meant for denoising.
  * **BSF**: minimal alteration unless periodic noise exists; SSIM often highest vs. noisy reference.

---

## LaTeX Report

A complete LaTeX document (Vietnamese) is provided in `latex_report/report_fourier_denoising.tex`.
It covers:

* Theory of 2D DFT/IDFT and frequency masks.
* Experiment design, results, and discussion.
* Auto-inclusion of generated report pages.

Compile with your LaTeX toolchain (XeLaTeX recommended for Vietnamese fonts).

---

## Troubleshooting

* **`File not found` / missing `.mat`:** Make sure you ran `run_denoising.py` first and updated paths in all scripts.
* **`info.mat` not found:** Verify your DND directory structure and that `data_folder` points to it.
* **Windows path issues:** Prefer raw strings (`r'C:\path\to\folder'`) or forward slashes.
* **Matplotlib displays blank:** If running headless, remove `plt.show()` or use `matplotlib` non-interactive backends.

---

## References

* Plötz, T. and Roth, S., “Benchmarking Denoising Algorithms with Real Photographs,” *CVPR*, 2017.
* Gonzalez, R. C., and Woods, R. E., *Digital Image Processing*, 4th ed., Pearson, 2018.

```bibtex
@inproceedings{plotz2017benchmark,
  title={Benchmarking Denoising Algorithms with Real Photographs},
  author={Pl{\"o}tz, Tobias and Roth, Stefan},
  booktitle={CVPR},
  year={2017}
}
```

## Acknowledgements

* DND authors for the dataset and evaluation framework inspiration.
* Open-source community for `numpy`, `scipy`, `h5py`, `opencv-python`, `matplotlib`, and `scikit-image`.
