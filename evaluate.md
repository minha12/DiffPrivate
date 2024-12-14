# Evaluation Guide

## Dataset Preparation

Before running the evaluation, you can choose between full datasets or smaller subsets for initial experiments:

### FFHQ Dataset
#### Full Dataset (70,000 images)
1. Download the dataset from [Google Drive](https://drive.google.com/file/d/1KUxJ-G6CBFzYpeg4PfTL93N8YybNExA7/view?usp=drive_link)
2. Create the directory and unzip the dataset:
   ```bash
   mkdir -p ./data/ffhq
   unzip ffhq_256.zip -d ./data/ffhq
   ```

#### Subset (200 images)
1. Download the subset from [Google Drive](https://drive.google.com/file/d/1jmX5WZK5Zyuod7VBRZ87kzNiTBMscmZG/view?usp=drive_link)
2. Create the directory and unzip the dataset:
   ```bash
   mkdir -p ./data/ffhq_256_subset
   unzip ffhq_256_subset.zip -d ./data/ffhq_256_subset
   ```

### CelebA-HQ Dataset
#### Full Dataset (30,000 images)
1. Download the dataset from [Google Drive](https://drive.google.com/file/d/1hKK99bKgH9UzmdNcDwWYwxpZUqkoo0Lj/view?usp=drive_link)
2. Create the directory and unzip the dataset:
   ```bash
   mkdir -p ./data/celeba_hq
   unzip celeba_hq_256.zip -d ./data/celeba_hq
   ```

#### Subset (200 images)
1. Download the subset from [Google Drive](https://drive.google.com/file/d/1XXqxCetWZipHbmvIfMUU4dUWK02xgCl4/view?usp=drive_link)
2. Create the directory and unzip the dataset:
   ```bash
   mkdir -p ./data/celeba_hq_256_subset
   unzip celeba_hq_256_subset.zip -d ./data/celeba_hq_256_subset
   ```

### Processing Time Estimates
Using an NVIDIA A100 GPU:
- Single image processing: 20-60 seconds
- Subset datasets (200 images each): 1-3 hours
- FFHQ full dataset (70,000 images): 16-48 days for sequential processing
- CelebA-HQ full dataset (30,000 images): 7-21 days for sequential processing

**Important Note on Large-Scale Processing**: 
Processing the complete datasets is only feasible using High-Performance Computing (HPC) with SLURM parallel jobs:
- Even with parallel processing on multiple GPUs using SLURM job arrays, complete dataset evaluation takes several days
- We recommend using HPC facilities and splitting the workload into multiple parallel jobs
- Example SLURM configuration and job array scripts will be provided separately **on request**

**Note**: 
- Google Drive links may have download restrictions due to rate limiting. If you encounter issues, try downloading at a different time.
- Consider processing a subset of images for initial experiments due to the long processing times.

---

## Table of Contents

- [Generating Adversarial Examples](#generating-adversarial-examples)
  - [Command to Run `run-dpp.py` with Custom Paths](#command-to-run-run-dpppy-with-custom-paths)
- [Evaluating the Generated Images](#evaluating-the-generated-images)
  - [Command to Run `evaluate.py` with the Results Folder](#command-to-run-evaluatepy-with-the-results-folder)
- [Notes](#notes)

---

## Generating Adversarial Examples

The `run-dpp.py` script generates adversarial examples using a pretrained diffusion model. It reads images from an input directory, processes them, and saves both the original and adversarial images to an output directory.

### Command to Run `run-dpp.py` with Custom Paths

To override the input (`images_root`) and output (`save_dir`) paths, use the following command:

```bash
python run-dpp.py \
  paths.images_root=/path/to/your/input_images \
  paths.save_dir=/path/to/your/output_results \
```

**Parameters:**

- `paths.images_root`: Path to the directory containing your input images.
- `paths.save_dir`: Path to the directory where you want to save the results (both original and adversarial images).

**Example:**

```bash
python run-dpp.py \
  paths.images_root=./data/ffhq \
  paths.save_dir=./data/ffhq_outputs \
```

**Explanation:**

- This command runs `run-dpp.py` with the specified input and output paths.
- The script will process all images in `./data/ffhq` and save the results in `./data/ffhq_outputs`.
---

## Evaluating the Generated Images

After generating the adversarial examples, use `evaluate.py` to assess their effectiveness against various facial recognition models and compute the LPIPS metric.

### Command to Run `evaluate.py` with the Results Folder

Since the results (original and adversarial images) are saved in a single folder, use the `folder_type=single` option:

```bash
python evaluate.py \
  evaluation.folder_type=single \
  evaluation.data_folder=/path/to/your/output_results \
  evaluation.log_dir=/path/to/save/logs
```

**Parameters:**

- `evaluation.folder_type`: Set to `single` because both original and adversarial images are in the same folder.
- `evaluation.data_folder`: Path to the directory containing the results from `run-dpp.py`.
- `evaluation.log_dir`: Path to the directory where you want to save the evaluation logs.

**Example:**

```bash
python evaluate.py \
  evaluation.folder_type=single \
  evaluation.data_folder=./data/ffhq_outputs \
  evaluation.log_dir=./data/logs/celebahq_logs
  evaluation.log_dir=./data/logs/ffhq_logs
```

**Explanation:**

- This command runs `evaluate.py` using the results from `./data/ffhq_outputs`.
- The evaluation metrics and logs will be saved in `./data/logs/ffhq_logs`.

---

## Notes

- **Image Naming Convention**: Ensure that `run-dpp.py` saves the images with the appropriate naming convention expected by `evaluate.py`. Original images should end with `_originImage.png` and adversarial images with `_adv_image.png`.
- **Configuration Files**: Default values are set in the configuration files located in the `configs` directory. Override them via the command line as shown.
- **Pretrained Models**: Ensure that the paths to the pretrained diffusion and facial recognition models are correct.
- **Result Interpretation**: The evaluation script will output success rates and average distances for different models, as well as the average LPIPS distance, providing insight into the effectiveness and perceptual quality of the adversarial examples.
