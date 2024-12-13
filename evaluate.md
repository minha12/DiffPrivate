# Evaluation Guide

This guide provides instructions on how to generate adversarial examples using `run-dpp.py` with custom input and output paths and then evaluate the generated images using `evaluate.py`. The evaluation includes metrics from different facial recognition models and the LPIPS metric.

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
  paths.pretrained_diffusion_path=/path/to/pretrained/diffusion/model
```

**Parameters:**

- `paths.images_root`: Path to the directory containing your input images.
- `paths.save_dir`: Path to the directory where you want to save the results (both original and adversarial images).
- `paths.pretrained_diffusion_path`: Path to the pretrained diffusion model.

**Example:**

```bash
python run-dpp.py \
  paths.images_root=./data/clean_images \
  paths.save_dir=./results/adversarial_examples \
  paths.pretrained_diffusion_path=./models/stable-diffusion
```

**Explanation:**

- This command runs `run-dpp.py` with the specified input and output paths.
- The script will process all images in `./data/clean_images` and save the results in `./results/adversarial_examples`.
- The pretrained diffusion model is loaded from `./models/stable-diffusion`.

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
  evaluation.data_folder=./results/adversarial_examples \
  evaluation.log_dir=./logs/evaluation_results
```

**Explanation:**

- This command runs `evaluate.py` using the results from `./results/adversarial_examples`.
- The evaluation metrics and logs will be saved in `./logs/evaluation_results`.

---

## Notes

- **Image Naming Convention**: Ensure that `run-dpp.py` saves the images with the appropriate naming convention expected by `evaluate.py`. Original images should end with `_originImage.png` and adversarial images with `_adv_image.png`.
- **Configuration Files**: Default values are set in the configuration files located in the `configs` directory. Override them via the command line as shown.
- **Pretrained Models**: Ensure that the paths to the pretrained diffusion and facial recognition models are correct.
- **Result Interpretation**: The evaluation script will output success rates and average distances for different models, as well as the average LPIPS distance, providing insight into the effectiveness and perceptual quality of the adversarial examples.
