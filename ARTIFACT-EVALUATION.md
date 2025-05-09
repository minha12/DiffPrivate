# Artifact Appendix

Paper title: **DiffPrivate: Facial Privacy Protection with Diffusion Models**

Artifacts HotCRP Id: **#21**

Requested Badge: **Available**, **Functional**, **Reproduced**

## Description

This artifact provides the implementation of DiffPrivate, a framework for protecting facial privacy using diffusion models. The codebase includes scripts for generating privacy-preserving facial images and evaluating their effectiveness against multiple facial recognition systems. The implementation requires Python 3.8, CUDA 11.3, and an NVIDIA GPU with 16GB+ VRAM. Our evaluation framework focuses on reproducing three main criteria:

1. Privacy Protection Rate and Image Quality: Measuring attack success rates against five facial recognition models (IR-SE50, IR152, FaceNet, CurricularFace, MobileFaceNet) and visual quality using LPIPS metric.

2. Cross Evaluation for Transferability: Assessing how well the privacy protection transfers across different facial recognition models through extensive cross-model evaluations.

3. Resistance Against Purification Methods: Evaluating the robustness of our privacy protection against various image purification techniques.

The cross evaluation is highly computationally intensive - to fully reproduce our exhaustive cross evaluations, we conducted the evaluation on a HPC cluster with SLURM job scheduling and access to multiple NVIDIA A100 GPUs. We recommend the same configurations for reproducing these results. The artifact includes configuration files, pretrained models, and detailed instructions for reproducing our experimental results on standard facial datasets.

**Important Note on Large-Scale Processing**: 
Processing the complete datasets is only feasible using High-Performance Computing (HPC) with SLURM parallel jobs:
- Even with parallel processing on multiple GPUs using SLURM job arrays, complete dataset evaluation takes several days
- We recommend using HPC facilities and splitting the workload into multiple parallel jobs
- Example SLURM configuration and job array scripts are provided

**Processing Time Estimates**:
Using an NVIDIA A100 GPU:
- Single image processing: 20-60 seconds
- Subset datasets (200 images each): 1-3 hours
- FFHQ full dataset (70,000 images): 16-48 days for sequential processing
- CelebA-HQ full dataset (30,000 images): 7-21 days for sequential processing

### Security/Privacy Issues and Ethical Concerns

Our artifact does not contain any malicious data or unsafe information. All images and datasets used in our experiments are publicly available and used in accordance with their respective licenses.

---

## Requirements

### Hardware Requirements

#### 1. Protection Evaluation (Section 1,3)

- **GPU:** NVIDIA GPU with at least 16GB VRAM (e.g., RTX 2080 Ti or higher).
- **CPU:** Modern multi-core processor.
- **RAM:** At least 32GB recommended.
- **Estimated Time:** Several hours to days, depending on computing resources for Table 1.

#### 2. Cross Evaluation (Section 2)

- **Compute Cluster:** Access to a High-Performance Computing (HPC) system with a Slurm workload manager.
- **GPUs:** Multiple NVIDIA GPUs across compute nodes.
- **RAM:** Depends on cluster nodes, typically sufficient on HPC systems.
- **Estimated Time:** Several hours to days, depending on cluster workload and resources.

> **Note:** Since the cross-evaluation is computationally intensive, it may not be feasible to reproduce without access to an HPC cluster.

---

### Software Requirements

- **Operating System:** Linux (tested on Ubuntu 20.04).
- **Python:** 3.8.
- **CUDA:** Compatible with your GPU (tested with CUDA 11.3).
- **Environment Management:** Miniforge3 or Anaconda.
- **Slurm:** For job scheduling on HPC systems (Section 2 only).

---

## Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/minha12/DiffPrivate.git
cd DiffPrivate
```

### Step 2: Set Up Environment (Choose A or B)

#### A. Using Setup Script (Recommended)

```bash
bash src/scripts/setup.sh

conda activate diffprivate
```

#### B. Using Docker

1. Build the Docker image:
```bash
docker build . -t diffprivate:latest
```

or pull the Docker image from our repository:

```bash
docker pull hale0007/diffprivate:latest
docker tag hale0007/diffprivate:latest diffprivate:latest
```

2. Run the container with GPU support:
```bash
docker run -d --gpus all --name diffprivate_container -v "$(pwd)/../DiffPrivate:/app/DiffPrivate" diffprivate:latest tail -f /dev/null
```

3. Access the container:
```bash
docker exec -it diffprivate_container bash
```

Now you can run Python commands directly inside the container, for example:
```bash
python run-dpp.py
```

### Step 3: Prepare the Dataset

Before running the evaluation, you can choose between full datasets or smaller subsets for initial experiments:

### FFHQ Dataset
#### Full Dataset (70,000 images)
1. Download and unzip the dataset:
   ```bash
   python src/scripts/download_datasets.py 
   ```

#### Subset (200 images)
1. Create the directory and unzip the dataset:
   ```bash
   mkdir -p ./data/ffhq_256_subset
   cd ./data
   unzip ffhq_256_subset.zip
   ```

### CelebA-HQ Dataset
#### Full Dataset (30,000 images)
1. Download and unzip the dataset (don't have to do this again if command already run):
   ```bash
   python src/scripts/download_datasets.py
   ```

#### Subset (200 images)
1. Create the directory and unzip the dataset:
   ```bash
   mkdir -p ./data/celeba_hq_256_subset
   cd ./data
   unzip celeba_hq_256_subset.zip
   ```

**Note**: 
- Google Drive links may have download restrictions due to rate limiting. If you encounter issues, try downloading at a different time.
- Consider processing a subset of images for initial experiments due to the long processing times.

---

## Artifact Evaluation

### Section 1: Protection Evaluation (Reproducing Table 1)

In this section, we will run the evaluation script on the FFHQ dataset to reproduce the results presented in Table 1 of the paper.

#### Step 1: Create Protected Image

To generate privacy-protected images using **DiffPrivate Perturb**, run:

```bash
python run-dpp.py paths.images_root=<path_to_images> paths.save_dir=<save_path>
```

- Replace `<path_to_images>` with the path to your images.
- Replace `<save_path>` with the directory where you want to save the results.

Examples:

```bash
python run-dpp.py paths.images_root=./data/ffhq paths.save_dir=./data/output_ffhq
```

The results, including the protected images and logs, will be saved in the specified `save_dir`.

#### Step 2: Run the Evaluation Script

```bash
python src/scripts/evaluate.py evaluation.data_folder=data/ffhq evaluation.log_dir=logs
```

**Notes:**

- The `evaluate.py` script processes image pairs and evaluates the effectiveness of the privacy protection.
- Ensure that the `data_folder` contains the original and adversarial images as per the script's expectations.

#### Step 3: Retrieve the Results

- After the script completes, results will be saved in the `logs` directory.
- The `output.txt` and `report.txt` files contain the success rates against different facial recognition models and the average LPIPS distances.

#### Step 3: Analyze `report.txt`

- The `report.txt` file provides metrics corresponding to Table 1 in the paper.
- Verify that the success rates and LPIPS values match the reported results.

---

### Section 2: Cross Evaluation (Reproducing Figure 7)

This section involves large-scale evaluation using multiple models and requires submitting batch jobs to a Slurm-managed HPC cluster.

**Important**: Before proceeding, install the required conda environment on your HPC system using the setup script from Option A above.

See `run_cross_attacks.sh` script for default values, for example module name (`Miniforge3`) and conda environment (`diffprivate`). You may need to adapt these parameters to match your HPC system's configuration:

```bash
--module_name=<your_conda_module>  # e.g., Anaconda3, miniconda3
--conda_env=<your_env_name>         # the conda environment created 
--constraint=<your_gpu_constraint> # in our system: thin|fat -> A100 40GB|80GB
```

**Note on Small-Scale Testing Without HPC**: While full evaluation requires an HPC cluster with SLURM, you can still test the cross-evaluation functionality on a regular machine with a small dataset. The script supports an `execution_mode` parameter that lets you choose between SLURM job submission and direct bash execution:

```bash
bash src/scripts/run_cross_attacks.sh \
  --image_dir=./data/ffhq_256_subset \
  --batch_size=100 \ # batch_size does not make much sense in bash mode but make sure it is smaller than total number of files
  --execution_mode=bash  # Use 'bash' for direct execution or 'slurm' (default) for HPC
```

With `--execution_mode=bash`, the script will run tasks sequentially on your local machine instead of submitting SLURM jobs. This allows you to verify functionality even without access to an HPC system, though processing will be much slower. For testing purposes, we recommend:

- Using a very small subset of images (10-20 images)
- Starting with fewer model combinations initially
- Expecting longer processing times (minutes to hours depending on your GPU)

Note that while this approach is suitable for verifying functionality, reproducing the full cross-evaluation results from the paper still requires HPC resources.

#### Step 1: Configure the Slurm Batch Job Script

Edit `src/scripts/run_cross_attacks.sh` to ensure that the `#SBATCH` directives match your HPC environment.

Example adjustments:

- **Account Allocation:**

  ```bash
  #SBATCH -A your_account
  ```

- **Time Allocation:**

  ```bash
  #SBATCH -t 48:00:00  # Set appropriate time limit
  ```

#### Step 2: Submit Jobs to Slurm

Make sure the script is executable:

```bash
chmod +x src/scripts/run_cross_attacks.sh
```

Execute the script by default values:

```bash
bash src/scripts/run_cross_attacks.sh
```

or execute with specific arguments:

```bash
bash run_cross_attacks.sh \
  --image_dir=./data/ffhq_256_subset \
  --batch_size=200 \
  --account=<your_account> \
  --gpus=1 \
  --constraint=thin \
  --time=72:00:00 \
  --module=Miniforge3 \
  --conda_env=diffprivate
```

This script:
- Processes all attacker-victim model pairs (5×5 matrix, excluding same-model pairs)
- Divides datasets into batches for parallel processing
- Submits SLURM jobs with specified computational resources
- Stores results in experiment-specific directories
- Handles job dependencies and cleanup automatically

#### Step 3: Monitor Job Status

Check the status of your jobs:

```bash
squeue -u your_username
```

#### Step 4: Wait for Job Completion

- Ensure all jobs have completed before proceeding.
- Job outputs will be stored as specified in the script.

---

### Post-Processing After Job Completion

#### Step 5: Run Cross Evaluation Script

After all jobs have completed, collect and summarize the results.

```bash
bash src/scripts/cross_evaluation.sh
```

This script:

- Runs evaluation on each experiment folder.
- Aggregates logs and outputs into a summary file.

#### Step 6: Visualize the Results

Generate visualizations analogous to Figure 7.

```bash
python src/scripts/vis_cross_eval.py
```

- The script reads the summary data and creates plots.
- Output files (e.g., PDF or PNG) will be saved in the `./visualized` directory.

#### Step 7: Verify the Results

- Compare the generated plots with Figure 7 in the paper.
- The trends and values should closely match.

---

### Section 3: Purification Methods Evaluation (Reproducing Figure 8)

This section evaluates how different purification methods affect our privacy protection technique, reproducing the results from Figure 8 in the paper.

#### Step 1: Set Data Directories

Before running the purification experiments, ensure the correct data paths are set in `src/scripts/purify_and_evaluate.sh`:

```bash
# Configure data directories in the script
ORIGINAL_DATA_DIR="./data/demo/images"    # Directory with original images
PROTECTED_DATA_DIR="./data/output"        # Directory with protected images
OUTPUT_DIR="./experiment_purify"          # Directory for experiment results
```

These paths should point to:
- Original facial images
- Protected images generated from previous steps
- Output directory for purification results

#### Step 2: Run Purification and Evaluation

Make the script executable and run:

```bash
chmod +x src/scripts/purify_and_evaluate.sh
bash src/scripts/purify_and_evaluate.sh
```

The script will:
1. Apply various purification methods to protected images
2. Evaluate each method against multiple facial recognition models
3. Generate a summary table in both file and terminal output

#### Step 3: View Results

Results are also saved to `./experiment_purify/summary_success_rates.txt`.


---

## Expected Results

### Section 1: Protection Evaluation

- **Outputs:**
  - `logs/output.txt`: Detailed evaluation metrics.
  - `logs/report.txt`: Aggregated results corresponding to Table 1.
- **Metrics:**
  - Success rates against each facial recognition model.
  - Average LPIPS distance indicating visual similarity.

### Section 2: Cross Evaluation

  - **Outputs:**
    - `experiments_cross/logs/summary_success_rates.txt`: Aggregated success rates.
    - Visualization files generated by `vis_cross_eval.py`.
  - **Metrics:**
    - Cross-model attack success rates.
    - Plots showing the performance across different attacker-victim pairs.

### Section 3: Purification Methods Evaluation

  - **Outputs:**
    - `experiment_purify/summary_success_rates.txt`: Detailed purification results.
  - **Metrics:**
    - Success rates for each purification method.
    - Comparative performance across facial recognition models.
---

## Notes and Troubleshooting

- **Dataset Preparation:**
  - Ensure that image files are named and formatted correctly.
  - The scripts expect specific naming conventions (e.g., pairs of images).

- **Slurm Job Submission:**
  - Adjust the `run_cross_attacks.sh` script according to your HPC environment.
  - Verify that module loads and environment activations work as intended.

- **Script Execution Errors:**
  - Check for missing dependencies or incorrect paths.
  - Use debugging statements (`echo`, `set -x`) to trace script execution.

- **Permissions:**
  - Ensure scripts are executable (`chmod +x script.sh`).
  - Check file and directory permissions if encountering access issues.
