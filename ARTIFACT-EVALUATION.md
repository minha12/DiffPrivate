# Artifact Appendix

Paper title: **DiffPrivate: Facial Privacy Protection with Diffusion Models**

Artifacts HotCRP Id: **#17**

Requested Badge: **Available**, **Functional**, **Reproduced**

## Description

This artifact provides the implementation of DiffPrivate, a framework for protecting facial privacy using diffusion models. The codebase includes scripts for generating privacy-preserving facial images and evaluating their effectiveness against multiple facial recognition systems. The implementation requires Python 3.8, CUDA 11.3, and an NVIDIA GPU with 16GB+ VRAM. Our evaluation framework measures both the attack success rates against five facial recognition models (IR-SE50, IR152, FaceNet, CurricularFace, MobileFaceNet) and the visual quality using LPIPS metric. The artifact includes configuration files, pretrained models, and detailed instructions for reproducing our experimental results on standard facial datasets.

### Security/Privacy Issues and Ethical Concerns

Our artifact does not contain any malicious data or unsafe information. All images and datasets used in our experiments are publicly available and used in accordance with their respective licenses.

---

## Requirements

### Hardware Requirements

#### 1. Protect Evaluation (Section 1)

- **GPU:** NVIDIA GPU with at least 16GB VRAM (e.g., RTX 2080 Ti or higher).
- **CPU:** Modern multi-core processor.
- **RAM:** At least 32GB recommended.
- **Estimated Time:** Approximately 2 hours to reproduce results for Table 1.

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

#### Python Dependencies

A `requirements.txt` file is provided. Key packages include:

- `torch`
- `torchvision`
- `tqdm`
- `numpy`
- `pandas`
- `matplotlib`
- `hydra-core`
- `lpips`

---

## Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/minha12/DiffPrivate.git
cd DiffPrivate
```

### Step 2: Install Miniforge3 (if not already installed)

Download and install from [Miniforge3 GitHub](https://github.com/conda-forge/miniforge/releases/latest).

### Step 3: Create and Activate the Conda Environment

```bash
conda env create -n diffprivate python=3.8
conda activate diffprivate
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Place your facial image dataset in the appropriate directory.

For example, using the FFHQ dataset:

```bash
mkdir -p data/ffhq
# Place your images (e.g., .jpg files) into the data/ffhq directory
```

---

## Artifact Evaluation

### Section 1: Current Evaluation (Reproducing Table 1)

In this section, we will run the evaluation script on a small dataset to reproduce the results presented in Table 1 of the paper.

#### Step 1: Run the Evaluation Script

```bash
python src/evaluate.py --data_folder data/ffhq --log_dir logs
```

**Notes:**

- The `evaluate.py` script processes image pairs and evaluates the effectiveness of the privacy protection.
- Ensure that the `data_folder` contains the original and adversarial images as per the script's expectations.

#### Step 2: Retrieve the Results

- After the script completes, results will be saved in the `logs` directory.
- The `output.txt` and `report.txt` files contain the success rates against different facial recognition models and the average LPIPS distances.

#### Step 3: Analyze `report.txt`

- The `report.txt` file provides metrics corresponding to Table 1 in the paper.
- Verify that the success rates and LPIPS values match the reported results.

---

### Section 2: Cross Evaluation (Reproducing Figure 7)

This section involves large-scale evaluation using multiple models and requires submitting batch jobs to a Slurm-managed HPC cluster.

#### Step 1: Configure the Slurm Batch Job Script

Edit `src/scripts/slurm-batch-job.sh` to ensure that the `#SBATCH` directives match your HPC environment.

Example adjustments:

- **Account Allocation:**

  ```bash
  #SBATCH -A your_account
  ```

- **Time Allocation:**

  ```bash
  #SBATCH -t 48:00:00  # Set appropriate time limit
  ```

- **Partition or QoS:**

  ```bash
  #SBATCH -p your_partition
  ```

#### Step 2: Submit Jobs to Slurm

Make sure the script is executable:

```bash
chmod +x src/scripts/slurm-batch-job.sh
```

Execute the script:

```bash
bash src/scripts/slurm-batch-job.sh
```

This script:

- Iterates over combinations of attacker and victim models.
- Submits individual jobs for each image pair using `sbatch`.

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
python src/visualize_cross_eval.py
```

- The script reads the summary data and creates plots.
- Output files (e.g., PDF or PNG) will be saved in the current directory.

#### Step 7: Verify the Results

- Compare the generated plots with Figure 7 in the paper.
- The trends and values should closely match.

---

## Expected Results

### Section 1: Current Evaluation

- **Outputs:**
  - `logs/output.txt`: Detailed evaluation metrics.
  - `logs/report.txt`: Aggregated results corresponding to Table 1.
- **Metrics:**
  - Success rates against each facial recognition model.
  - Average LPIPS distance indicating visual similarity.

### Section 2: Cross Evaluation

- **Outputs:**
  - `experiments_cross/logs/summary_success_rates.txt`: Aggregated success rates.
  - Visualization files generated by `visualize_cross_eval.py`.
- **Metrics:**
  - Cross-model attack success rates.
  - Plots showing the performance across different attacker-victim pairs.

---

## Notes and Troubleshooting

- **Dataset Preparation:**
  - Ensure that image files are named and formatted correctly.
  - The scripts expect specific naming conventions (e.g., pairs of images).

- **Slurm Job Submission:**
  - Adjust the `slurm-batch-job.sh` script according to your HPC environment.
  - Verify that module loads and environment activations work as intended.

- **Script Execution Errors:**
  - Check for missing dependencies or incorrect paths.
  - Use debugging statements (`echo`, `set -x`) to trace script execution.

- **Permissions:**
  - Ensure scripts are executable (`chmod +x script.sh`).
  - Check file and directory permissions if encountering access issues.
