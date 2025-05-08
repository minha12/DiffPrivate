#!/bin/bash
# Default parameters
image_dir="./data/ffhq"  # Default dataset path
batch_size=200  # Default batch size
account="berzelius-2024-460"
gpus="1"
constraint="thin"
time="72:00:00"
module_name="Miniforge3"
conda_env="diffprivate"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image_dir=*)
      image_dir="${1#*=}"
      shift
      ;;
    --batch_size=*)
      batch_size="${1#*=}"
      shift
      ;;  
    --account=*)
      account="${1#*=}"
      shift
      ;;
    --gpus=*)
      gpus="${1#*=}"
      shift
      ;;
    --constraint=*)
      constraint="${1#*=}"
      shift
      ;;
    --time=*)
      time="${1#*=}"
      shift
      ;;
    --module=*)
      module_name="${1#*=}"
      shift
      ;;
    --conda_env=*)
      conda_env="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--image_dir=VALUE] [--batch_size=VALUE] [--account=VALUE] [--gpus=VALUE] [--constraint=VALUE] [--time=VALUE] [--module=VALUE] [--conda_env=VALUE]"
      exit 1
      ;;
  esac
done

# Display the parameters being used
echo "Using parameters:"
echo "  Dataset path: ${image_dir}"
echo "  Batch size: ${batch_size}"
echo "  SLURM account: ${account}"
echo "  GPUs: ${gpus}"
echo "  Constraint: ${constraint}"
echo "  Time: ${time}"
echo "  Module: ${module_name}"
echo "  Conda environment: ${conda_env}"

# Extract dataset name from path for naming purposes
dataset=$(basename "${image_dir}")
# Define thresholds for each model
declare -A threshold_dict=(
    ["irse50"]=0.4
    ["ir152"]=0.4277
    ["facenet"]=0.33519999999999994
    ["cur_face"]=0.4332
    ["mobile_face"]=0.3875
)
# List of all models
models=("irse50" "ir152" "facenet" "cur_face" "mobile_face")
# Base directory for storing experiments
base_save_dir="experiments_cross"
# Create base_save_dir if it does not exist
mkdir -p "${base_save_dir}"
# Set targeted attack flag
targeted_attack="True"
# Batch size is already defined in the command line parsing section
# Find all jpg images in the directory, sort them, and read into an array
readarray -t files < <(find "${image_dir}" -name "*.jpg" | sort)
# Get the total number of files
num_files=${#files[@]}
# Debugging statement
echo "Found ${num_files} jpg files in ${image_dir}"
# Create slurm-logs directory if it doesn't exist
mkdir -p slurm-logs
# Loop through all attacker models
for attacker in "${models[@]}"; do
    echo "Working with attacker model: ${attacker}"
    # Loop through all victim models
    for victim in "${models[@]}"; do
        # Skip if attacker and victim are the same
        if [ "$attacker" == "$victim" ]; then
            continue
        fi
        echo "Working with victim model: ${victim}"
        # Set the victim threshold
        victim_threshold=${threshold_dict[$victim]}
        # Set temporary directory prefix
        temp_dir_prefix="${attacker}_${victim}_${dataset}_tgt_temp_"
        # Dynamically set the save_dir based on the attacker, victim, and targeted attack
        save_dir="${base_save_dir}/ars_${attacker}_vs_${victim}_$(basename "${image_dir}")_TA-${targeted_attack}"
        echo "Save directory: ${save_dir}"
        # Loop through the files in batches
        for ((i=0; i<num_files; i+=batch_size)); do
            # Calculate batch index for naming
            batch_index=$((i / batch_size + 1))
            temp_dir="./temp/${temp_dir_prefix}${batch_index}"
            # Create a temporary directory for the current batch
            mkdir -p "${temp_dir}"
            # Copy the batch of images to the temporary directory
            for ((j=0; j<batch_size && i + j < num_files; j++)); do
                cp "${files[$i + $j]}" "${temp_dir}"
            done
            # Create a temporary sbatch script for the current batch
            sbatch_script="${temp_dir}/sbatch_script_${batch_index}.sh"
            cat << EOF > "${sbatch_script}"
#!/bin/bash
#SBATCH -A ${account}
#SBATCH --gpus ${gpus}
#SBATCH -C "${constraint}"
#SBATCH -t ${time}
#SBATCH --output=slurm-logs/slurm-%j.out  # Standard output log file
#SBATCH --error=slurm-logs/slurm-%j.err   # Standard error log file
module load ${module_name}
conda activate ${conda_env}
# Run the Python script with parameters for the current batch
python run-dpp.py \\
    paths.save_dir="${save_dir}" \\
    paths.images_root="${temp_dir}" \\
    model.attacker_model="${attacker}" \\
    model.victim_model="${victim}" \\
    attack.overhead=0.04 \\
    attack.learning_rate=2e-2 \\
    attack.targeted_attack=${targeted_attack} \\
    attack.victim_threshold=${victim_threshold} \\
    model.ensemble=False
# Delete the temporary directory after job completion
rm -rf "${temp_dir}"
EOF
            # Make the sbatch script executable
            chmod +x "${sbatch_script}"
            # Submit the sbatch script
            echo "Submitting job: sbatch \"${sbatch_script}\""
            sbatch "${sbatch_script}"
        done
    done
done
echo "All combinations of cross-validation jobs have been submitted."