#!/bin/bash

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

# Dataset and image directory configuration
dataset="ffhq"
image_dir="data/${dataset}"
targeted_attack="True"

# **Define the batch size variable**
batch_size=200  # Change this value to your desired batch size

# Find all jpg images in the directory, sort them, and read into an array
readarray -t files < <(find "${image_dir}" -name "*.jpg" | sort)

# Get the total number of files
num_files=${#files[@]}

# Debugging statement
echo "Found ${num_files} jpg files in ${image_dir}"

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

#SBATCH -A berzelius-2024-460
#SBATCH --gpus 1
#SBATCH -C "thin"
#SBATCH -t 72:00:00
#SBATCH --output=slurm-logs/slurm-%j.out  # Standard output log file
#SBATCH --error=slurm-logs/slurm-%j.err   # Standard error log file

module load Miniforge3

conda activate diffprivate

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