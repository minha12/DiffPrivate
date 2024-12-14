#!/bin/bash
################################################################################
################################ EVALUATION ####################################
################################################################################

# Define the base directory for the experiments
BASE_DIR="./experiments_cross"
# Define the log directory as a 'logs' sub-folder under the base directory
LOG_DIR="$BASE_DIR/logs"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Loop through all subdirectories of the base directory
for SUBDIR in "$BASE_DIR"/*; do
  if [ -d "$SUBDIR" ] && [ "$SUBDIR" != "$LOG_DIR" ]; then  # Check if it is a directory
    
    # Run the evaluation script with the subdirectory as the data folder
    # and the logs directory directly under the base directory
    echo "Evaluating folder: $SUBDIR"
    python src/scripts/evaluate.py evaluation.data_folder="$SUBDIR" evaluation.log_dir="$SUBDIR"

    # Echo a separator for readability
    echo "------------------------------------------------"
  fi
done

################################################################################
################################ LOGS COLLECTION ###############################
################################################################################

# Loop through all subdirectories of the base directory
for SUBDIR in "$BASE_DIR"/*; do
  if [ -d "$SUBDIR" ] && [ "$SUBDIR" != "$LOG_DIR" ]; then  # Check if it is a directory and not the log directory
    # Extract the folder name without the path
    FOLDER_NAME=$(basename "$SUBDIR")

    # Correctly format the new filename using sed
    # This will remove the 'ars_' prefix and '_ffhq_TA-True' suffix,
    # then replace the first occurrence of '_vs_' with '_vs_',
    # ensuring consistent naming for all files.
    NEW_FILENAME=$(echo "$FOLDER_NAME" | sed -E 's/^ars_(.*)_vs_(.*)_ffhq_TA-True/\1_vs_\2_ffhq.txt/')

    # Copy the output.txt file to the log directory with the new name
    if [ -f "$SUBDIR/output.txt" ]; then
      cp "$SUBDIR/output.txt" "$LOG_DIR/$NEW_FILENAME"
      echo "Copied and renamed output.txt from $SUBDIR to $LOG_DIR/$NEW_FILENAME"
    else
      echo "No output.txt found in $SUBDIR"
    fi
  fi
done

################################################################################
################################ SUMMARY #######################################
################################################################################

# Define the mapping in an associative array
declare -A rename_map=(
    ["cur_face"]="IR101"
    ["facenet"]="FaceNet"
    ["ir152"]="IR152"
    ["irse50"]="IRSE50"
    ["mobile_face"]="MobileFace"
)

# Output file
summary_file="$LOG_DIR/summary_success_rates.txt"

# Initialize the summary file with the header row
echo "Attacker,Victim,IRSE50,IR152,FaceNet,IR101,MobileFace" > "$summary_file"

# Loop through each file in the folder
for file in "$LOG_DIR"/*.txt; do
    # Extract the attacker and victim names from the filename
    if [[ $file =~ ([a-zA-Z0-9_]+)_vs_([a-zA-Z0-9_]+)_ffhq.txt ]]; then
        attacker="${BASH_REMATCH[1]}"
        victim="${BASH_REMATCH[2]}"

        # Remap the names
        remapped_attacker="${rename_map[$attacker]}"
        remapped_victim="${rename_map[$victim]}"

        # Extract Success rate using awk, assuming it's on the second line of the file
        success_rate=$(awk -F': ' '/Success rate:/ {print $2}' "$file" | tr -d '[]')
        
        # Format the success_rate to include commas correctly
        success_rate_formatted=$(echo $success_rate | sed 's/ /,/g')

        # Append the information to the summary file, formatted as CSV
        echo "$remapped_attacker,$remapped_victim,$success_rate_formatted" >> "$summary_file"
    fi
done

echo "Summary has been saved to $summary_file."