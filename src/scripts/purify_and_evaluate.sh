#!/bin/bash
################################################################################
################################ PURIFICATION ##################################
################################################################################

ORIGINAL_DATA_DIR="./data/demo/images"
PROTECTED_DATA_DIR="./data/output"
OUTPUT_DIR="./experiment_purify"

# Run purify.py with the specified input and output paths
python src/scripts/purify.py defense.input_path="$PROTECTED_DATA_DIR" defense.output_path="$OUTPUT_DIR"

################################################################################
################################ EVALUATION ####################################
################################################################################

# Loop through each purification method's output directory
for METHOD_DIR in "$OUTPUT_DIR"/*; do
  if [ -d "$METHOD_DIR" ]; then
    METHOD_NAME=$(basename "$METHOD_DIR")
    LOG_DIR="$METHOD_DIR/logs"
    mkdir -p "$LOG_DIR"
    echo "Evaluating purified images in $METHOD_DIR"
    # Update the python command to include data_folder, adv_folder, and folder_type
    python src/scripts/evaluate.py \
      evaluation.clean_folder="$ORIGINAL_DATA_DIR" \
      evaluation.adv_folder="$METHOD_DIR" \
      evaluation.log_dir="$LOG_DIR" \
      evaluation.folder_type="separate" \
      evaluation.ignore_extension=True
    echo "------------------------------------------------"
  fi
done

################################################################################
################################ SUMMARY #######################################
################################################################################

# Define the output summary file
SUMMARY_FILE="$OUTPUT_DIR/summary_success_rates.txt"

# Initialize the summary file with the header row
echo "Purification Method,IRSE50,IR152,FaceNet,IR101,MobileFace" > "$SUMMARY_FILE"

# Loop through logs and extract success rates
for METHOD_DIR in "$OUTPUT_DIR"/*; do
  if [ -d "$METHOD_DIR" ]; then
    METHOD_NAME=$(basename "$METHOD_DIR")
    LOG_FILE="$METHOD_DIR/logs/output.txt"
    if [ -f "$LOG_FILE" ]; then
      # Extract Success rate using awk
      success_rates=$(awk -F': ' '/Success rate:/ {print $2}' "$LOG_FILE" | tr -d '[]')
      # Format the success rates to include commas
      success_rates_formatted=$(echo $success_rates | sed 's/ /,/g')
      # Append the data to the summary file
      echo "$METHOD_NAME,$success_rates_formatted" >> "$SUMMARY_FILE"
    else
      echo "No output.txt found in $LOG_DIR"
    fi
  fi
done

echo "Summary has been saved to $SUMMARY_FILE."

# Beautify and display the summary table
echo -e "\Summary Table:"
column -t -s',' "$SUMMARY_FILE"