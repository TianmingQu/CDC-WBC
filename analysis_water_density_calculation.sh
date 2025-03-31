#!/bin/bash

start_time=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0  # Start the timer

echo "Script started at: $start_time"
# Check if a PDB_ID is provided as an argument
if [ -z "$1" ]; then
    echo "Error: No PDB ID provided. Please provide a PDB ID as the first argument."
    exit 1
fi

# Convert the input PDB_ID to uppercase
PDB_ID=$(echo "$1" | tr '[:lower:]' '[:upper:]')

# Part 1: Water Oxygen Copy Process
# Define the required parameters for the first part


PROTEIN_PDB=../step4_equilibration.pdb
PROTEIN_UNALIGNED_DCD=../dyn.dcd
OUTPUT_DIR=$(pwd)
TEMP_DIR=$(pwd)

# Define output PDB and DCD file names (protein-only and aligned)
OUTPUT_PDB="$OUTPUT_DIR/aligned_protein.pdb"
OUTPUT_DCD="$OUTPUT_DIR/aligned_protein.dcd"

# Create the output and temp directories if they do not exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Path to the Python script for water oxygen copy
PYTHON_WATER_COPY_SCRIPT=/home/tq19b/general_files/analysis_script/copy_water.py

# Run the Python script for water oxygen copy
echo "Water oxygen copy starts."
python "$PYTHON_WATER_COPY_SCRIPT" "$PROTEIN_PDB" "$PROTEIN_UNALIGNED_DCD" "$OUTPUT_DIR" "$TEMP_DIR"

# Check if the water oxygen copy process was successful
if [ $? -eq 0 ]; then
    echo "Water oxygen copy completed successfully."
    echo "Results are saved in $OUTPUT_DIR"
else
    echo "Error: Water Oxygen Copy Failed"
    exit 1
fi

# Run the Python script to align the protein and output the PDB and DCD files
echo "Align protein only trajectory starts."
PYTHON_ALIGNMENT_SCRIPT=/home/tq19b/general_files/analysis_script/align_protein.py
python "$PYTHON_ALIGNMENT_SCRIPT" "$PROTEIN_PDB" "$PROTEIN_UNALIGNED_DCD" "$OUTPUT_PDB" "$OUTPUT_DCD"

# Check if the protein alignment process was successful
if [ $? -eq 0 ]; then
    echo "Protein alignment complete."
    echo "Output PDB: $OUTPUT_PDB"
    echo "Output DCD: $OUTPUT_DCD"
else
    echo "Error: Protein alignment failed."
    exit 1
fi

# Part 2: RMS Alignment Process
# Set the input file paths for the second part (these are generated by the first part)

REFERENCE_PDB="$OUTPUT_DIR/protein_copied_Water_0.pdb"
TRAJECTORY_DCD="$OUTPUT_DIR/protein_copied_water_unaligned.dcd"

# Check if the necessary files exist
if [ ! -f "$REFERENCE_PDB" ] || [ ! -f "$TRAJECTORY_DCD" ]; then
    echo "Error: Required files for RMS alignment not found."
    exit 1
fi

# Path to the Python script for RMS alignment
PYTHON_ALIGNMENT_SCRIPT=/home/tq19b/general_files/analysis_script/align_water_copied_trajectory.py

# Run the Python script for RMS alignment
echo "Align Copied Water Trajectory Starts"
python3 "$PYTHON_ALIGNMENT_SCRIPT" "$REFERENCE_PDB" "$TRAJECTORY_DCD" "$OUTPUT_DIR"

# Check if the RMS alignment process was successful
if [ $? -eq 0 ]; then
    echo "RMS alignment completed successfully. Aligned trajectory saved in $OUTPUT_DIR"
else
    echo "Error: RMS alignment failed."
    exit 1
fi

# Part 3: Water Density Calculation Process
# Define input variables for the water density calculation
PDB_PATH=../step4_equilibration.pdb
DCD_PATH=../dyn.dcd
REF_PDB_PATH="$OUTPUT_DIR/protein_copied_Water_0.pdb"
REF_DCD_PATH="$OUTPUT_DIR/aligned_water_copied_trajectory.dcd"
RESOLUTION1=2
RESOLUTION2=2
GPU_ID=0

# Define the new output directory for water density maps
WATER_DENSITY_OUTPUT_DIR="$OUTPUT_DIR/water_density_maps"
# Create the water density output directory if it does not exist
mkdir -p "$WATER_DENSITY_OUTPUT_DIR"

# Call the Python script for water density calculation
echo "Calculate Water Density Starts"
python /home/tq19b/general_files/analysis_script/calcualte_water_density.py "$PDB_PATH" "$DCD_PATH" "$REF_PDB_PATH" "$REF_DCD_PATH" "$RESOLUTION1" "$RESOLUTION2" "$GPU_ID"

# Check if the water density calculation process was successful
if [ $? -eq 0 ]; then
    echo "Water density calculation completed successfully."
    echo "Results are saved in $WATER_DENSITY_OUTPUT_DIR"
else
    echo "Error: Water density calculation failed."
    exit 1
fi

end_time=$(date +"%Y-%m-%d %H:%M:%S")
elapsed_time=$SECONDS

echo "Script started at: $start_time"
echo "Script ended at: $end_time"
echo "Total time elapsed: $(($elapsed_time / 3600)) hours, $((($elapsed_time % 3600) / 60)) minutes, and $(($elapsed_time % 60)) seconds."