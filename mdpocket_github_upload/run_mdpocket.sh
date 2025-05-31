#!/usr/bin/env bash
set -euo pipefail

# Record start time and initialize timer
start_time=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0
echo "mdpocket run started at $start_time"

# Input files
PDB="1jwp_noh.pdb"
DCD="1jwp_noh.dcd"

# Output pocket file
POCKET_OUT="mdpocket_selected_pocket.pdb"

# Run mdpocket
mdpocket \
  --trajectory_file "$DCD" \
  --trajectory_format dcd \
  -f "$PDB" \
  --selected_pocket "$POCKET_OUT"

# Record end time and compute elapsed time
end_time=$(date +"%Y-%m-%d %H:%M:%S")
elapsed=$SECONDS
hours=$(( elapsed/3600 ))
minutes=$(( (elapsed%3600)/60 ))
seconds=$(( elapsed%60 ))

echo "mdpocket run completed at $end_time"
echo "Total time elapsed: ${hours}h ${minutes}m ${seconds}s (${elapsed} seconds)"
