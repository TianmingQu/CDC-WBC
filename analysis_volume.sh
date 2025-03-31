
PDB_PATH="/scratch2/users/tq19b/analysis_2024/1jwp_2024_water_analysis_data/1jwp_noh.pdb"
DCD_PATH="/scratch2/users/tq19b/analysis_2024/1jwp_2024_water_analysis_data/1jwp_noh.dcd"
convex_selection='name CA and (resid 236 to 237 or resid 215 to 224 or resid 242 to 245 or resid 268 to 284)'
target_selection='name CA and (resid 236 to 237 or resid 215 to 224 or resid 242 to 245 or resid 268 to 284)'
DENSITY_MAP_DIR='/scratch2/users/tq19b/analysis_2024/1jwp_2024_water_analysis_data/water_density_maps_3/'
OUTPUT_PREFIX='1jwp'

python /scratch/users/tq19b/analysis/1jwp_analysis/new_analysis_CBT_each_reference/calculate_cryptic_site_volume_weighted_sumD_1115.py "$PDB_PATH" "$DCD_PATH" "$convex_selection" "$target_selection" "$DENSITY_MAP_DIR" "$OUTPUT_PREFIX"
