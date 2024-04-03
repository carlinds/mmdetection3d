# Run over different values of range_fraction
for range_fraction in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/petr_orig.txt --results_b result_paths/petr_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/petr/per_sample --output_file result_paths/petr_agreement_output_files.txt --range_fraction $range_fraction
  sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/petr_orig.txt --results_b result_paths/petr_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/petr/all --output_file result_paths/petr_agreement_output_files.txt --aggregate_all --range_fraction $range_fraction
  sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/petr_orig.txt --results_b result_paths/petr_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/petr/per_scene --output_file result_paths/petr_agreement_output_files.txt --aggregate_per_scene --range_fraction $range_fraction

  sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/fcos3d_orig.txt --results_b result_paths/fcos3d_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/fcos3d/per_sample --output_file result_paths/fcos3d_agreement_output_files.txt --range_fraction $range_fraction
  sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/fcos3d_orig.txt --results_b result_paths/fcos3d_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/fcos3d/all --output_file result_paths/fcos3d_agreement_output_files.txt --aggregate_all --range_fraction $range_fraction
  sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/fcos3d_orig.txt --results_b result_paths/fcos3d_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/fcos3d/per_scene --output_file result_paths/fcos3d_agreement_output_files.txt --aggregate_per_scene --range_fraction $range_fraction
done
