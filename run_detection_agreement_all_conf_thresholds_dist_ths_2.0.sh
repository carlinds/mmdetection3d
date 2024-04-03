# Run over different values of range_fraction
for confidence_ths in 0.1 0.3 0.5 0.7
do
  #sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/bevformer_orig.txt --results_b result_paths/bevformer_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/bevformer/per_sample --output_file result_paths/bevformer_agreement_output_files.txt --dist_ths 2.0 --confidence_ths $confidence_ths
  sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/bevformer_orig.txt --results_b result_paths/bevformer_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/bevformer/all --output_file result_paths/bevformer_agreement_output_files.txt --aggregate_all --dist_ths 2.0 --confidence_ths $confidence_ths
  #sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/bevformer_orig.txt --results_b result_paths/bevformer_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/bevformer/per_scene --output_file result_paths/bevformer_agreement_output_files.txt --aggregate_per_scene --dist_ths 2.0 --confidence_ths $confidence_ths

  #sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/petr_orig.txt --results_b result_paths/petr_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/petr/per_sample --output_file result_paths/petr_agreement_output_files.txt --dist_ths 2.0 --confidence_ths $confidence_ths
  sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/petr_orig.txt --results_b result_paths/petr_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/petr/all --output_file result_paths/petr_agreement_output_files.txt --aggregate_all --dist_ths 2.0 --confidence_ths $confidence_ths
  #sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/petr_orig.txt --results_b result_paths/petr_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/petr/per_scene --output_file result_paths/petr_agreement_output_files.txt --aggregate_per_scene --dist_ths 2.0 --confidence_ths $confidence_ths

  #sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/fcos3d_orig.txt --results_b result_paths/fcos3d_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/fcos3d/per_sample --output_file result_paths/fcos3d_agreement_output_files.txt --dist_ths 2.0 --confidence_ths $confidence_ths
  sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/fcos3d_orig.txt --results_b result_paths/fcos3d_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/fcos3d/all --output_file result_paths/fcos3d_agreement_output_files.txt --aggregate_all --dist_ths 2.0 --confidence_ths $confidence_ths
  #sbatch scripts/berzelius.sh python detection_agreement.py --results_a result_paths/fcos3d_orig.txt --results_b result_paths/fcos3d_nerf.txt --data_root /proj/adas-data/data/nuscenes --output_dir outputs/detection_agreement/fcos3d/per_scene --output_file result_paths/fcos3d_agreement_output_files.txt --aggregate_per_scene --dist_ths 2.0 --confidence_ths $confidence_ths
done
