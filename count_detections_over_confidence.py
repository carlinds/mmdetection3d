import json

root_dir = '/home/s0001038/Downloads/bevformer_res_complete/'
result_files = [
    ('Real data, no aug',
     'nuscenes_nuscenes_infos_temporal_val_clear_bevf_s_sweeps_ft_2e5/pts_bbox/results_nusc.json'
     ),
    ('Real data, img aug',
     'nuscenes_nuscenes_infos_temporal_val_clear_bevf_s_sweeps_ft_aug_4e5/pts_bbox/results_nusc.json'
     ),
    ('Real data, nerf aug',
     'nuscenes_nuscenes_infos_temporal_val_clear_bevf_s_sweeps_ft_nerf_4e5/pts_bbox/results_nusc.json'
     ),
    ('Real data, pix2pix aug',
     'nuscenes_nuscenes_infos_temporal_val_clear_bev_s_pix2pix_4e5_p05/pts_bbox/results_nusc.json'
     ),
    ('Sim data, no aug',
     'nusc_val_subset-neurader_no_keyframes_fullres_nuscenes_infos_temporal_val_clear_bevf_s_sweeps_ft_2e5/pts_bbox/results_nusc.json'
     ),
    ('Sim data, img aug,',
     'nusc_val_subset-neurader_no_keyframes_fullres_nuscenes_infos_temporal_val_clear_bevf_s_sweeps_ft_aug_4e5/pts_bbox/results_nusc.json'
     ),
    ('Sim data, nerf aug',
     'nusc_val_subset-neurader_no_keyframes_fullres_nuscenes_infos_temporal_val_clear_bevf_s_sweeps_ft_nerf_4e5/pts_bbox/results_nusc.json'
     ),
    ('Sim data, pix2pix',
     'nusc_val_subset-neurader_no_keyframes_fullres_nuscenes_infos_temporal_val_clear_bev_s_pix2pix_4e5_p05/pts_bbox/results_nusc.json'
     ),
]
for name, results_file in result_files:
    with open(root_dir + results_file) as f:
        results = json.load(f)
        results = results['results']

    confidence_threshold = 0.1
    n_detections = 0
    n_detections_over_confidence = 0
    n_samples = len(results)
    for sample_token in results:
        for box in results[sample_token]:
            n_detections += 1
            if box['detection_score'] >= confidence_threshold:
                n_detections_over_confidence += 1

    print(f'Results for {name}')
    print(f'Total number of samples: {n_samples}')
    print(f'Total number of detections: {n_detections}')
    print(
        f'Number of detections with confidence >= {confidence_threshold}: {n_detections_over_confidence}'
    )
    print(
        f'Average number of detections with confidence >= {confidence_threshold} per image: {n_detections_over_confidence/(n_samples*6)}'
    )
    print()
