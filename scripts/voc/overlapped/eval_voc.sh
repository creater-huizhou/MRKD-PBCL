SAVEDIR='save_voc_MRKD_PBCL-w11'

python eval_voc.py -c ./configs/config_voc.json -d 0,1 --multiprocessing_distributed --name 'DKDSeg' \
--save_dir ${SAVEDIR} --task_name 15-1 --task_step 5 \
-r ./save_voc_MRKD_PBCL-w11/models/overlap_15-1_DKDSeg/step_5/checkpoint-epoch30.pth --test