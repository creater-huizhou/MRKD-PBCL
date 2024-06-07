SAVEDIR='save_ade_MSRD_PBCL-3'
EPOCHS=20

python eval_ade.py -c ./configs/config_ade.json -d 0,1 --multiprocessing_distributed --save_dir ${SAVEDIR} --name 'DKDSeg' \
--task_name 100-50 --task_step 1 --epochs ${EPOCHS} \
-r ./save_ade_MSRD_PBCL-3/models/overlap_100-50_DKDSeg/step_1/checkpoint-epoch20.pth --test