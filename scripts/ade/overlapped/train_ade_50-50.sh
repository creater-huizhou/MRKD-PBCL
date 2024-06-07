#!/bin/bash

GPU=0,1
SAVEDIR='save_ade'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='50-50'
INIT_LR=0.0001
LR=0.00001
INIT_POSWEIGHT=5
INIT_EPOCHS=60
EPOCHS=20
BATCH_SIZE=6 # total size is 16
LR_SCHEDULER='PolyLR'
OPTIMIZER='AdamW'
MEMORY_SIZE=0 # 100 for DKD-MLR_SCHEDULER
SAMPLE_TYPE='cims'
MONITOR_SETTINGS='off'

NAME='DKDSeg'
#python train_ade.py -c configs/config_ade.json -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} \
#--name ${NAME} --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} \
#--pos_weight ${INIT_POSWEIGHT} --epochs ${INIT_EPOCHS} --batch_size ${BATCH_SIZE} --lr_scheduler ${LR_SCHEDULER} \
#--optimizer ${OPTIMIZER}

python train_ade.py -c configs/config_ade.json -d ${GPU} --multiprocessing_distributed  --save_dir ${SAVEDIR} \
--name ${NAME} --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} \
--mem_size ${MEMORY_SIZE} --sample_type ${SAMPLE_TYPE} --monitor ${MONITOR_SETTINGS} --pos_weight ${INIT_POSWEIGHT} \
--epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr_scheduler ${LR_SCHEDULER} --optimizer ${OPTIMIZER} \
--flag_MRKD --flag_PBCL

python train_ade.py -c configs/config_ade.json -d ${GPU} --multiprocessing_distributed  --save_dir ${SAVEDIR} \
--name ${NAME} --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} \
--mem_size ${MEMORY_SIZE} --sample_type ${SAMPLE_TYPE} --monitor ${MONITOR_SETTINGS} --pos_weight ${INIT_POSWEIGHT} \
--epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr_scheduler ${LR_SCHEDULER} --optimizer ${OPTIMIZER} \
--flag_MRKD --flag_PBCL