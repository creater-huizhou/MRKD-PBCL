#!/bin/bash

GPU=0,1
SAVEDIR='save_voc_MRKD-w1'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='15-1'
INIT_LR=0.0001
LR=0.00001
INIT_POSWEIGHT=2
POSWEIGHT=1
INIT_EPOCHS=30
EPOCHS=30
BATCH_SIZE=12
LR_SCHEDULER='PolyLR'
OPTIMIZER='AdamW'
MEMORY_SIZE=0 # 100 for DKD-M
SAMPLE_TYPE='cims'
MONITOR_SETTINGS='off'

NAME='DKDSeg'
#python train_voc.py -c configs/config_voc.json -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} \
#--name ${NAME} --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} \
#--pos_weight ${INIT_POSWEIGHT} --epochs ${INIT_EPOCHS} --batch_size ${BATCH_SIZE} \
#--lr_scheduler ${LR_SCHEDULER} --optimizer ${OPTIMIZER} --freeze

python train_voc.py -c configs/config_voc.json -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} \
--name ${NAME} --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} \
--pos_weight ${POSWEIGHT} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --monitor ${MONITOR_SETTINGS} \
--lr_scheduler ${LR_SCHEDULER} --optimizer ${OPTIMIZER} --freeze --flag_MRKD

python train_voc.py -c configs/config_voc.json -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} \
--name ${NAME} --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} \
--pos_weight ${POSWEIGHT} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --monitor ${MONITOR_SETTINGS} \
--lr_scheduler ${LR_SCHEDULER} --optimizer ${OPTIMIZER} --freeze --flag_MRKD

python train_voc.py -c configs/config_voc.json -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} \
--name ${NAME} --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} \
--pos_weight ${POSWEIGHT} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --monitor ${MONITOR_SETTINGS} \
--lr_scheduler ${LR_SCHEDULER} --optimizer ${OPTIMIZER} --freeze --flag_MRKD

python train_voc.py -c configs/config_voc.json -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} \
--name ${NAME} --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} \
--pos_weight ${POSWEIGHT} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --monitor ${MONITOR_SETTINGS} \
--lr_scheduler ${LR_SCHEDULER} --optimizer ${OPTIMIZER} --freeze --flag_MRKD

python train_voc.py -c configs/config_voc.json -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} \
--name ${NAME} --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} \
--pos_weight ${POSWEIGHT} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --monitor ${MONITOR_SETTINGS} \
--lr_scheduler ${LR_SCHEDULER} --optimizer ${OPTIMIZER} --freeze --flag_MRKD