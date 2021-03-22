#! /bin/bash
cp extract.py ~/mmdetection/tools/.
GPU_NUM=2 # ${GPU_NUM}
CONFIG_FILE=/auto/phd/09/igor/mmdetection/configs/i2102/210218JPG_MM4G1_Pretrained_UNetResNet50.py
CHECKPOINT=/tmp2/igor/checkpoints/PUR50_epoch_107.pth
# ${CONFIG_FILE}
EXPERIMENT_DIR=$(pwd)
pwd
echo "Navigating to mmdet..."
cd ~/mmdetection/tools
pwd
python extract.py ${CONFIG_FILE} ${CHECKPOINT} --output_dir=${EXPERIMENT_DIR} --dataset_dir=/tmp2/igor/LL-tSNE/resized