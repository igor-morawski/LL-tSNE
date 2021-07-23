#! /bin/bash
cp extract.py ~/mmdetection/tools/.
GPU_ID=$1
# CONFIG_FILE=/auto/phd/09/igor/mmdetection/configs/i2102/210218JPG_MM4G1_ResNet50.py
# CHECKPOINT=/tmp2/igor/checkpoints/S_R50_epoch_121.pth
# CONFIG_FILE=/auto/phd/09/igor/mmdetection/configs/i2102/210218JPG_MM4G1_Pretrained_UNetResNet50.py
# CHECKPOINT=/tmp2/igor/checkpoints/S_PUR50_epoch_126.pth
# CONFIG_FILE=/auto/phd/09/igor/mmdetection/configs/i2102/210218JPG_MM4G1_ResNet50.py
# CHECKPOINT=/tmp2/igor/checkpoints/COCO_R50_epoch_1.pth
# CONFIG_FILE=/auto/phd/09/igor/mmdetection/configs/i2102/210218JPG_MM4G1_Pretrained_UNetResNet50.py
# CHECKPOINT=/tmp2/igor/checkpoints/S_PURS50_epoch_112.pth

# ${CONFIG_FILE}
EXPERIMENT_DIR=$(pwd)
pwd
echo "Navigating to mmdet..."
cd ~/mmdetection/tools
pwd

CONFIG_FILE=/auto/phd/09/igor/mmdetection/configs/i2102/210218JPG_MM4G1_ResNet50.py
CHECKPOINT=/tmp2/igor/checkpoints/COCO_R50_epoch_1.pth
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=0 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=1 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=2 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=3 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=4 --part=neck

CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=0 
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=1
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=2 
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=3 

CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=0 --part=unet
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=1 --part=unet
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=2 --part=unet



CONFIG_FILE=/auto/phd/09/igor/mmdetection/configs/i2102/210218JPG_MM4G1_ResNet50.py
CHECKPOINT=/tmp2/igor/checkpoints/S_R50_epoch_121.pth
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=0 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=1 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=2 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=3 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=4 --part=neck

CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=0 
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=1
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=2 
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=8 --feature_level=3 

CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=0 --part=unet
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=1 --part=unet
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=2 --part=unet


CONFIG_FILE=/auto/phd/09/igor/mmdetection/configs/i2102/210218JPG_MM4G1_Pretrained_UNetResNet50.py
CHECKPOINT=/tmp2/igor/checkpoints/S_PUR50_epoch_126.pth
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=0 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=1 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=2 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=3 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=4 --part=neck

CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=0 
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=1
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=2 
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=3 

CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=0 --part=unet
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=1 --part=unet
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=2 --part=unet


CONFIG_FILE=/auto/phd/09/igor/mmdetection/configs/i2102/210218JPG_MM4G1_Pretrained_UNetResNet50.py
CHECKPOINT=/tmp2/igor/checkpoints/S_PURS50_epoch_112.pth
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=0 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=1 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=2 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=3 --part=neck
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=4 --part=neck

CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=0 
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=1
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=2 
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=2 --feature_level=3 

CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=0 --part=unet
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=1 --part=unet
CUDA_VISIBLE_DEVICES=$GPU_ID python extract.py ${CONFIG_FILE} ${CHECKPOINT} /tmp2/igor/LL-tSNE/testset_resized/ --output_dir=${EXPERIMENT_DIR}/features_testset --samples_per_gpu=1 --feature_level=2 --part=unet
