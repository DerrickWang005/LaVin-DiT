set -x

export PYTHONPATH="."

QUERY_IMAGE="test_sample/hinton.png"
OUTPUT="hinton.png"

python inference.py \
    --query $QUERY_IMAGE \
    --output result_depth_$OUTPUT \
    --task_dir template/depth_estimation \
    --height 512 \
    --width 512

python inference.py \
    --query $QUERY_IMAGE \
    --output result_normal_$OUTPUT \
    --task_dir template/normal_estimation \
    --height 512 \
    --width 512

python inference.py \
    --query $QUERY_IMAGE \
    --output result_detection_$OUTPUT \
    --task_dir template/object_detection \
    --height 512 \
    --width 512

python inference.py \
    --query $QUERY_IMAGE \
    --output result_pseg_$OUTPUT \
    --task_dir template/panoptic_segmentation \
    --height 512 \
    --width 512

python inference.py \
    --query $QUERY_IMAGE \
    --output result_fseg_$OUTPUT \
    --task_dir template/foreground_segmentation0 \
    --height 512 \
    --width 512

python inference.py \
    --query $QUERY_IMAGE \
    --output result_pose_$OUTPUT \
    --task_dir template/pose_estimation \
    --height 512 \
    --width 512

python inference.py \
    --query $QUERY_IMAGE \
    --output result_edge_$OUTPUT \
    --task_dir template/edge_detection \
    --height 512 \
    --width 512
