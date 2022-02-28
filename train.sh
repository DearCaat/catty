CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cub/data/ --output=/home/tangwenhao/output/ --title=cub_swin --cfg=/home/tangwenhao/rdd_code/rdd_transformer/configs/cub_fgvc.yaml --opts MODEL_EMA True TRAIN.WEIGHT_DECAY 0.0

CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cub/data/ --output=/home/tangwenhao/output/ --title=cub_swin --cfg=/home/tangwenhao/rdd_code/rdd_transformer/configs/cub_fgvc.yaml --opts MODEL_EMA True TRAIN.WEIGHT_DECAY 0.005

CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cub/data/ --output=/home/tangwenhao/output/ --title=cub_swin --cfg=/home/tangwenhao/rdd_code/rdd_transformer/configs/cub_fgvc.yaml --opts MODEL_EMA True TRAIN.WEIGHT_DECAY 0.0005

CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cub/data/ --output=/home/tangwenhao/output/ --title=cub_swin --cfg=/home/tangwenhao/rdd_code/rdd_transformer/configs/cub_fgvc.yaml --opts MODEL_EMA True TRAIN.BASE_LR 0.0001

CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cub/data/ --output=/home/tangwenhao/output/ --title=cub_swin --cfg=/home/tangwenhao/rdd_code/rdd_transformer/configs/cub_fgvc.yaml --opts MODEL_EMA True AUG.COLOR_JITTER .4

CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cub/data/ --output=/home/tangwenhao/output/ --title=cub_swin --cfg=/home/tangwenhao/rdd_code/rdd_transformer/configs/cub_fgvc.yaml --opts MODEL_EMA True MODEL.LABEL_SMOOTHING .1

CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cub/data/ --output=/home/tangwenhao/output/ --title=cub_swin --cfg=/home/tangwenhao/rdd_code/rdd_transformer/configs/cub_fgvc.yaml --opts MODEL_EMA True AUG.REPROB .25

CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cub/data/ --output=/home/tangwenhao/output/ --title=cub_swin --cfg=/home/tangwenhao/rdd_code/rdd_transformer/configs/cub_fgvc.yaml --opts MODEL_EMA True AUG.AUTO_AUGMENT rand-m5-n2-mstd0.5

CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cub/data/ --output=/home/tangwenhao/output/ --title=cub_swin --cfg=/home/tangwenhao/rdd_code/rdd_transformer/configs/cub_fgvc.yaml --opts MODEL_EMA True AUG.AUTO_AUGMENT rand-m5-n4-mstd0.5