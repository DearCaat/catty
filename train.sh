python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --title=cub_swin --cfg=/home/tangwenhao/rdd/rdd_transformer/configs/cub_fgvc.yaml --opts TRAIN.OPTIMIZER.NAME lookahead_adamw

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --title=cub_swin --cfg=/home/tangwenhao/rdd/rdd_transformer/configs/cub_fgvc.yaml --opts TRAIN.OPTIMIZER.NAME adamw

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --title=cub_swin --cfg=/home/tangwenhao/rdd/rdd_transformer/configs/cub_fgvc.yaml --opts TRAIN.OPTIMIZER.NAME adamw TRAIN.BASE_LR 1e-4

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --title=cub_swin --cfg=/home/tangwenhao/rdd/rdd_transformer/configs/cub_fgvc.yaml --opts TRAIN.OPTIMIZER.NAME adamw MODEL.LABEL_SMOOTHING .1

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --title=cub_swin --cfg=/home/tangwenhao/rdd/rdd_transformer/configs/cub_fgvc.yaml --opts TRAIN.OPTIMIZER.NAME adamw AUG.COLOR_JITTER .4

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --title=cub_swin --cfg=/home/tangwenhao/rdd/rdd_transformer/configs/cub_fgvc.yaml --opts TRAIN.OPTIMIZER.NAME adamw AUG.REPROB .25

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --title=cub_swin --cfg=/home/tangwenhao/rdd/rdd_transformer/configs/cub_fgvc.yaml --opts TRAIN.OPTIMIZER.NAME adamw TRAIN.WEIGHT_DECAY 0.005

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --title=cub_swin --cfg=/home/tangwenhao/rdd/rdd_transformer/configs/cub_fgvc.yaml --opts TRAIN.OPTIMIZER.NAME adamw TRAIN.WEIGHT_DECAY 0.0005

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --title=cub_swin --cfg=/home/tangwenhao/rdd/rdd_transformer/configs/cub_fgvc.yaml --opts TRAIN.OPTIMIZER.NAME adamw TRAIN.WEIGHT_DECAY 0.00005
