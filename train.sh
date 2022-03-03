
python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --output=/mnt/f/wsl/output/ --title=cub_swin_small --cfg=./configs/cub_fgvc_small.yaml

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --output=/mnt/f/wsl/output/ --title=cub_swin_small --cfg=./configs/cub_fgvc_small.yaml --opts DATA.INTERPOLATION bicubic

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --output=/mnt/f/wsl/output/ --title=cub_swin_small --cfg=./configs/cub_fgvc_small.yaml --opts DATA.INTERPOLATION random

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --output=/mnt/f/wsl/output/ --title=cub_swin_small --cfg=./configs/cub_fgvc_small.yaml --opts MODEL.LABEL_SMOOTHING 0.1

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --output=/mnt/f/wsl/output/ --title=cub_swin_small --cfg=./configs/cub_fgvc_small.yaml --opts AUG.COLOR_JITTER 0.4

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --output=/mnt/f/wsl/output/ --title=cub_swin_small --cfg=./configs/cub_fgvc_small.yaml --opts AUG.REPROB 0.25

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --output=/mnt/f/wsl/output/ --title=cub_swin_small --cfg=./configs/cub_fgvc_small.yaml --opts TRAIN.CLIP_GRAD 5.0

python3 main.py --data-path=/mnt/d/wsl/data/CUB/data/ --output=/mnt/f/wsl/output/ --title=cub_swin_small --cfg=./configs/cub_fgvc_small.yaml --opts TRAIN.ACCUMULATION_STEPS 2