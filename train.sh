python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=resnet50_7 --model-name=resnet50 --thumb --train-mode=predict --resume=/mnt/d/wsl/output/resnet50_7/model/resnet50_best_model.pth --opts MODEL.NUM_CLASSES 7
python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=vgg16_7 --model-name=vgg16 --thumb --train-mode=predict --resume=/mnt/d/wsl/output/vgg16_7/model/vgg16_best_model.pth --opts MODEL.NUM_CLASSES 7 TRAIN.LR_SCHEDULER.NAME 'cosine' TRAIN.OPTIMIZER.NAME 'lookahead_adamw'
python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=inception_v3_7 --model-name=tf_inception_v3 --thumb --opts DATA.IMG_SIZE '(299,299)' MODEL.NUM_CLASSES 7 --train-mode=predict --resume=/mnt/d/wsl/output/inception_v3_7/model/tf_inception_v3_best_model.pth

python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=vit_s_7 --model-name=deit_small_patch16_224 --thumb --train-mode=predict --resume=/mnt/d/wsl/output/vit_s_7/model/deit_small_patch16_224_best_model.pth --opts MODEL.NUM_CLASSES 7

python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=vit_b_7 --model-name=deit_base_patch16_224 --thumb --train-mode=predict --resume=/mnt/d/wsl/output/vit_b_7/model/deit_base_patch16_224_best_model.pth --opts MODEL.NUM_CLASSES 7

# python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=inception_v3_bin --model-name=tf_inception_v3 --thumb --binary-train --opts DATA.IMG_SIZE '(299,299)'
#python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=effi_bin --model-name=tf_efficientnet_b3 --thumb --binary-train --opts DATA.IMG_SIZE '(300,300)'

# python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=resnet50_7 --model-name=resnet50 --thumb --opts MODEL.NUM_CLASSES 7 DATA.VAL_SPLIT 'test' --epochs=20
# python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=vgg16_7 --model-name=vgg16 --thumb --opts MODEL.NUM_CLASSES 7 TRAIN.LR_SCHEDULER.NAME 'cosine' TRAIN.OPTIMIZER.NAME 'lookahead_adamw' DATA.VAL_SPLIT 'test'

# python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=inception_v3_7 --model-name=tf_inception_v3 --thumb --opts DATA.IMG_SIZE '(299,299)' MODEL.NUM_CLASSES 7 DATA.VAL_SPLIT 'test'

# python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=vit_s_7 --model-name=deit_small_patch16_224 --thumb --opts MODEL.NUM_CLASSES 7 TRAIN.BASE_LR '1e-4' TRAIN.LR_SCHEDULER.NAME 'cosine' TRAIN.OPTIMIZER.NAME 'lookahead_adamw' DATA.VAL_SPLIT 'test' --epochs=20

# python3 main.py --data-path=/mnt/d/wsl/data/cqu_bpdd/ --output=/mnt/d/wsl/output/ --title=vit_b_7 --model-name=deit_base_patch16_224 --thumb --opts MODEL.NUM_CLASSES 7 TRAIN.BASE_LR '1e-4' TRAIN.LR_SCHEDULER.NAME 'cosine' TRAIN.OPTIMIZER.NAME 'lookahead_adamw' DATA.VAL_SPLIT 'test' --epochs=20

