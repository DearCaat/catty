# # 单loss 不同初始阶段epoch
# CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 0
# CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 1
# CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2
# CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 5
# #        不同测试方法及阈值
# CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2 RDD_TRANS.TEST_MAX_POOL True RDD_TRANS.TEST_THR 0.99
# CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2 RDD_TRANS.TEST_THR 0.99
# #        不同阈值
# CUDA_VISIBLE_DEVICES=0 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2 RDD_TRANS.THR_ABS_DIS_ True RDD_TRANS.THR_ABS_NOR_ False
CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2 RDD_TRANS.THR_ABS_DIS_ True RDD_TRANS.THR_REL_ True

CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2 RDD_TRANS.THR_ABS_UPDATE_NAME fix

CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2 RDD_TRANS.THR_ABS_UPDATE_NAME linear

CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2 RDD_TRANS.THR_ABS_UPDATE_NAME sigmod_iter

CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2 RDD_TRANS.FILTER_SAMPLES 1

CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2 RDD_TRANS.CLASSIFY_LOSS 1

CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2 RDD_TRANS.CLASSIFY_LOSS 0.1