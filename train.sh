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
# 讨论初始化阶段
CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 1
CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 2
CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 3
CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 5
CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 10
# 讨论multi view
CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 3
AUG.MULTI_VIEW strong_weak
CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 3
AUG.MULTI_VIEW strong_none
CUDA_VISIBLE_DEVICES=1 python3 main.py --data-path=/home/tangwenhao/data/cqu_bpdd/ --output=/home/tangwenhao/output/ --title=swin_multi_view_persudo_noclus --opts RDD_TRANS.INIT_STAGE_EPOCH 3
AUG.MULTI_VIEW waek_none
