#DATA_PATH="/mnt/d/wsl/data/cqu_bpdd"
DATA_PATH="/raid/Data/zhangyi/cqu_bpdd/"
if [ $1 -eq 0 ]; then
    mv $DATA_PATH/test/normal $DATA_PATH/tenormal
    mv $DATA_PATH/train/normal $DATA_PATH/trnormal
    mv $DATA_PATH/val/normal $DATA_PATH/vanormal
elif [ $1 -eq 1 ]; then
    mv $DATA_PATH/tenormal $DATA_PATH/test/normal
    mv $DATA_PATH/trnormal $DATA_PATH/train/normal
    mv $DATA_PATH/vanormal $DATA_PATH/val/normal 
fi

