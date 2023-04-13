#!/bin/bash

# 使用 `"$@"' 来让每个命令行参数扩展为一个单独的单词。 `$@' 周围的引号是必不可少的！
# 使用 getopt 整理参数
ARGS=$(getopt -o 'h:p:t:d:c:b:v:m::leo:' -l 'host:,project:,title:,dataset:,config:,batch-size:,val-batch-size:,multi-gpu::,log-wandb,ema,pin-memo,no-amp,no-val,option:' -- "$@")

if [ $? != 0 ] ; then echo "Parse error! Terminating..." >&2 ; exit 1 ; fi

# 将参数设置为 getopt 整理后的参数
# $ARGS 需要用引号包围
eval set -- "$ARGS"
CONN_BATCH_SIZE=''
CONN_MULTI_GPU=1
# 循环解析参数
while true ; do
     # 从第一个参数开始解析
     case "$1" in
          # 主机
          -h|--host) CONN_HOST="$2" ; shift 2 ;;
          # project
          -p|--project) CONN_PROJECT="$2" ; shift 2 ;;
          # title
          -t|--title) CONN_TITLE="$2" ; shift 2 ;;
          # 数据集名称
          -d|--dataset) CONN_DATASET="$2" ; shift 2 ;;
          # 配置文件，多个config，多次-c
          -c|--config) CONN_CONFIG+=("$2") ; shift 2 ;;
          # 单GPU的BS
          -b|--batch-size) CONN_BATCH_SIZE="$2" ; shift 2 ;;
          # 单GPU的BS
          -v|--val-batch-size) CONN_VAL_BATCH_SIZE="$2" ; shift 2 ;;
          # 多GPU,默认为单GPU，如使用多GPU，则指定GPU数目
          -m|--multi-gpu)
               case "$2" in
                    "") CONN_MULTI_GPU=1 ; shift 2 ;;
                    *)  CONN_MULTI_GPU=$2 ; shift 2 ;;
               esac ;;
          -l|--log-wandb) CONN_LOG_WANDB=true ; shift ;;
          -e|--ema) CONN_EMA=true ; shift ;;
          --pin-memo) CONN_PIN_MEMO=true ; shift ;;
          --no-amp) CONN_NO_AMP=true ; shift ;;
          --no-val) CONN_NO_VAL=true ; shift ;;
          # --opt 
          -o|--option) CONN_OPT="$*" ; shift 2 ;;
          --) shift ; break ;;
          *) echo "$2" ; shift 2 ;;
     esac
done

func_list2str(){
    local _arr
    local _str
    _str=''
    _arr=($(echo "$@"))
    for _i in ${_arr[@]};do
        _str=$_str' '$_i;
    done
    echo "$_str"
}

arr_opt=($CONN_OPT)
unset arr_opt[0]
unset arr_opt[2]
for s in ${CONN_CONFIG[@]};do
    echo "$s"
done
opt=$(func_list2str ${arr_opt[*]})
config=$(func_list2str ${CONN_CONFIG[*]})

# 显示获取参数结果
# echo '用户名：    '  "$CONN_USERNAME"
echo 'host:       '  "$CONN_HOST"
echo 'project:    '  "$CONN_PROJECT"
echo 'title:      '  "$CONN_TITLE"
echo 'dataset:    '  "$CONN_DATASET"
echo 'multi-gpu:  '  "$CONN_MULTI_GPU"
echo 'log-wandb:  '  "$CONN_LOG_WANDB"
echo 'ema:  '  "$CONN_EMA"
echo 'configs:    '  "$config"
echo 'batch-size: '  "$CONN_BATCH_SIZE"
echo 'val-batch-size: '  "$CONN_VAL_BATCH_SIZE"
echo 'pin-memory: '  "$CONN_PIN_MEMO"
echo 'no-amp: '      "$CONN_NO_AMP"
echo 'no-val: '      "$CONN_NO_VAL"
echo 'options:     ' "$opt"

# 处理True\False的选项
if [ $CONN_LOG_WANDB ]; then
    log_wandb_str="--log-wandb"
else
    log_wandb_str=""
fi
if [ $CONN_EMA ]; then
    ema_str="--ema"
else
    ema_str=""
fi
if [ $CONN_PIN_MEMO ]; then
    p_m_str="--pin-memory"
else
    p_m_str=""
fi
if [ $CONN_NO_AMP ]; then
    n_amp_str="--no-amp"
else
    n_amp_str=""
fi
if [ $CONN_NO_VAL ]; then
    n_val_str="--no-val"
else
    n_val_str=""
fi
case "$CONN_MULTI_GPU" in
    1) multi_gpu_str='';;
    '') multi_gpu_str='';;
    *) multi_gpu_str="-m torch.distributed.launch --nproc_per_node="$CONN_MULTI_GPU
esac

if [ -z $CONN_BATCH_SIZE ];then
    bs_str=''
else
    bs_str='--batch-size='$CONN_BATCH_SIZE
fi

if [ -z $CONN_VAL_BATCH_SIZE ];then
    vbs_str=''
else
    vbs_str='--val-batch-size='$CONN_VAL_BATCH_SIZE
fi

# 根据不同主机，处理不同的数据集文件夹和输出文件夹
case "$CONN_HOST" in
    "3090") data_path="/data/tangwenhao/pdc/"; output_path="/data/tangwenhao/output/";git pull origin master;;
    "amax") data_path="/data/zhangxiaoxian/pdc/"; output_path="/nas/zhangxiaoxian/output/"; git pull origin master;;
    "DGX") data_path="/raid/Data/zhangyi/pdc/"; output_path="/raid/Data/zhangyi/output/";git pull origin master;;
    "3090_1") data_path="/home/tangwenhao/dataset/pdc/"; output_path="/data/tangwenhao/output/";;
    "2080") data_path="/home/tangwenhao/data/pdc/"; output_path="/home/tangwenhao/output/";git pull origin master;;
    *) echo "Error Host!"; exit ;;
esac

# 处理--opt
if [ -z "$opt" -a -z "$extra_opt" ]; then
    opt_str=''
else
    opt=' '$opt
    extra_opt=' '$extra_opt
    opt_str='--opt'$opt$extra_opt
fi

python3 $multi_gpu_str ../main.py --data-path=$data_path$CONN_DATASET"/data/" --output=$output_path --project=$CONN_PROJECT --cfg $config $bs_str $vbs_str --title=$CONN_TITLE $log_wandb_str $ema_str $p_m_str $n_amp_str $n_val_str $opt_str