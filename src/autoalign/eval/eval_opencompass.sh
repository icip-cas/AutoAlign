export model_name=$1
export model_path=$2
export eval_type=$3
export batch_size=$4
export num_gpus=$5

# 检查是否有空值
if [ -z "$model_name" ] || [ -z "$model_path" ] || [ -z "$eval_type" ] || [ -z "$batch_size" ] || [ -z "$num_gpus" ]; then
    echo "存在空值，请检查输入参数，参数顺序为：
    model_name: 模型评测时使用的名称
    model_path: 模型路径
    eval_type: 评测类型
    batch_size: batch size, 单个模型使用的batch size
    num_gpus: 单个模型使用的gpu数量, 如当num_gpus=2，可用gpu数量为4时，将启动2个模型"
    exit 1
fi

# 检查是否存在opencompass文件夹
if [ ! -d "opencompass" ]; then
    echo "opencompass文件夹不存在，请检查opencompass是否已经安装"
    exit 1
fi

python eval_utils/generate_oc_config.py \
    --model_name $model_name \
    --model_path $model_path \
    --batch_size $batch_size \
    --num_gpus $num_gpus \
    --eval_type $eval_type

python opencompass/run.py opencompass/tmp_config/${model_name}_${eval_type}.py
