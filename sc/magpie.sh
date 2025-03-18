
model_path=/
total_prompts=100
ins_topp=0.9
ins_temp=0.7
config=/
model_id=ii
device="0"
tensor_parallel=1
gpu_memory_utilization=0.8
n=1

# Get Current Time
timestamp=$(date +%s)

# Generate Pretty Name
job_name="$ii_topp$0.9_temp$0.7_$"#job_name="$bruh_topp$0.9_temp$0.7_$"

### Setup Logging
log_dir="data"
if [ ! -d "../$log_dir" ]; then
    mkdir -p "../$log_dir"
fi

job_path="../$log_dir/$job_name"

mkdir -p $job_path
exec > >(tee -a "$job_path/$job_name.log") 2>&1
echo "[magpie.sh] Model Name: $model_path"
echo "[magpie.sh] Pretty name: $job_name"
echo "[magpie.sh] Total Prompts: $total_prompts"
echo "[magpie.sh] Instruction Generation Config: temp=$ins_temp, top_p=$ins_topp"
echo "[magpie.sh] System Config: device=$device, n=$n, tensor_parallel=$tensor_parallel"
echo "[magpie.sh] Timestamp: $timestamp"
echo "[magpie.sh] Job Name: $job_name"

echo "[magpie.sh] Start Generating Instructions..."
CUDA_VISIBLE_DEVICES=$device python src/autoalign/data/instruction/magpie.py \
    --device $device \
    --model_path $model_path \
    --total_prompts $total_prompts \
    --top_p $ins_topp \
    --temperature $ins_temp \
    --tensor_parallel $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --n $n \
    --job_name $job_name \
    --timestamp $timestamp \
    --model-id $model_id \
    --config $config

echo "[magpie.sh] Finish Generating Instructions!"
