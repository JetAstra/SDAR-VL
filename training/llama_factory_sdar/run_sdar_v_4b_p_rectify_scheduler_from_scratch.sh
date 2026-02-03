# for communication
export NCCL_P2P_DISABLE=0       # 启用 P2P 直接通信
export NCCL_IB_DISABLE=0        # 启用 InfiniBand 支持
export NCCL_IB_HCA=mlx5_0       # 指定 InfiniBand 接口（需要核对你的硬件）
export NCCL_IB_TIMEOUT=22       # 增加 timeout 的时间
export NCCL_DEBUG=INFO          # 输出详细的 NCCL 调试信息
export PYTHONWARNINGS=ignore    # 忽略 NCCL 的 Future 警告


# for debug, main reduce training efficiency
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
export NCCL_TIMEOUT=3600000
# export NCCL_DEBUG=INFO
# export TORCHDYNAMO_DISABLE=1
wandb offline

# run
cd /mnt/shared-storage-user/chengshuang/projects/mdllm/llama_factory_sdar
dpkg -l | grep nccl

NODE_COUNT=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${PROC_PER_NODE:-4}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-6000}

# # stage 1, run with four nodes.
# LOG_FILE="./logs/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage1/node_${NODE_RANK}.log"
# LOG_DIR=$(dirname "${LOG_FILE}")
# mkdir -p "${LOG_DIR}"

# torchrun \
#     --nnodes ${NODE_COUNT} \
#     --node_rank ${NODE_RANK} \
#     --nproc_per_node ${NPROC_PER_NODE} \
#     --master_addr ${MASTER_ADDR} \
#     --master_port ${MASTER_PORT} \
#     ./src/llamafactory/launcher.py \
#     ./examples/train_full/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_full_sft_stage1.yaml 2>&1 | tee ${LOG_FILE}

# cp /mnt/shared-storage-user/chengshuang/projects/modelZoo/sdar_v_4b_hf/modeling_sdar.py /mnt/shared-storage-user/dllm-share/chengshuang/sdar_v_ckpt/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage1

# # stage 2_1, run with four nodes.
# LOG_FILE="./logs/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage2_1/node_${NODE_RANK}.log"
# LOG_DIR=$(dirname "${LOG_FILE}")
# mkdir -p "${LOG_DIR}"

# torchrun \
#     --nnodes ${NODE_COUNT} \
#     --node_rank ${NODE_RANK} \
#     --nproc_per_node ${NPROC_PER_NODE} \
#     --master_addr ${MASTER_ADDR} \
#     --master_port ${MASTER_PORT} \
#     ./src/llamafactory/launcher.py \
#     ./examples/train_full/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_full_sft_stage2_1.yaml 2>&1 | tee ${LOG_FILE}

# cp /mnt/shared-storage-user/chengshuang/projects/modelZoo/sdar_v_4b_hf/modeling_sdar.py /mnt/shared-storage-user/dllm-share/chengshuang/sdar_v_ckpt/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage2_1

# # stage 2_2, run with four nodes.
# LOG_FILE="./logs/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage2_2/node_${NODE_RANK}.log"
# LOG_DIR=$(dirname "${LOG_FILE}")
# mkdir -p "${LOG_DIR}"

# torchrun \
#     --nnodes ${NODE_COUNT} \
#     --node_rank ${NODE_RANK} \
#     --nproc_per_node ${NPROC_PER_NODE} \
#     --master_addr ${MASTER_ADDR} \
#     --master_port ${MASTER_PORT} \
#     ./src/llamafactory/launcher.py \
#     ./examples/train_full/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_full_sft_stage2_2.yaml 2>&1 | tee ${LOG_FILE}

# cp /mnt/shared-storage-user/chengshuang/projects/modelZoo/sdar_v_4b_hf/modeling_sdar.py /mnt/shared-storage-user/dllm-share/chengshuang/sdar_v_ckpt/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage2_2

# stage 3_1, run with four nodes.
LOG_FILE="./logs/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage3_1/node_${NODE_RANK}.log"
LOG_DIR=$(dirname "${LOG_FILE}")
mkdir -p "${LOG_DIR}"

torchrun \
    --nnodes ${NODE_COUNT} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    ./src/llamafactory/launcher.py \
    ./examples/train_full/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_full_sft_stage3_1.yaml 2>&1 | tee ${LOG_FILE}

cp /mnt/shared-storage-user/chengshuang/projects/modelZoo/sdar_v_4b_hf/modeling_sdar.py /mnt/shared-storage-user/dllm-share/chengshuang/sdar_v_ckpt/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage3_1

# stage 3_2, run with four nodes.
LOG_FILE="./logs/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage3_2/node_${NODE_RANK}.log"
LOG_DIR=$(dirname "${LOG_FILE}")
mkdir -p "${LOG_DIR}"

torchrun \
    --nnodes ${NODE_COUNT} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    ./src/llamafactory/launcher.py \
    ./examples/train_full/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_full_sft_stage3_2.yaml 2>&1 | tee ${LOG_FILE}

cp /mnt/shared-storage-user/chengshuang/projects/modelZoo/sdar_v_4b_hf/modeling_sdar.py /mnt/shared-storage-user/dllm-share/chengshuang/sdar_v_ckpt/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage3_2

# stage 4_1, run with four nodes.
LOG_FILE="./logs/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage4_1/node_${NODE_RANK}.log"
LOG_DIR=$(dirname "${LOG_FILE}")
mkdir -p "${LOG_DIR}"

torchrun \
    --nnodes ${NODE_COUNT} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    ./src/llamafactory/launcher.py \
    ./examples/train_full/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_full_sft_stage4_1.yaml 2>&1 | tee ${LOG_FILE}

cp /mnt/shared-storage-user/chengshuang/projects/modelZoo/sdar_v_4b_hf/modeling_sdar.py /mnt/shared-storage-user/dllm-share/chengshuang/sdar_v_ckpt/sdar_v_4b_p_rectify_scheduler_from_scratch/sdar_v_stage4_1
