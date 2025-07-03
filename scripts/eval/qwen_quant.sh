methods=("headquant")
mkdir -p ./ocrbench_results/qwen_results/
for method in ${methods[@]}; do
    export CUDA_VISIBLE_DEVICES=2,5,6,7
    export METHOD=${method}
    python3 -m accelerate.commands.launch \
        --num_processes=4 \
        --main_process_port 54321\
        -m lmms_eval \
        --model qwen2_vl \
        --model_args pretrained=/obs/users/guixiyan/qwen2vl/ \
        --tasks ocrbench \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix qwen2-vl \
        --output_path ./logs/ \
        --gen_kwargs temperature=0 \
        --verbosity=DEBUG 2>&1 | tee ./ocrbench_results/qwen_results/ocrbench_${method}.log
    done 