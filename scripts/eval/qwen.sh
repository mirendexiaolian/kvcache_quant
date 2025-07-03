ratios=(0.1)
methods=("sparsemm")
budgets=( 1024 2048)
mask_ratio=0.1 # only used for "mask" / "mask_random"

for budget in ${budgets[@]}; do
    for ratio in ${ratios[@]}; do
        for method in ${methods[@]}; do
    
            export METHOD=${method}
            export BUDGET=${budget}
            export RATIO=${ratio}
            export MASK_RATIO=${mask_ratio}

            mkdir -p ./ocrbench_results/qwen_results/

            export CUDA_VISIBLE_DEVICES=2,3,5,6
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
                --verbosity=DEBUG 2>&1 | tee ./ocrbench_results/qwen_results/ocrbench_${method}_${budget}_${ratio}.log
        done
    done
done
