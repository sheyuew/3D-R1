export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=1

# LoRA training script for 3D-R1
# This script enables LoRA for efficient fine-tuning

python main.py \
    --ngpus 1 \
    --checkpoint_dir /root/autodl-tmp/checkpoints/fast_eval \
    --dataset scenecold_dataset \
    --vocab "/root/autodl-tmp/Qwen2.5-VL-3B-Instruct" \
    --qformer_vocab "/root/autodl-tmp/bert-base-uncased" \
    --captioner 3dr1 \
    --detector point_encoder \
    --max_des_len 256 \
    --max_gen_len 64 \
    --batchsize_per_gpu 4 \
    --base_lr 1e-4 \
    --final_lr 5e-6 \
    --weight_decay 0.01 \
    --clip_gradient 0.5 \
    --warm_lr 1e-5 \
    --warm_lr_epochs 1 \
    --max_epoch 15 \
    --log_every 10 \
    --save_every 5000 \
    --eval_every_iteration 2000 \
    --start_eval_after 400000 \
    --criterion "CiDEr" \
    --use_color \
    --max_multiview_images 4 \
    --max_multiview_depth 4 \
    --depth_encoder_dim 256 \
    --image_encoder_dim 256 \
    --use_lora \
    --use_multimodal_model \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --view_selection_weight 0.05 \
    --eval_batch_size 8 \
    --eval_max_samples 100 \
    --eval_use_fp16 \
    --eval_skip_metrics \
    --seed 42 \
    --dataset_num_workers 8 \
    2>&1 | tee run_log.txt
    # --use_multiview \
    # --use_additional_encoders \
    # --use_depth \
    # --use_image \
    # --enable_dynamic_views

/usr/bin/shutdown