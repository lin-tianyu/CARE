export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="../STEP1-AutoEncoderModel/klvae/vae_kl6_lr4_std/checkpoint-150000"
export UNET_MODEL_DIR="../STEP2-DiffusionModel/text_l2_concat_lr5/checkpoint-68000"
# STEP2-DiffusionModel/l2_cat_df2_noblur/checkpoint-14000
export TRAIN_DATA_DIR="dataset/datah5"

accelerate launch train_controlnet.py \
    --finetuned_vae_name_or_path=$FT_VAE_NAME \
    --trained_unet_name_or_path=$UNET_MODEL_DIR \
    --pretrained_model_name_or_path=$SD_MODEL_NAME \
    --train_data_dir=$TRAIN_DATA_DIR \
    --output_dir="logs/controlnet" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=100_000 \
    --validation_steps=1000 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=5 \
    --report_to=wandb \
    --validation_image dataset/datah5/3A4asPCs6GI_noncontrast/ct.h5 dataset/datah5/3A4asPCs6GI_portal-venous/ct.h5 dataset/datah5/3A4asPCs6GI_delayed/ct.h5 \
    --validation_prompt "A delayed phase CT slice." "A noncontrast phase CT slice." "A portal-venous phase CT slice." \
    --dataloader_num_workers=2 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=2