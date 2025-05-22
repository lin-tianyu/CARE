# Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?

Official PyTorch implementation of **CARE**, a framework that improves sparse-view or low-dose CT reconstructions by enforcing anatomical completeness. CARE works as a modular post-processing pipeline, transforming initial reconstructions (e.g., from NeRF, TensoRF, FDK) into anatomically faithful CT volumes using latent diffusion and segmentation-driven supervision.

---

## Method

CARE consists of three stages:

1. **Autoencoder Compression**: A KL-VAE encodes high-quality CTs into a latent space.
2. **Latent Diffusion Denoising**: A diffusion model refines noisy latent codes from initial reconstructions.
3. **Anatomy-Aware Finetuning**: Segmentations from nnUNet guide the final reconstruction with anatomy-aware losses (NSD, clDice, etc).

CARE is model-agnostic and compatible with any upstream reconstruction method.

---

## Environment Setup

We recommend using `conda`:

```bash
conda create -n care python=3.11 -y
conda activate care
pip install -r requirements.txt
```
Requirements include:
	•	torch >= 2.1
	•	diffusers == 0.32.2
	•	nnunetv2 == 2.6.0
	•	surface-distance
	•	scikit-image, numpy, einops, wandb, etc.

## Data Preparation
	1.	Download the dataset: Use the AbdomenAtlas or any well-labeled abdominal CT dataset.
	2.	Preprocess using nnUNet v2:

`nnUNetv2_extract_dataset /path/to/abdomenatlas 500`
`nnUNetv2_plan_and_preprocess -d 500 -c 3d_fullres`

	3.	Symlink the processed dataset:

`ln -s /path/to/nnunet_preprocessed/Task500_AbdomenAtlas datasets/Task500_AbdomenAtlas`




## Training

Stage 1: Autoencoder

`python STEP1-AutoEncoderModel/train_vae.py --cfg configs/vae.yaml`

Stage 2: Diffusion

`python STEP2-DiffusionModel/train_diffusion.py --cfg configs/diff.yaml --resume checkpoints/vae.pth`

Stage 3: CARE Finetune

`python STEP3-CAREModel/train_care.py --cfg configs/care.yaml --resume checkpoints/diff.pth`

Each script supports:
	•	WandB logging
	•	Resuming from checkpoints
	•	Mixed precision with --fp16
	•	Configurable batch size and eval interval



## Inference

To enhance a reconstruction:

```
python STEP3-CAREModel/infer.py \
    --ckpt checkpoints/care.pth \
    --input path/to/recon_volume.nii.gz \
    --output path/to/enhanced_volume.nii.gz
```

Results are printed (PSNR, SSIM, NSD, clDice) and optionally logged to Weights & Biases.



## Evaluation

We evaluate on:
	•	Pixel-wise: PSNR, SSIM
	•	Surface-based: NSD (Normalized Surface Dice), MSD, HD95
	•	Topology-aware: clDice for tubular structures

These metrics better reflect clinical relevance than traditional pixel-wise scores.

To run the paper’s full benchmark:

bash scripts/paper_pipeline.sh




## Pretrained Models

You can download pre-trained weightsn via: [https://drive.google.com/drive/folders/1MbiI2T1aJm06fP-f9qHvZ8uu58cPRTCx?usp=sharing](https://drive.google.com/drive/folders/1MbiI2T1aJm06fP-f9qHvZ8uu58cPRTCx?usp=sharing)

Place them under checkpoints/.

## Results

\**CARE** substantially improves structural completeness in reconstructed CT scans, yielding performance gains of up to **+32%** for large organs, **22%** for small organs, **40%** for intestines, and **36%** for vessels. 




## License

This repository is licensed under the Apache 2.0 License.
Please review licenses of dependencies (Diffusers, nnUNet, surface-distance, etc.)



## Acknowledgements

We thank the creators of:
	•	nnUNet-v2
	•	HuggingFace Diffusers
	•	AbdomenAtlas dataset
	•	surface-distance metrics



If you use this code, please consider citing our work. Thank you!

---

Let me know if you'd like this exported to a file, or want to embed images like training curves, architecture overviews, or anatomy diagrams.
