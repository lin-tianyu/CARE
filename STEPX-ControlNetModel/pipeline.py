from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, AutoencoderKL, UNet2DConditionModel, ControlNetModel

from diffusers.models import ImageProjection, MultiControlNetModel
from diffusers.utils.torch_utils import randn_tensor, is_compiled_module, is_torch_version
from diffusers import DDIMScheduler, DDPMScheduler
import torch
import torch.nn as nn
from PIL import Image
import PIL
import nibabel as nib 
import os
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils import is_torch_xla_available
from types import SimpleNamespace
import safetensors
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


def init_unet(trained_unet_name_or_path, zero_cond_conv_in=False):
    # 加载预训练模型
    unet = UNet2DConditionModel.from_pretrained(
        trained_unet_name_or_path, subfolder="unet",
    )
    
    # double conv_in channel size, 
    # half with pretrained weight for input, 
    # half with zeros for cond
    if zero_cond_conv_in:   
        # 获取原始输入卷积层
        original_conv = unet.conv_in
        original_in_channels = original_conv.in_channels

        # 创建新卷积层（输入通道翻倍）
        new_conv = nn.Conv2d(
            in_channels=original_in_channels * 2,  # 输入通道翻倍
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            dilation=original_conv.dilation,
            groups=original_conv.groups,
            bias=original_conv.bias is not None
        )

        # 参数初始化
        with torch.no_grad():
            # 初始化新权重张量
            new_weight = torch.zeros_like(
                new_conv.weight[:, :original_in_channels*2, :, :],
                device=new_conv.weight.device,
                dtype=new_conv.weight.dtype
            )
            
            # 前一半通道使用预训练参数
            new_weight[:, :original_in_channels] = original_conv.weight
            
            # 后一半通道保持0初始化（默认已经是0，这里显式强调）
            new_weight[:, original_in_channels:] = 0.0
            
            new_conv.weight.copy_(new_weight)
            
            # 复制偏置参数
            if new_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        # 替换模型中的卷积层
        unet.conv_in = new_conv
        unet.config.in_channels = original_in_channels * 2
    return unet


class ConcatInputStableDiffusionControlNetPipeline(StableDiffusionControlNetPipeline): 
    # NOTE: COPIED from diffusers repo
    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        cond_latents = None, # NOTE: added for concating a image's latents
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted
                as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or
                width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`,
                images must be passed as a list such that each element of the list can be correctly batched for input
                to a single ControlNet. When `prompt` is a list, and if a list of images is passed for a single
                ControlNet, each will be paired with each prompt in the `prompt` list. This also applies to multiple
                ControlNets, where a list of image lists can be passed to batch for each prompt and each ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            # Nested lists as ControlNet condition
            if isinstance(image[0], list):
                # Transpose the nested image list
                image = [list(t) for t in zip(*image)]

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                cond_latent_input = torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents  # NOTE: added
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    # control_model_input,
                    torch.cat([latent_model_input, cond_latent_input], dim=1),   # NOTE: concate input!!!
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and self.do_classifier_free_guidance:
                    # Inferred ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    # latent_model_input,
                    torch.cat([latent_model_input, cond_latent_input], dim=1),   # NOTE: concate input!!!
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()
        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


    # # NOTE: COPIED from diffusers repo
    # @torch.no_grad()
    # # @replace_example_docstring(EXAMPLE_DOC_STRING)
    # def __call__(
    #     self,
    #     prompt: Union[str, List[str]] = None,
    #     height: Optional[int] = None,
    #     width: Optional[int] = None,
    #     num_inference_steps: int = 50,
    #     timesteps: List[int] = None,
    #     sigmas: List[float] = None,
    #     guidance_scale: float = 7.5,
    #     negative_prompt: Optional[Union[str, List[str]]] = None,
    #     num_images_per_prompt: Optional[int] = 1,
    #     eta: float = 0.0,
    #     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    #     latents: Optional[torch.Tensor] = None,
    #     cond_latents = None, # NOTE: added for concating a image's latents
    #     prompt_embeds: Optional[torch.Tensor] = None,
    #     negative_prompt_embeds: Optional[torch.Tensor] = None,
    #     ip_adapter_image = None,
    #     ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    #     output_type: Optional[str] = "pil",
    #     return_dict: bool = True,
    #     cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    #     guidance_rescale: float = 0.0,
    #     clip_skip: Optional[int] = None,
    #     callback_on_step_end: Optional[
    #         Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    #     ] = None,
    #     callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    #     **kwargs,
    # ):
    #     r"""
    #     The call function to the pipeline for generation.

    #     Args:
    #         prompt (`str` or `List[str]`, *optional*):
    #             The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
    #         height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
    #             The height in pixels of the generated image.
    #         width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
    #             The width in pixels of the generated image.
    #         num_inference_steps (`int`, *optional*, defaults to 50):
    #             The number of denoising steps. More denoising steps usually lead to a higher quality image at the
    #             expense of slower inference.
    #         timesteps (`List[int]`, *optional*):
    #             Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
    #             in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
    #             passed will be used. Must be in descending order.
    #         sigmas (`List[float]`, *optional*):
    #             Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
    #             their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
    #             will be used.
    #         guidance_scale (`float`, *optional*, defaults to 7.5):
    #             A higher guidance scale value encourages the model to generate images closely linked to the text
    #             `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
    #         negative_prompt (`str` or `List[str]`, *optional*):
    #             The prompt or prompts to guide what to not include in image generation. If not defined, you need to
    #             pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
    #         num_images_per_prompt (`int`, *optional*, defaults to 1):
    #             The number of images to generate per prompt.
    #         eta (`float`, *optional*, defaults to 0.0):
    #             Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
    #             to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
    #         generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
    #             A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
    #             generation deterministic.
    #         latents (`torch.Tensor`, *optional*):
    #             Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
    #             generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
    #             tensor is generated by sampling using the supplied random `generator`.
    #         cond_latents (`torch.Tensor`):
    #             Pre-generated latents of the target image. Used to guide diffusion process for controlable generation.
    #         prompt_embeds (`torch.Tensor`, *optional*):
    #             Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
    #             provided, text embeddings are generated from the `prompt` input argument.
    #         negative_prompt_embeds (`torch.Tensor`, *optional*):
    #             Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
    #             not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
    #         ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
    #         ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
    #             Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
    #             IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
    #             contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
    #             provided, embeddings are computed from the `ip_adapter_image` input argument.
    #         output_type (`str`, *optional*, defaults to `"pil"`):
    #             The output format of the generated image. Choose between `PIL.Image` or `np.array`.
    #         return_dict (`bool`, *optional*, defaults to `True`):
    #             Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
    #             plain tuple.
    #         cross_attention_kwargs (`dict`, *optional*):
    #             A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
    #             [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
    #         guidance_rescale (`float`, *optional*, defaults to 0.0):
    #             Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
    #             Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
    #             using zero terminal SNR.
    #         clip_skip (`int`, *optional*):
    #             Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
    #             the output of the pre-final layer will be used for computing the prompt embeddings.
    #         callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
    #             A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
    #             each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
    #             DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
    #             list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
    #         callback_on_step_end_tensor_inputs (`List`, *optional*):
    #             The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
    #             will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
    #             `._callback_tensor_inputs` attribute of your pipeline class.

    #     Examples:

    #     Returns:
    #         [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
    #             If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
    #             otherwise a `tuple` is returned where the first element is a list with the generated images and the
    #             second element is a list of `bool`s indicating whether the corresponding generated image contains
    #             "not-safe-for-work" (nsfw) content.
    #     """

    #     callback = kwargs.pop("callback", None)
    #     callback_steps = kwargs.pop("callback_steps", None)

    #     if callback is not None:
    #         deprecate(
    #             "callback",
    #             "1.0.0",
    #             "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
    #         )
    #     if callback_steps is not None:
    #         deprecate(
    #             "callback_steps",
    #             "1.0.0",
    #             "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
    #         )

    #     if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
    #         callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    #     # 0. Default height and width to unet
    #     if not height or not width:
    #         height = (
    #             self.unet.config.sample_size
    #             if self._is_unet_config_sample_size_int
    #             else self.unet.config.sample_size[0]
    #         )
    #         width = (
    #             self.unet.config.sample_size
    #             if self._is_unet_config_sample_size_int
    #             else self.unet.config.sample_size[1]
    #         )
    #         height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
    #     # to deal with lora scaling and other possible forward hooks

    #     # 1. Check inputs. Raise error if not correct
    #     self.check_inputs(
    #         prompt,
    #         height,
    #         width,
    #         callback_steps,
    #         negative_prompt,
    #         prompt_embeds,
    #         negative_prompt_embeds,
    #         ip_adapter_image,
    #         ip_adapter_image_embeds,
    #         callback_on_step_end_tensor_inputs,
    #     )

    #     self._guidance_scale = guidance_scale
    #     self._guidance_rescale = guidance_rescale
    #     self._clip_skip = clip_skip
    #     self._cross_attention_kwargs = cross_attention_kwargs
    #     self._interrupt = False

    #     # 2. Define call parameters
    #     if prompt is not None and isinstance(prompt, str):
    #         batch_size = 1
    #     elif prompt is not None and isinstance(prompt, list):
    #         batch_size = len(prompt)
    #     else:
    #         batch_size = prompt_embeds.shape[0]

    #     device = self._execution_device

    #     # 3. Encode input prompt
    #     lora_scale = (
    #         self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    #     )

    #     prompt_embeds, negative_prompt_embeds = self.encode_prompt(
    #         prompt,
    #         device,
    #         num_images_per_prompt,
    #         self.do_classifier_free_guidance,
    #         negative_prompt,
    #         prompt_embeds=prompt_embeds,
    #         negative_prompt_embeds=negative_prompt_embeds,
    #         lora_scale=lora_scale,
    #         clip_skip=self.clip_skip,
    #     )

    #     # For classifier free guidance, we need to do two forward passes.
    #     # Here we concatenate the unconditional and text embeddings into a single batch
    #     # to avoid doing two forward passes
    #     if self.do_classifier_free_guidance:
    #         prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    #     if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
    #         image_embeds = self.prepare_ip_adapter_image_embeds(
    #             ip_adapter_image,
    #             ip_adapter_image_embeds,
    #             device,
    #             batch_size * num_images_per_prompt,
    #             self.do_classifier_free_guidance,
    #         )

    #     # 4. Prepare timesteps
    #     timesteps, num_inference_steps = retrieve_timesteps(
    #         self.scheduler, num_inference_steps, device, timesteps, sigmas
    #     )

    #     # 5. Prepare latent variables
    #     num_channels_latents = self.unet.config.in_channels
    #     latents = self.prepare_latents(
    #         batch_size * num_images_per_prompt,
    #         num_channels_latents,
    #         height,
    #         width,
    #         prompt_embeds.dtype,
    #         device,
    #         generator,
    #         latents,
    #     )

    #     # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    #     extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    #     # 6.1 Add image embeds for IP-Adapter
    #     added_cond_kwargs = (
    #         {"image_embeds": image_embeds}
    #         if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
    #         else None
    #     )

    #     # 6.2 Optionally get Guidance Scale Embedding
    #     timestep_cond = None
    #     if self.unet.config.time_cond_proj_dim is not None:
    #         guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
    #         timestep_cond = self.get_guidance_scale_embedding(
    #             guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
    #         ).to(device=device, dtype=latents.dtype)

    #     # 7. Denoising loop
    #     num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    #     self._num_timesteps = len(timesteps)
    #     with self.progress_bar(total=num_inference_steps) as progress_bar:
    #         for i, t in enumerate(timesteps):
    #             if self.interrupt:
    #                 continue

    #             # expand the latents if we are doing classifier free guidance
    #             latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
    #             cond_latent_input = torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents  # NOTE: added
    #             latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

    #             # predict the noise residual
    #             noise_pred = self.unet(
    #                 torch.cat([latent_model_input, cond_latent_input], dim=1),   # NOTE: concate input!!!
    #                 t,
    #                 encoder_hidden_states=prompt_embeds,
    #                 timestep_cond=timestep_cond,
    #                 cross_attention_kwargs=self.cross_attention_kwargs,
    #                 added_cond_kwargs=added_cond_kwargs,
    #                 return_dict=False,
    #             )[0]

    #             # perform guidance
    #             if self.do_classifier_free_guidance:
    #                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #                 noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

    #             if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
    #                 # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
    #                 noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

    #             # compute the previous noisy sample x_t -> x_t-1
    #             latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

    #             if callback_on_step_end is not None:
    #                 callback_kwargs = {}
    #                 for k in callback_on_step_end_tensor_inputs:
    #                     callback_kwargs[k] = locals()[k]
    #                 callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

    #                 latents = callback_outputs.pop("latents", latents)
    #                 prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
    #                 negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

    #             # call the callback, if provided
    #             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
    #                 progress_bar.update()
    #                 if callback is not None and i % callback_steps == 0:
    #                     step_idx = i // getattr(self.scheduler, "order", 1)
    #                     callback(step_idx, t, latents)

    #             if XLA_AVAILABLE:
    #                 xm.mark_step()

    #     if not output_type == "latent":
    #         image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
    #             0
    #         ]
    #         image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    #     else:
    #         image = latents
    #         has_nsfw_concept = None

    #     if has_nsfw_concept is None:
    #         do_denormalize = [True] * image.shape[0]
    #     else:
    #         do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    #     image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    #     # Offload all models
    #     self.maybe_free_model_hooks()

    #     if not return_dict:
    #         return (image, has_nsfw_concept)

    #     return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


# class NoNoiseImg2ImgPipeline(StableDiffusionImg2ImgPipeline):
#     """No Noise: means not adding noise to the input image to start reverse process"""  # NOTE: image input range [0, 1]
#     # NOTE: COPIED from the `StableDiffusionImg2ImgPipeline` in diffusers
#     def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
#         if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
#             raise ValueError(
#                 f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
#             )

#         image = image.to(device=device, dtype=dtype)

#         batch_size = batch_size * num_images_per_prompt

#         if image.shape[1] == 4:
#             init_latents = image

#         else:
#             if isinstance(generator, list) and len(generator) != batch_size:
#                 raise ValueError(
#                     f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
#                     f" size of {batch_size}. Make sure the batch size matches the length of the generators."
#                 )

#             elif isinstance(generator, list):
#                 if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
#                     image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
#                 elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
#                     raise ValueError(
#                         f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
#                     )

#                 init_latents = [
#                     retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
#                     for i in range(batch_size)
#                 ]
#                 init_latents = torch.cat(init_latents, dim=0)
#             else:
#                 init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

#             init_latents = self.vae.config.scaling_factor * init_latents

#         if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
#             # expand init_latents for batch_size
#             deprecation_message = (
#                 f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
#                 " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
#                 " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
#                 " your script to pass as many initial images as text prompts to suppress this warning."
#             )
#             deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
#             additional_image_per_prompt = batch_size // init_latents.shape[0]
#             init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
#         elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
#             raise ValueError(
#                 f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
#             )
#         else:
#             init_latents = torch.cat([init_latents], dim=0)

#         shape = init_latents.shape
#         noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
#         # NOTE: delete the above 2 lines and the below 2 lines to eliminate noise adding for img2img pipeline
#         # get latents
#         init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
#         latents = init_latents

#         return latents

if __name__ == "__main__":
    """Method 1: StableDiffusionPipeline"""
    # 初始化Pipeline
    finetuned_vae_name_or_path = "../STEP1-AutoEncoderModel/klvae/vae_kl6_lr4_std/checkpoint-150000"
    finetuned_unet_name_or_path = "../STEP2-DiffusionModel/text_l2_concat_lr5/checkpoint-48000"
    data_dir = "/mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/"
    vae = AutoencoderKL.from_pretrained(
            finetuned_vae_name_or_path, subfolder="vae", #revision=args.revision, variant=args.variant,
            torch_dtype=torch.float16
        )
    # unet = UNet2DConditionModel.from_pretrained(
    #     finetuned_unet_name_or_path, subfolder="unet", #revision=args.non_ema_revision,
    #     torch_dtype=torch.float16
    # )
    args = SimpleNamespace(pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5")
    unet = init_unet(args, zero_cond_conv_in=True)
    unet_ckpt = safetensors.torch.load_file(os.path.join(finetuned_unet_name_or_path, "unet", "diffusion_pytorch_model.safetensors"))
    unet.load_state_dict(unet_ckpt, strict=True)
    unet = unet.half()

    pipe = ConcatInputStableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        unet=unet,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)  # 使用DDIM调度器
    pipe = pipe.to("cuda")

    # 输入参数
    bdmap_id = "BDMAP_00000001"
    slice_id = 150
    nii = nib.load(os.path.join(data_dir, bdmap_id, "ct.nii.gz"))
    nii_slice = nii.dataobj[:, :, slice_id:slice_id+3].copy()
    nii_slice[nii_slice > 1000] = 1000
    nii_slice[nii_slice < -1000] = -1000
    nii_slice = (nii_slice + 1000) / 2000
    input_image = nii_slice * 255
    input_image = Image.fromarray((input_image*1).astype(np.uint8))
    input_image.save(f"../input_{bdmap_id}_{slice_id}.png")

    prompt = "A arterial phase CT slice."  # 提示词

    # --- Step 1: 图像编码为潜变量 ---
    image = (torch.from_numpy(np.asarray(input_image).copy())[None].permute(0, 3, 1, 2) / 255).to("cuda").half() * 2 - 1
    with torch.no_grad():
        cond_latents = pipe.vae.encode(image).latent_dist.sample() * 0.18215
        latents = torch.randn_like(cond_latents)


    # --- Step 3: 执行推理 ---
    image = pipe(
        num_inference_steps=500, 
        prompt=prompt,
        latents=cond_latents,  # 直接传入原图的潜变量
        cond_latents=cond_latents
    ).images[0]

    image.save(f"../output_addnoise_{bdmap_id}_{slice_id}.png")


    # """Method 2: StableDiffusionImg2ImgPipeline"""
    # # 初始化自定义Pipeline
    # finetuned_vae_name_or_path = "../STEP1-AutoEncoderModel/klvae/vae_kl6_lr4_std/checkpoint-150000"
    # finetuned_unet_name_or_path = "../STEP2-DiffusionModel/text_l1_concat_lr5/checkpoint-48000"
    # data_dir = "/mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/"
    # vae = AutoencoderKL.from_pretrained(
    #         finetuned_vae_name_or_path, subfolder="vae", #revision=args.revision, variant=args.variant,
    #         torch_dtype=torch.float16
    #     )
    # unet = UNet2DConditionModel.from_pretrained(
    #     finetuned_unet_name_or_path, subfolder="unet", #revision=args.non_ema_revision,
    #     torch_dtype=torch.float16
    # )
    # # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    # pipe = NoNoiseImg2ImgPipeline.from_pretrained(
    #     "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    #     unet=unet,
    #     vae=vae,
    #     safety_checker=None,
    #     torch_dtype=torch.float16)
    # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)  # 使用DDIM调度器
    # pipe = pipe.to("cuda")

    # # 输入参数
    # bdmap_id = "BDMAP_00000001"
    # slice_id = 150
    # nii = nib.load(os.path.join(data_dir, bdmap_id, "ct.nii.gz"))
    # nii_slice = nii.dataobj[:, :, slice_id:slice_id+3].copy()
    # nii_slice[nii_slice > 1000] = 1000
    # nii_slice[nii_slice < -1000] = -1000
    # nii_slice = (nii_slice + 1000) / 2000
    # input_image = nii_slice * 255
    # input_image = Image.fromarray((input_image*1).astype(np.uint8))
    # input_image.save(f"../input_{bdmap_id}_{slice_id}.png")
    # # input_image = Image.open("../example.png").convert("RGB")  # 输入图像

    # prompt = 'A portal-venous phase CT slice.'  # 提示词
    # strength = 0.04  # 控制生成图像与原图的相似度（0~1，越小越接近原图）

    # # --- Step 1: 图像预处理 ---
    # image = (torch.from_numpy(np.asarray(input_image).copy())[None].permute(0, 3, 1, 2) / 255).to("cuda").half()
    # # image = (pipe.image_processor.preprocess(input_image, height=512, width=512)/2 + 1/2).to("cuda").half()
    # print(image.min(), image.max())

    # def callback_on_step_end(pipe, step, timestep, callback_kwargs):
    #     if step % 20 == 0:
    #         print(step, timestep, callback_kwargs.keys())
    #         latents = callback_kwargs["latents"]
    #         with torch.no_grad():
    #             image = pipe.decode_latents(latents)
    #             # 将图像转换为PIL格式并保存 
    #             image = pipe.numpy_to_pil(image)[0]
    #             image.save(f"pics/intermediate_timestep_{timestep}.png")
    #     return callback_kwargs

    # # --- Step 2: 执行推理 ---
    # image = pipe(
    #     prompt=prompt,
    #     image=image,  # 输入图像
    #     strength=strength,  # 控制生成图像与原图的相似度
    #     # num_inference_steps=1000,
    #     # callback_on_step_end=callback_on_step_end
    # ).images[0]

    # image.save(f"../output_addnoise_{bdmap_id}_{slice_id}.png")