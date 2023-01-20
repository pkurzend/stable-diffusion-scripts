4d3
< import random
7a7,8
> import random
> import shutil
15,16d15
< from torch.utils.data import Dataset
< 
20,21d18
< from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
< from diffusers.optimization import get_scheduler
23a21
> from torch.utils.data import Dataset
27a26,28
> from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
>                        StableDiffusionInpaintPipeline, UNet2DConditionModel)
> from diffusers.optimization import get_scheduler
53c54
<         default=None,
---
>         default="fp16",
87,92c88,93
<     parser.add_argument(
<         "--save_sample_prompt",
<         type=str,
<         default=None,
<         help="The prompt used to generate sample outputs to save.",
<     )
---
>     # parser.add_argument(
>     #     "--save_sample_prompt",
>     #     type=str,
>     #     default=None,
>     #     help="The prompt used to generate sample outputs to save.",
>     # )
238c239
<         default=None,
---
>         default="no",
241,243c242,244
<             "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
<             " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
<             " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
---
>             "Whether to use mixed precision. Choose"
>             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
>             "and an Nvidia Ampere GPU."
267a269,292
> def get_cutout_holes(height, width, min_holes=8, max_holes=32, min_height=16, max_height=128, min_width=16, max_width=128):
>     holes = []
>     for _n in range(random.randint(min_holes, max_holes)):
>         hole_height = random.randint(min_height, max_height)
>         hole_width = random.randint(min_width, max_width)
>         y1 = random.randint(0, height - hole_height)
>         x1 = random.randint(0, width - hole_width)
>         y2 = y1 + hole_height
>         x2 = x1 + hole_width
>         holes.append((x1, y1, x2, y2))
>     return holes
> 
> 
> def generate_random_mask(image):
>     mask = torch.zeros_like(image[:1])
>     holes = get_cutout_holes(mask.shape[1], mask.shape[2])
>     for (x1, y1, x2, y2) in holes:
>         mask[:, y1:y2, x1:x2] = 1.
>     if random.uniform(0, 1) < 0.25:
>         mask.fill_(1.)
>     masked_image = image * (mask < 0.5)
>     return mask, masked_image
> 
> 
326a352
>         example["instance_masks"], example["instance_masked_images"] = generate_random_mask(example["instance_images"])
339a366
>             example["class_masks"], example["class_masked_images"] = generate_random_mask(example["class_images"])
448c475
<                     pipeline = StableDiffusionPipeline.from_pretrained(
---
>                     pipeline = StableDiffusionInpaintPipeline.from_pretrained(
452d478
<                             subfolder=None if args.pretrained_vae_name_or_path else "vae",
471c497,500
<                 with  torch.inference_mode(): # torch.autocast("cuda"),
---
>                 inp_img = Image.new("RGB", (512, 512), color=(0, 0, 0))
>                 inp_mask = Image.new("L", (512, 512), color=255)
> 
>                 with torch.autocast("cuda"),torch.inference_mode():
475c504,509
<                         images = pipeline(example["prompt"]).images
---
>                         images = pipeline(
>                             prompt=example["prompt"],
>                             image=inp_img,
>                             mask_image=inp_mask,
>                             num_inference_steps=args.save_infer_steps
>                         ).images
570a605,606
>         mask_values = [example["instance_masks"] for example in examples]
>         masked_image_values = [example["instance_masked_images"] for example in examples]
576a613,614
>             mask_values += [example["class_masks"] for example in examples]
>             masked_image_values += [example["class_masked_images"] for example in examples]
578,579c616,618
<         pixel_values = torch.stack(pixel_values)
<         pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
---
>         pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
>         mask_values = torch.stack(mask_values).to(memory_format=torch.contiguous_format).float()
>         masked_image_values = torch.stack(masked_image_values).to(memory_format=torch.contiguous_format).float()
589a629,630
>             "mask_values": mask_values,
>             "masked_image_values": masked_image_values
594c635
<         train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
---
>         train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=8
682c723
<                 text_enc_model = accelerator.unwrap_model(text_encoder)
---
>                 text_enc_model = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
685,686c726,727
<             scheduler =  DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
<             pipeline = StableDiffusionPipeline.from_pretrained(
---
>             scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
>             pipeline = StableDiffusionInpaintPipeline.from_pretrained(
688,689c729,730
<                 unet=accelerator.unwrap_model(unet),
<                 text_encoder=text_enc_model,
---
>                 unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True).to(torch.float16),
>                 text_encoder=text_enc_model.to(torch.float16),
705,706c746,750
<             if args.save_sample_prompt is not None:
<                 pipeline = pipeline.to(accelerator.device)
---
>             shutil.copy("train_inpainting_dreambooth.py", save_dir)
> 
>             pipeline = pipeline.to(accelerator.device)
>             pipeline.set_progress_bar_config(disable=True)
>             for idx, concept in enumerate(args.concepts_list):
708,709c752
<                 pipeline.set_progress_bar_config(disable=True)
<                 sample_dir = os.path.join(save_dir, "samples")
---
>                 sample_dir = os.path.join(save_dir, "samples", str(idx))
711c754,756
<                 with torch.inference_mode(): # torch.autocast("cuda"), 
---
>                 inp_img = Image.new("RGB", (512, 512), color=(0, 0, 0))
>                 inp_mask = Image.new("L", (512, 512), color=255)
>                 with torch.inference_mode():
714c759,761
<                             args.save_sample_prompt,
---
>                             prompt=concept["instance_prompt"],
>                             image=inp_img,
>                             mask_image=inp_mask,
721,723c768,770
<                 del pipeline
<                 if torch.cuda.is_available():
<                     torch.cuda.empty_cache()
---
>             del pipeline
>             if torch.cuda.is_available():
>                 torch.cuda.empty_cache()
724a772,773
>             unet.to(torch.float32)
>             text_enc_model.to(torch.float32)
735a785
>         random.shuffle(train_dataset.class_images_path)
743a794
>                         masked_latent_dist = vae.encode(batch["masked_image_values"].to(dtype=weight_dtype)).latent_dist
744a796,797
>                     masked_image_latents = masked_latent_dist.sample() * 0.18215
>                     mask = F.interpolate(batch["mask_values"], scale_factor=1 / 8)
766a820
>                 latent_model_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
768,776c822
<                 noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
< 
<                 # Get the target for loss depending on the prediction type
<                 if noise_scheduler.config.prediction_type == "epsilon":
<                     noise = noise
<                 elif noise_scheduler.config.prediction_type == "v_prediction":
<                     noise = noise_scheduler.get_velocity(latents, noise, timesteps)
<                 else:
<                     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
---
>                 noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
