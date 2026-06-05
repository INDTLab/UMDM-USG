def log_validation(accelerator, config, model, logger, step, device):
    torch.cuda.empty_cache()
    model = accelerator.unwrap_model(model).eval()
    null_y = torch.load(null_embed_path, map_location="cpu")
    null_y = null_y["uncond_prompt_embeds"].to(device)

    logger.info("Running validation... ")

    # Run sampling
    latents = []
    for prompt in validation_prompts:
        z = torch.randn(1, 1 + num_conditions, config.vae.vae_latent_dim, latent_size, latent_size, device=device)
        embed = torch.load(
            osp.join(config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"),
            map_location="cpu",
        )
        caption_embs, emb_masks = embed["caption_embeds"].to(device), embed["emb_mask"].to(device)
        role = torch.zeros((1, 1 + num_conditions), dtype=torch.long, device=device)
        model_kwargs = dict(mask=emb_masks, role=role)
        dpm_solver = DPMS(
            model.forward,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=4.5,
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
        )
        denoised = dpm_solver.sample(
            z,
            steps=20,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=config.scheduler.flow_shift,
        )
        latents.append(denoised)
    torch.cuda.empty_cache()

    # Decode latents
    image_logs = []
    for prompt, latent in zip(validation_prompts, latents):
        latent = latent.to(torch.float16)
        latent = torch.unbind(latent, dim=1)
        images = []
        for lat in latent:
            sample = vae_decode(config.vae.vae_type, vae, lat)
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255)
            sample = sample.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
            images.append(Image.fromarray(sample))
        image_logs.append({"validation_prompt": prompt, "images": images})

    # Save images
    def concatenate_images(image_caption, images_per_row=5, image_format="webp"):
        import io
        images = list(itertools.chain.from_iterable([log["images"] for log in image_caption]))
        if images[0].size[0] > 1024:
            images = [image.resize((1024, 1024)) for image in images]
        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights[i : i + images_per_row][0] for i in range(0, len(images), images_per_row))
        new_im = Image.new("RGB", (max_width * images_per_row, total_height))
        y_offset = 0
        for i in range(0, len(images), images_per_row):
            row_images = images[i : i + images_per_row]
            x_offset = 0
            for img in row_images:
                new_im.paste(img, (x_offset, y_offset))
                x_offset += max_width
            y_offset += heights[i]
        webp_image_bytes = io.BytesIO()
        new_im.save(webp_image_bytes, format=image_format)
        webp_image_bytes.seek(0)
        new_im = Image.open(webp_image_bytes)
        return new_im

    file_format = "png"  # "webp"
    local_vis_save_path = osp.join(config.work_dir, "log_vis")
    os.umask(0o000)
    os.makedirs(local_vis_save_path, exist_ok=True)
    concatenated_image = concatenate_images(image_logs, images_per_row=num_conditions+1, image_format=file_format)
    save_path = osp.join(local_vis_save_path, f"vis_{step}.{file_format}")
    concatenated_image.save(save_path)

    model.train()
    flush()
    return image_logs
    
    
    
    
    log_validation(
          accelerator=accelerator,
          config=config,
          model=model,
          logger=logger,
          step=global_step,
          device=accelerator.device,
                    )
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    