import os
import gc
import json
from dataclasses import dataclass
from typing import Optional

import streamlit as st
import torch
from diffusers import DiffusionPipeline
from loguru import logger
from PIL.PngImagePlugin import PngInfo

from diffuzers import utils

@dataclass
class Text2Image:
    device: Optional[str] = None
    model: Optional[str] = None
    output_path: Optional[str] = None

    def __str__(self) -> str:
        return f"Text2Image(model={self.model}, device={self.device}, output_path={self.output_path})"

    def __post_init__(self):
        # Set environment variable to avoid memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.pipeline.to(self.device)
        self.pipeline.safety_checker = utils.no_safety_checker

        # Enable memory-efficient attention and slicing if using CUDA
        if self.device == "cuda":
            self.pipeline.unet.enable_gradient_checkpointing()
            self.pipeline.enable_memory_efficient_attention()
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_xformers_memory_efficient_attention()
            # Warmup with minimal steps
            prompt = "a photo of an astronaut riding a horse on mars"
            _ = self.pipeline(prompt, num_inference_steps=2)
            # Clear cache after warmup
            torch.cuda.empty_cache()
            gc.collect()

    def _set_scheduler(self, scheduler_name):
        scheduler = self.pipeline.scheduler.compatibles[scheduler_name].from_config(self.pipeline.scheduler.config)
        self.pipeline.scheduler = scheduler

    def generate_image(self, prompt, negative_prompt, scheduler, image_size, num_images, guidance_scale, steps, seed):
        self._set_scheduler(scheduler)
        logger.info(self.pipeline.scheduler)
        
        if self.device == "cuda":
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        output_images = []
        for _ in range(num_images):
            generated_image = self.pipeline(
                prompt,
                negative_prompt=negative_prompt,
                width=image_size[1],
                height=image_size[0],
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images
            output_images.extend(generated_image)
            
            # Clear cache after each image generation
            torch.cuda.empty_cache()
            gc.collect()
        
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scheduler": scheduler,
            "image_size": image_size,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": seed,
        }
        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text("text2img", metadata)

        utils.save_images(
            images=output_images,
            module="text2img",
            metadata=metadata,
            output_path=self.output_path,
        )

        return output_images, _metadata

    def app(self):
        available_schedulers = list(self.pipeline.scheduler.compatibles.keys())
        if "EulerAncestralDiscreteScheduler" in available_schedulers:
            available_schedulers.insert(
                0, available_schedulers.pop(available_schedulers.index("EulerAncestralDiscreteScheduler"))
            )
        
        col1, col2 = st.columns(2)
        with col1:
            prompt = st.text_area("Prompt", "Blue elephant")
        with col2:
            negative_prompt = st.text_area("Negative Prompt", "")

        # Sidebar options
        scheduler = st.sidebar.selectbox("Scheduler", available_schedulers, index=0)
        image_height = st.sidebar.slider("Image height", 128, 1024, 512, 128)
        image_width = st.sidebar.slider("Image width", 128, 1024, 512, 128)
        guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 40.0, 7.5, 0.5)
        num_images = st.sidebar.slider("Number of images per prompt", 1, 30, 1, 1)
        steps = st.sidebar.slider("Steps", 1, 150, 50, 1)

        seed_placeholder = st.sidebar.empty()
        seed = seed_placeholder.number_input("Seed", value=42, min_value=1, max_value=999999, step=1)
        random_seed = st.sidebar.button("Random seed")
        _seed = torch.randint(1, 999999, (1,)).item()
        if random_seed:
            seed = seed_placeholder.number_input("Seed", value=_seed, min_value=1, max_value=999999, step=1)

        sub_col, download_col = st.columns(2)
        with sub_col:
            submit = st.button("Generate")

        if submit:
            with st.spinner("Generating images..."):
                output_images, metadata = self.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    scheduler=scheduler,
                    image_size=(image_height, image_width),
                    num_images=num_images,
                    guidance_scale=guidance_scale,
                    steps=steps,
                    seed=seed,
                )

            utils.display_and_download_images(output_images, metadata, download_col)
