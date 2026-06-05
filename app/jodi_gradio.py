import os
import sys
import time
import argparse
import gradio as gr
from PIL import Image
from typing import Any
from pathlib import Path

import torch
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["GRADIO_TEMP_DIR"] = "./tmp"

from app.jodi_pipeline import JodiPipeline
from data.datasets.jodi_dataset import JodiDataset
from model.postprocess import (
    ImagePostProcessor, LineartPostProcessor, EdgePostProcessor, DepthPostProcessor,
    NormalPostProcessor, AlbedoPostProcessor, SegADE20KPostProcessor, OpenposePostProcessor,
)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint file.")
    return parser

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

def change_height_width_by_ar(ar):
    height, width = JodiDataset.aspect_ratio[ar]
    return int(height), int(width)

def detect_aspect_ratio_from_image(image):
    (height, width), ratio = get_closest_ratio(image.height, image.width, JodiDataset.aspect_ratio)
    return str(ratio), height, width

def tab1():
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3)
            negative_prompt = gr.Textbox(label="Negative Prompt")
            with gr.Group():
                with gr.Row():
                    num_inference_steps = gr.Slider(label="Inference Steps", minimum=2, maximum=100, value=20)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=4.5)
                with gr.Row():
                    seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, value=1234)
                    batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=20, value=1)
                with gr.Row():
                    ratio = gr.Dropdown(label="Aspect Ratio", choices=list(JodiDataset.aspect_ratio.keys()), value="1.0")
                    height = gr.Number(label="Height", interactive=False, value=1024)
                    width = gr.Number(label="Width", interactive=False, value=1024)
            generate_button = gr.Button("Generate")
        # output_gallery = gr.Gallery(label="Generated Images")
        output_gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery", columns=1, height=768)


    ratio.input(change_height_width_by_ar, inputs=ratio, outputs=[height, width])

    def generate(prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, seed, batch_size):
        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = pipe(
            images=[None] * (1 + pipe.num_conditions),
            role=[0] * (1 + pipe.num_conditions),
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            generator=generator,
        )
        results = [post_processors[i](outputs[i]) for i in range(1 + pipe.num_conditions)]
        results = torch.stack(results, dim=1).reshape(-1, 3, height, width)
        pil_images = [T.ToPILImage()(res).convert("RGB") for res in results.unbind(0)]

        timestamp = int(time.time())
        save_dir = "./generated_images/tab1"
        os.makedirs(save_dir, exist_ok=True)
        for idx, img in enumerate(pil_images):
            img.save(os.path.join(save_dir, f"tab1_seed{seed}_time{timestamp}_{idx}.png"))

        return pil_images

    generate_button.click(generate, [prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, seed, batch_size], [output_gallery])

def tab2():
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3)
            negative_prompt = gr.Textbox(label="Negative Prompt")
            with gr.Group():
                with gr.Row():
                    num_inference_steps = gr.Slider(label="Inference Steps", minimum=2, maximum=100, value=20)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=4.5)
                with gr.Row():
                    seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, value=1234)
                    batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=20, value=1)
                with gr.Row():
                    ratio = gr.Dropdown(label="Aspect Ratio", choices=list(JodiDataset.aspect_ratio.keys()), value="1.0")
                    height = gr.Number(label="Height", interactive=False, value=1024)
                    width = gr.Number(label="Width", interactive=False, value=1024)
            control_images = [gr.Image(label=label, type="pil") for label in pipe.config.conditions]
            generate_button = gr.Button("Generate")
        # output_gallery = gr.Gallery(label="Generated Images")
        output_gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery", columns=1, height=768)


    ratio.input(change_height_width_by_ar, inputs=ratio, outputs=[height, width])

    def generate(prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, seed, batch_size, *control_images):
        role = [0] + [1 if img is not None else 2 for img in control_images]
        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = pipe(
            images=[None] + list(control_images),
            role=role,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            generator=generator,
        )
        results = post_processors[0](outputs[0])
        pil_images = [T.ToPILImage()(res).convert("RGB") for res in results.unbind(0)]

        timestamp = int(time.time())
        save_dir = "./generated_images/tab2"
        os.makedirs(save_dir, exist_ok=True)
        for idx, img in enumerate(pil_images):
            img.save(os.path.join(save_dir, f"tab2_seed{seed}_time{timestamp}_{idx}.png"))

        return pil_images

    generate_button.click(generate, [prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, seed, batch_size] + control_images, [output_gallery])

def tab3():
    with gr.Row():
        with gr.Column():
            seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, value=1234)
            num_inference_steps = gr.Slider(label="Inference Steps", minimum=2, maximum=100, value=10)
            ratio = gr.Dropdown(label="Aspect Ratio", choices=list(JodiDataset.aspect_ratio.keys()), value="1.0")
            height = gr.Number(label="Height", interactive=False, value=1024)
            width = gr.Number(label="Width", interactive=False, value=1024)
            input_image = gr.Image(label="Input Image", type="pil")
            checkbox = [gr.Checkbox(label=label, value=True) for label in pipe.config.conditions]
            generate_button = gr.Button("Generate")
        # output_gallery = gr.Gallery(label="Generated Images")
        output_gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery", columns=1, height=768)


    input_image.upload(detect_aspect_ratio_from_image, inputs=input_image, outputs=[ratio, height, width])
    ratio.input(change_height_width_by_ar, inputs=ratio, outputs=[height, width])

    def generate(num_inference_steps, height, width, seed, input_image, *checkbox):
        if all(not cb for cb in checkbox):
            raise gr.Error("Select at least one checkbox.")
        role = [1] + [0 if cb else 2 for cb in checkbox]
        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = pipe(
            images=[input_image] + [None] * pipe.num_conditions,
            role=role,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,
            num_images_per_prompt=1,
            generator=generator,
        )
        results = [post_processors[i](outputs[i]) for i in range(1 + pipe.num_conditions) if role[i] == 0]
        results = torch.cat(results, dim=0).reshape(-1, 3, height, width)
        pil_images = [T.ToPILImage()(res).convert("RGB") for res in results.unbind(0)]

        timestamp = int(time.time())
        save_dir = "./generated_images/tab3"
        os.makedirs(save_dir, exist_ok=True)
        labels = [pipe.config.conditions[i - 1] for i in range(1, 1 + pipe.num_conditions) if role[i] == 0]
        for idx, (img, label) in enumerate(zip(pil_images, labels)):
            img.save(os.path.join(save_dir, f"tab3_seed{seed}_time{timestamp}_{label}.png"))

        return pil_images

    generate_button.click(generate, [num_inference_steps, height, width, seed, input_image] + checkbox, [output_gallery])

if __name__ == "__main__":
    args = get_parser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pipe = JodiPipeline(args.config)
    pipe.from_pretrained(args.model_path)

    post_processors: list[Any] = [ImagePostProcessor()]
    for condition in pipe.config.conditions:
        if condition == "lineart":
            post_processors.append(LineartPostProcessor())
        elif condition == "edge":
            post_processors.append(EdgePostProcessor())
        elif condition == "depth":
            post_processors.append(DepthPostProcessor())
        elif condition == "normal":
            post_processors.append(NormalPostProcessor())
        elif condition == "albedo":
            post_processors.append(AlbedoPostProcessor())
        elif condition == "mask":
            post_processors.append(SegADE20KPostProcessor(color_scheme="colors12", only_return_image=True))
        elif condition == "openpose":
            post_processors.append(OpenposePostProcessor())
        else:
            post_processors.append(ImagePostProcessor())

    blocks = gr.Blocks().queue()
    with blocks:
        with gr.Row():
            gr.Markdown("# Jodi")
        with gr.Tab(label="Joint Generation"):
            tab1()
        with gr.Tab(label="Controllable Generation"):
            tab2()
        with gr.Tab(label="Image Perception"):
            tab3()
    blocks.launch(share=True)