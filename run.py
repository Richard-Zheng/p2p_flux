from PIL import Image
import torch
import numpy as np
from diffusers import FluxPipeline
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
import p2p_attn

image = Image.open("grid.png")
mask = Image.open("mask.png")

pipe = FluxFillPipeline.from_pretrained("/home/frain/Documents/FLUX.1-Fill", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

num_inference_steps = 50
prompt=["you're given a 2x2 grid, find the difference of up-left to up-right and apply it to down-left to fill down-right"]
controller = p2p_attn.L2LAttentionStore(prompt, pipe, num_inference_steps)
controller.register_attention_control(pipe)

image = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    height=1632,
    width=1232,
    guidance_scale=30,
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]
image.save(f"flux-fill-dev.png")

attn_tensor_cpu = controller.attention_store.detach().cpu()
torch.save(attn_tensor_cpu, "controller_attention_store.pt")

"""
pipe = FluxPipeline.from_pretrained("/home/frain/Documents/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompts = ["A basket of apples on a table with background of a garden",
          "A basket of oranges on a table with background of a garden"]
num_inference_steps = 20

controller = p2p_attn.AttentionReplace(
    prompts,
    pipe,
    num_inference_steps
)
controller.register_attention_control(pipe)

image = pipe(
    prompts,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images
image[0].save("flux-dev.png")
image[1].save("flux-dev-2.png")
"""