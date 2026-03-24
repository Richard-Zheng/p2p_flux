from PIL import Image
import torch
import numpy as np
from diffusers import FluxPipeline
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from attn_proc.vanillia import VanilliaFluxAttnProcessor
import torchvision.transforms as T

image = Image.open("grid.png")
mask = Image.open("mask.png")

pipe = FluxFillPipeline.from_pretrained("/home/frain/Documents/FLUX.1-Fill", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

num_inference_steps = 50
prompt=["you're given a 2x2 grid, find the difference of up-left to up-right and apply it to down-left to fill down-right"]
# controller = p2p_attn.L2LAttentionStore(prompt, pipe, num_inference_steps)
# controller.register_attention_control(pipe)
out_width, out_height = 1232, 1632
attn_processor = VanilliaFluxAttnProcessor(pipe, prompt, 1232, 1632)

image = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    height=out_height,
    width=out_width,
    guidance_scale=30,
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(42)
).images
image[0].save(f"out.png")
to_tensor = T.ToTensor()
save_dict = {
    "image": torch.stack([to_tensor(img) for img in image]),
    "attention_map": attn_processor.attention_store,
    "out_width": out_width,
    "out_height": out_height,
}
torch.save(save_dict, "controller_attention_store.pt")

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