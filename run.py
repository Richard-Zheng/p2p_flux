import torch
import numpy as np
from diffusers import FluxPipeline
import p2p_attn

pipe = FluxPipeline.from_pretrained("/home/frain/Documents/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompts = ["A cat riding a bicycle",
          "A dog riding a bicycle"]
num_inference_steps = 20

controller = p2p_attn.AttentionReplace(
    prompts,
    pipe,
    num_inference_steps
)
controller.register_attention_control(pipe)

tokens = pipe.tokenizer_2(
            prompts,
            return_length=False,
            return_tensors="pt",
        ).input_ids.squeeze(0)
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
