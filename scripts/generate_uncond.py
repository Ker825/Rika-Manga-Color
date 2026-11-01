from pathlib import Path

import torch
from transformers import CLIPTextModel, CLIPTokenizer

model_id = "runwayml/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

inputs = tokenizer("", padding="do_not_pad", return_tensors="pt")
with torch.no_grad():
    # MangaNinja extrae el token 0 del embedding [1, 2, 768] -> [1, 1, 768]
    uncond_tensor = text_encoder(inputs.input_ids)[0][:, 0:1, :].to(torch.float32)

out_path = Path("models/weights/uncond_embed.pt")
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(uncond_tensor, out_path)
print(f"Guardado exitoso en {out_path} ({out_path.stat().st_size} bytes)")
