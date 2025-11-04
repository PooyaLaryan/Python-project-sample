from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# 1. Load the Processor and Model
# This downloads and caches the pre-trained weights
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. Prepare the Image (Example)
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# 3. Process the input (image + optional text prompt)
text = "a photography of" # Optional text prompt for conditional captioning
inputs = processor(raw_image, text, return_tensors="pt")

# 4. Generate the caption
out = model.generate(**inputs)

# 5. Decode the output
print(processor.decode(out[0], skip_special_tokens=True))
# Expected output: "a photography of a woman and her dog"


#----------------------------------------------------------------------------------------

# import torch
# from transformers import AutoProcessor, Blip2VisionModelWithProjection
# from transformers.image_utils import load_image

# device = "cuda" if torch.cuda.is_available() else "cpu"

# processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")
# model = Blip2VisionModelWithProjection.from_pretrained(
#     "Salesforce/blip2-itm-vit-g", dtype=torch.float16
# )
# model.to(device)
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = load_image(url)

# inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

# with torch.inference_mode():
#     outputs = model(**inputs)
# image_embeds = outputs.image_embeds
# print(image_embeds.shape)