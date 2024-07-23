"""
Code written by Khushwant Sanwalot, 23-07-2027
"""

import pandas as pd
import os
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 


# device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# model and processor initialization
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)


# prompt
prompt = "<OD>"


# saving the bounding boxes
def save_bounding_boxes_to_excel(data, excel_filename):
    df = pd.DataFrame(data)
    df.to_excel(excel_filename, index=False)


# fuction for generating the bounding boxes and saving to the excel file
def get_boxes(images):
    data = []

    for image_name in images:
        image = Image.open("./dataset/subset/" + image_name)

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

        bboxes = parsed_answer['<OD>']['bboxes']
        labels = parsed_answer['<OD>']['labels']

        for bbox, label in zip(bboxes, labels):
            x, y, width, height = bbox
            data.append({
                "Image Name": image_name,
                "Class Name": label,
                "X": x,
                "Y": y,
                "Width": width,
                "Height": height
            })

    save_bounding_boxes_to_excel(data, "bounding_boxes.xlsx")
if __name__ == "__main__":
    images = os.listdir("dataset/subset")
    images = images[:20] 
    get_boxes(images)       