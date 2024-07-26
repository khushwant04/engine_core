"""
Code written by Khushwant Sanwalot, 23-07-2027
"""

import os
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model and processor initialization
try:
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit(1)

# Prompt
prompt = "<OD>"

# Saving the bounding boxes
def save_bounding_boxes_to_excel(data, excel_filename):
    try:
        df = pd.DataFrame(data)
        df.to_excel(excel_filename, index=False)
    except Exception as e:
        print(f"Error saving to Excel: {e}")

# Function for generating bounding boxes and saving to the Excel file
def get_boxes(images):
    data = []

    for image_name in images:
        try:
            image = Image.open(f"./dataset/subset/{image_name}")
        except Exception as e:
            print(f"Error opening image {image_name}: {e}")
            continue

        try:
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

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")

    save_bounding_boxes_to_excel(data, "bounding_boxes.xlsx")

if __name__ == "__main__":
    try:
        images = os.listdir("dataset/subset")
        images = images[:20]  # For testing, limit to first 20 images
        get_boxes(images)
    except Exception as e:
        print(f"Error in main execution: {e}")
