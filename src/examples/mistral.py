import os
import base64
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Encode your image as base64
with open("./IMG_5526_masked_divided.JPG", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

# Create Mistral client
with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as mistral:
    # Call the OCR endpoint directly
    res = mistral.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        }
    )

    # Print full JSON response
    print(res)
