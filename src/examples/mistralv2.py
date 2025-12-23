import os
import base64
from typing import List
from pydantic import BaseModel
from mistralai import Mistral, ImageURLChunk
from mistralai.extra import response_format_from_pydantic_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---- 1) Define the shape we want Mistral to return ----
class Clue(BaseModel):
    num: int
    clue: str

class CrosswordClues(BaseModel):
    ACROSS: List[Clue]
    DOWN: List[Clue]

# ---- 2) Encode your image as base64 ----
with open("./IMG_5526_masked_divided.JPG", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

# ---- 3) Create client ----
with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as mistral:
    res = mistral.ocr.process(
        model="mistral-ocr-latest",
        document=ImageURLChunk(
            image_url=f"data:image/jpeg;base64,{img_b64}"
        ),
        # this is the important part: tell it to annotate the whole doc
        document_annotation_format=response_format_from_pydantic_model(CrosswordClues),
        include_image_base64=False,
    )

    print(res)
