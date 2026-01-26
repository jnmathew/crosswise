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
image_path = "./IMG_5527 copy_masked_divided.JPG"
with open(image_path, "rb") as f:
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

    # ---- 4) Save results to markdown ----
    output_file = "./IMG_5527_manual_ocr_results.md"
    with open(output_file, "w") as f:
        f.write(f"# OCR Results - {os.path.basename(image_path)}\n\n")

        if hasattr(res, 'document_annotation') and res.document_annotation:
            import json

            # Parse the JSON string if it's a string
            if isinstance(res.document_annotation, str):
                clues = json.loads(res.document_annotation)
            else:
                clues = res.document_annotation

            # Handle dict or object
            if isinstance(clues, dict):
                f.write("## ACROSS\n\n")
                for clue in clues.get('ACROSS', []):
                    if isinstance(clue, dict):
                        f.write(f"{clue['num']}. {clue['clue']}\n")
                    else:
                        f.write(f"{clue.num}. {clue.clue}\n")

                f.write("\n## DOWN\n\n")
                for clue in clues.get('DOWN', []):
                    if isinstance(clue, dict):
                        f.write(f"{clue['num']}. {clue['clue']}\n")
                    else:
                        f.write(f"{clue.num}. {clue.clue}\n")
            else:
                f.write("## ACROSS\n\n")
                for clue in clues.ACROSS:
                    f.write(f"{clue.num}. {clue.clue}\n")

                f.write("\n## DOWN\n\n")
                for clue in clues.DOWN:
                    f.write(f"{clue.num}. {clue.clue}\n")
        else:
            f.write("No structured data found.\n\n")
            f.write(f"Raw response:\n```\n{res}\n```\n")

    print(f"Results saved to {output_file}")
