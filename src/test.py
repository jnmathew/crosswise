# file: crossword_vision_parser_full.py
import os
import base64
import json
from openai import OpenAI, APIConnectionError, APIStatusError
from dotenv import load_dotenv
from pathlib import Path

# load environment variables from .env if present
load_dotenv()

# 1. config (simple, hard-coded path to the example image)
IMAGE_PATH = "src/examples/IMG_5526.JPG"
MODEL_NAME = "gpt-5"         # stronger vision model

# 2. init client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=2, timeout=60.0)

# 3. load image and base64-encode it so we can send it inline
with open(IMAGE_PATH, "rb") as f:
    img_bytes = f.read()
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# 4. prompt: ask for grid + clues
prompt = """
You are reading a photo of a crossword puzzle. Identify the grid and output ONLY a valid JSON 2D array of 0/1 integers where:
- 0 = black square (blocked)
- 1 = white square (fillable)

Requirements:
- Output must be ONLY the JSON array (no extra text), for example: [[1,1,0],[0,1,1],...]
- Detect the full bounding grid. Use the best estimate of rows and columns.
- Orient so that the top row in the image is the first array row.
- Do not include clue text or any other fields—just the 2D array.
"""

def call_model(model_name: str) -> str:
  resp = client.responses.create(
    model=model_name,
    input=[
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": prompt},
          {
            "type": "input_image",
            # send as data URL so we don't need hosting
            "image_url": f"data:image/jpeg;base64,{img_b64}",
            "detail": "high",
          },
        ],
      }
    ],
  )
  return resp.output_text.strip()

# 5. call the API (try requested model; on connection failure, try a stable fallback)
fallback_model = "gpt-4o-mini"
try:
  output_text = call_model(MODEL_NAME)
except APIConnectionError as e:
  print(f"⚠️ Connection error with model '{MODEL_NAME}': {e}. Trying fallback '{fallback_model}'...")
  output_text = call_model(fallback_model)
except APIStatusError as e:
  # Retry once with fallback on 5xx
  if 500 <= e.status_code < 600:
    print(f"⚠️ Server error {e.status_code} for model '{MODEL_NAME}'. Trying fallback '{fallback_model}'...")
    output_text = call_model(fallback_model)
  else:
    raise

# 7. try to parse as JSON
out_dir = Path("data/temp")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{Path(IMAGE_PATH).stem}_vision.json"

try:
  data = json.loads(output_text)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
  print(json.dumps(data, indent=2))
  print(f"\nSaved JSON to: {out_path}")
except json.JSONDecodeError:
  # Save raw text so we can inspect/fix parsing issues later
  with open(out_path, "w", encoding="utf-8") as f:
    f.write(output_text)
  print("⚠️ Model did not return valid JSON. Raw output saved to:", out_path)
  print(output_text)
