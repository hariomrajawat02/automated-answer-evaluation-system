from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

print("Loading TrOCR model and processor...")
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

# Load your handwritten image snippet
# (Ensure 'handwritten_word.png' is in the same folder or provide the correct path)
image_path = r"E:\Projects\automated-answer-evaluation-system\backend\test-image\handwritten_word.jpg"

try:
  image = Image.open(image_path).convert("RGB")
  print(f"Successfully loaded image from: {image_path}")

  # Process the image for the model
  pixel_values = processor(images=image, return_tensors="pt").pixel_values

  # Generate digital text transcription
  print("Generating digital text...")
  generated_ids = model.generate(pixel_values)
  transcription = processor.batch_decode(
      generated_ids, skip_special_tokens=True
  )[0]

  print("\n" + "=" * 40)
  print(f"Transcription Result: {transcription}")
  print("=" * 40)

except FileNotFoundError:
  print(
      f"Error: Could not find '{image_path}'. Please place a valid image"
      " in the directory."
  )