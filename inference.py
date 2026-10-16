import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from optimum.quanto import quantize, qint8

# 1. Load the base model in full precision
print("🔄 Loading base model...")
base_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float32,  # Use float32 or float16 depending on your Mac's RAM
    device_map="auto"
)

# 2. Apply 8-bit quantization locally (compatible with Mac/MPS)
print("⚙️ Applying quantization...")
quantize(base_model, weights=qint8)

# 3. Load the processor
print("☕ Loading processor...")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor.tokenizer.padding_side = "right"

# 4. Attach your fine-tuned QLoRA adapter
print("🔌 Attaching fine-tuned adapter...")
model = PeftModel.from_pretrained(base_model, "./llava-finetuned")

print("✅ Model loaded successfully on your local PC!")