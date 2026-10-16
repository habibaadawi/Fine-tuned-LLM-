# LLaVA-Med Chest X-Ray Fine-Tuning Pipeline

A specialized pipeline for fine-tuning the **LLaVA-1.5-7B** vision-language model on medical imaging datasets to automate chest radiography report generation[cite: 2].

---

## 📋 Project Overview
This repository contains data curation scripts and training workflows designed to adapt general-purpose multi-modal models for clinical applications. By pairing frontal chest X-rays with standardized radiologist findings and impressions, the model learns to interpret visual medical data and draft detailed clinical reports.

---

## Dataset & Curation
* **Source:** Indiana University Chest X-Ray Dataset (via Kaggle)[cite: 2].
* **Filtered Samples:** **3,301** high-quality frontal projection image-report pairs[cite: 2].
* **Data Format:** Multi-turn conversational structures mapping `<image>` tokens and expert radiologist prompts to structured `FINDINGS` and `IMPRESSION` outputs[cite: 2].

---

## Tech Stack & Requirements
* **Model:** `llava-hf/llava-1.5-7b-hf`[cite: 2]
* **Frameworks:** PyTorch[cite: 2], Transformers[cite: 2], Tokenizers[cite: 2]
* **Environment:** Python 3.12+, CUDA-enabled GPU (NVIDIA T4 / A100 recommended)

---

## Training Results & Performance

The model was successfully fine-tuned for **1 full epoch** across **826 total steps**. 

* **Epochs Completed:** 1 / 1[cite: 2]
* **Loss Convergence:**
  * **Initial Step (710):** `10.7195`[cite: 2]
  * **Mid-Training (760):** `3.8801`[cite: 2]
  * **Final Step (820):** `3.6949`[cite: 2]

### Training Progression Snippet
| Step | Training Loss |
| :--- | :--- |
| 710 | 10.7195[cite: 2] |
| 730 | 4.3478[cite: 2] |
| 750 | 3.9135[cite: 2] |
| 770 | 3.8274[cite: 2] |
| 790 | 3.7378[cite: 2] |
| 810 | 3.7138[cite: 2] |
| **820** | **3.6949**[cite: 2] |

---

## Usage & Inference

To load and run inference with the fine-tuned model weights:

```python
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load model and processor from local directory
model = LlavaForConditionalGeneration.from_pretrained(
    "./llava-local", torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("./llava-local")

# Prepare prompt and image
prompt = "USER: <image>\nYou are an expert radiologist. Analyze this chest X-ray and generate a detailed radiology report.\nASSISTANT:"
image = Image.open("path_to_xray.png").convert("RGB")

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

# Generate response
output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
response = processor.decode(output[0], skip_special_tokens=True)
print(response.split("ASSISTANT:")[-1].strip())
