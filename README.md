# COE Project 3
# Authors: Sanika Nandpure and Melissa Huang


##
Our \href[https://huggingface.co/spnandpure/lora-marathi/tree/main]{HuggingFace} repo contains LoRA adapter weights trained on a Marathi → English translation dataset.
To use this model, load the base model and then apply this LoRA adapter.

```
!pip install transformers peft accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "your-base-model-here"
lora_model_id = "spnandpure/lora-marathi"

# Load base model
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, lora_model_id)

# Example usage
prompt = "Translate from Marathi to English:\nMarathi: मला भूक लागली आहे.\nEnglish:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
