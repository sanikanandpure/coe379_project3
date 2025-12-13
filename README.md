# COE Project 3
## Authors: Sanika Nandpure and Melissa Huang

## Summary
This is our final project submission for COE 379L. To improve Marathi-to-English translation quality, we employed Supervised Fine-Tuning with Low-Rank Adaptation (LoRA) on the Microsoft Phi-2 base model. We notice a significant improvement in translation ability after fine-tuning. We uploaded our LoRA adapter weights to HuggingFaceHub under the name ```spnandpure/lora-marathi```.

## Written Report
Our full written report can be found [here](https://docs.google.com/document/d/1Ks0Jph0KU8-saNxxxvcUEt2WSUn0rYKYtiEF8X43dD8/edit?usp=sharing). 

## Usage
Our [HuggingFace repo](https://huggingface.co/spnandpure/lora-marathi/tree/main) contains LoRA adapter weights trained on paired English-Marathi translation dataset.
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
