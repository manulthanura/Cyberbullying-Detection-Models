import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from dotenv import load_dotenv
load_dotenv()

# Load Hugging Face token from environment variable
hf_token = os.getenv("HF_AUTH_TOKEN")

# Define the paths
saved_model_path = "../Models/Gemma"
base_model_id = "google/gemma-3-270m"

# Configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_auth_token=hf_token,
)

# Load the PEFT model by adding the adapter to the base model
model = PeftModel.from_pretrained(base_model, saved_model_path)

# Merge the LoRA weights with the base model weights
model = model.merge_and_unload()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Define the formatting function
def format_prompt_inference(tweet_text):
    return f"Classify the following tweet as one of the cyberbullying types: 'not_cyberbullying', 'gender', 'religion', 'other_cyberbullying', 'age', or 'ethnicity'.\n\nTweet: {tweet_text}\n\nCyberbullying Type:"

# Example input text
input_text = "This is a test tweet about age."

# Format the input text
prompt = format_prompt_inference(input_text)

# Tokenize the input
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate a prediction
with torch.no_grad():
    outputs = model.generate(
        **input_ids,
        max_new_tokens=20,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode the output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Post-process the generated output to extract the classification
predicted_output_raw = decoded_output.replace(prompt, "").strip()
predicted_type = predicted_output_raw.split('\n')[0].strip()

is_cyberbullying = 'not_cyberbullying' not in predicted_type.lower()

# Print the output in the desired format
print("\n--- Formatted Output ---")
print(f"cyberbullying: {is_cyberbullying}")
print(f"type: {predicted_type}")