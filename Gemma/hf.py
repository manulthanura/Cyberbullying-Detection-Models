from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_url = "manulthanura/Gemma-3-270m-Cyberbullying-Classifier"

# Load the model directly from the Hugging Face Hub
model = AutoModelForCausalLM.from_pretrained(model_url)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_url)
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

# Update the logic to correctly determine if it's cyberbullying
is_cyberbullying = 'not_cyberbullying' not in predicted_type.lower()

# Print the output in the desired format
print("\n--- Formatted Output ---")
print(f"cyberbullying: {is_cyberbullying}")
print(f"type: {predicted_type}")