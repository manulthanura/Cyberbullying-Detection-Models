# Gemma-3-270m Fine-tuned for Cyberbullying Classification

Cyberbullying is a significant issue in online communities, and detecting it effectively is crucial for creating safer digital environments. Gemma is designed to identify instances of cyberbullying in text data, helping platforms moderate content and protect users.

This model contains the fine-tuned weights of Gemma-3-270m, a model specifically trained for the task of cyberbullying detection. It leverages the capabilities of large language models to understand and classify text based on the presence of harmful or abusive language.

## Model Details

- **Developed by**: [Manul Thanura](https://manulthanura.com)
- **Model Name**: Gemma-3-270m-Cyberbullying-Classifier
- **Model Task**: Cyberbullying Detection
- **Based Model**: [Gemma-3-270m](https://huggingface.co/google/gemma-3-270m)
- **Dataset**: [Cyberbullying Classification Dataset](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)
- **GitHub Repository**: [Cyberbullying-Detection-Models](https://github.com/manulthanura/Cyberbullying-Detection-Models)
- **License**: [MIT License](https://github.com/manulthanura/Cyberbullying-Detection-Models/blob/main/LICENSE)

## Training Details

- **Base Model:** `google/gemma-3-270m`
- **Quantization:** 4-bit quantization using `BitsAndBytesConfig` (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`)
- **PEFT Method:** LoRA (`peft.LoraConfig`)
- **Training Arguments:** (`transformers.TrainingArguments`)
- **Training Environment:** Google Colab with GPU support
- **Training Duration:** Approximately 3 hours
- The formatting function used for both training and inference.
- The process for loading the fine-tuned model and tokenizer for inference.

## Usage

```python
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
```

## Limitations and Bias

This model was trained on a specific dataset and may not generalize perfectly to all types of cyberbullying or different domains of text. Like all language models, it may reflect biases present in the training data. It's important to evaluate the model's performance on your specific use case and be aware of its potential limitations and biases.