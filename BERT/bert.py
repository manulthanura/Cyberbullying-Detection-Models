import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

# Define the prediction function (same as in your training notebook)
def predict_class(text, model, tokenizer, device, classes, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    return classes[preds.item()]


# Define model parameters (must match the trained model)
bert_model_name = 'bert-base-uncased'
num_classes = 6
max_length = 128
model_save_path = '../Models/bert_cyberbullying_classifier.pth'

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
loaded_model = BERTClassifier(bert_model_name, num_classes).to(device)

# Load the saved model state dictionary
try:
    loaded_model.load_state_dict(torch.load(model_save_path, map_location=device))
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_save_path}")
    print("Please ensure the model file is in the correct directory.")
    exit()


classes = ['age', 'ethnicity', 'gender', 'not_cyberbullying', 'other_cyberbullying', 'religion']

# --- Example Usage ---

test_texts = [
    "This is a test tweet about something neutral.",
    "You are so stupid and ugly!",
    "I hate you because of your religion.",
    "This is a beautiful day."
]

print("\nTesting the loaded model:")
for text in test_texts:
    predicted_class = predict_class(text, loaded_model, tokenizer, device, classes)
    print(f"Text: {text}")
    print(f"Predicted class: {predicted_class}")
    print("-" * 30)