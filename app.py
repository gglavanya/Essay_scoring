from flask import Flask, render_template, request
from transformers import BertTokenizer
import torch
from model.bert_bilstm_model import BertBiLSTMRegressor
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load tokenizer, model, and scaler
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = torch.load('bert_bilstm_essay_scorer.pt')
model.eval()

# Ensure to use the same device as the model (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


scaler = MinMaxScaler()
scaler.fit([[0], [9]])  # Assuming IELTS scores range from 0 to 9


# Function for preprocessing and tokenizing the essay
def preprocess_essay(essay):
    # Tokenize and preprocess the essay
    inputs = tokenizer(
        essay,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    # Ensure tensors are moved to the same device as the model
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    return input_ids, attention_mask

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    essay = request.form["essay"]
    input_ids, attention_mask = preprocess_essay(essay)

    # Make prediction using the model
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        predicted_score = output.item()  # Convert tensor to float

    # Denormalize the predicted score
    predicted_score_denormalized = scaler.inverse_transform([[predicted_score]])[0][0]
    
    return f"<h2>Predicted Score: {predicted_score_denormalized:.2f}</h2> <br><br><a href='/'>Go back</a>"

if __name__ == "__main__":
    app.run(debug=True)
