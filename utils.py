import torch
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def score_essay(model, essay_text, scaler=None, max_len=512):
    model.eval()
    encoding = tokenizer(
        essay_text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        prediction = model(input_ids, attention_mask).cpu().numpy().flatten()[0]

    if scaler:
        prediction = scaler.inverse_transform([[prediction]])[0][0]

    return prediction
