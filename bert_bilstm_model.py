import torch.nn as nn
from transformers import BertModel

class BertBiLSTMRegressor(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_dim=128, num_layers=1, dropout=0.3):
        super(BertBiLSTMRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        out = self.dropout(lstm_out[:, -1, :])
        return self.regressor(out).squeeze(1)
