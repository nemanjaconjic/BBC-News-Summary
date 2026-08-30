import torch.nn as nn
from transformers import AutoModel

class TransformerSentenceClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2, freeze_encoder=True, unfreeze_last_n_layers=0):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden = self.transformer.config.hidden_size
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden, num_labels)

        if freeze_encoder:
            for p in self.transformer.parameters(): p.requires_grad = False

        if unfreeze_last_n_layers > 0:
            for layer in self.transformer.transformer.layer[-unfreeze_last_n_layers:]:
                for p in layer.parameters(): p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.dropout(self.activation(self.pre_classifier(x)))
        return self.classifier(x)