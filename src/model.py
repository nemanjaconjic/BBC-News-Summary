import torch
import torch.nn as nn


class BaselineModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(
            embedding_dim,
            1,
        )

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)

        mask = (input_ids != 0).unsqueeze(-1)

        masked_embeddings = embeddings * mask

        summed = masked_embeddings.sum(dim=1)

        lengths = mask.sum(dim=1).clamp(min=1)

        pooled = summed / lengths

        pooled = self.dropout(pooled)

        logits = self.output(pooled)

        return logits.squeeze(1)

def main():
    vocab_size = 10000

    model = BaselineModel(
        vocab_size=vocab_size,
        embedding_dim=100,
        dropout=0.2,
    )

    dummy_input = torch.randint(
        low=0,
        high=vocab_size,
        size=(32, 60),
    )

    output = model(dummy_input)

    print(model)

    print("\nInput shape:")
    print(dummy_input.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nNumber of trainable parameters:")

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(parameters)


if __name__ == "__main__":
    main()