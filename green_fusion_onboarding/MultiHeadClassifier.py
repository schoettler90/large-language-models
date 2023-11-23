import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadClassifier(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim=32,
            heads_dims=(2, 2, 2),
            dropout=0.2
    ):
        super(MultiHeadClassifier, self).__init__()

        # MLP layer to reduce dimensionality
        self.mlp_layer1 = nn.Linear(input_dim, hidden_dim)
        self.mlp_layer2 = nn.Linear(hidden_dim, hidden_dim)

        # dropout layer
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        # unpack heads dims
        num_classes_head1, num_classes_head2, num_classes_head3 = heads_dims

        # Classification heads
        self.head1 = nn.Linear(hidden_dim, num_classes_head1)
        self.head2 = nn.Linear(hidden_dim, num_classes_head2)
        self.head3 = nn.Linear(hidden_dim, num_classes_head3)

    def forward(self, x):
        # Apply MLP layer
        x = F.relu(self.mlp_layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.mlp_layer2(x))
        x = self.dropout2(x)

        # Classification heads with softmax
        output1 = F.softmax(self.head1(x), dim=1)
        output2 = F.softmax(self.head2(x), dim=1)
        output3 = F.softmax(self.head3(x), dim=1)

        return output1, output2, output3


def main():
    # Example usage:
    # Assuming you have a sentence embedding 'input_embedding' with size (batch_size, embedding_dim)
    embedding_dim = 300  # Adjust this based on your actual embedding dimension
    num_classes_head1 = 2  # Adjust based on the actual number of classes for head 1
    num_classes_head2 = 2  # Adjust based on the actual number of classes for head 2
    num_classes_head3 = 3  # Adjust based on the actual number of classes for head 3

    model = MultiHeadClassifier(
        input_dim=embedding_dim,
        hidden_dim=256,
        heads_dims=(num_classes_head1, num_classes_head2, num_classes_head3),
        dropout=0.2,
    )

    # Example forward pass
    input_embedding = torch.randn((1, embedding_dim))  # Example batch size of 64
    output1, output2, output3 = model(input_embedding)

    # print("Output 1:")
    print(output1)
    print(output1.shape)
    # print("Output 2:")
    print(output2)
    print(output2.shape)
    # print("Output 3:")
    print(output3)
    print(output3.shape)


if __name__ == "__main__":
    main()
