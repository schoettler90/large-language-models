import torch
from torch import Tensor


def average_pooling(
        last_hidden_states: Tensor,
        attention_mask: Tensor | None = None,
) -> Tensor:
    """
    Average pool the last hidden states of a transformer model. The attention_mask is  used to mask out the p
    adding tokens.

    :param last_hidden_states: last hidden states of the transformer model. Shape: (batch_size, seq_len, hidden_dim)
    :param attention_mask: attention mask of the transformer model. Shape: (batch_size, seq_len)
    :return: sentence embeddings. Shape: (batch_size, hidden_dim)

    """
    seq_dim = -2
    if attention_mask is None:
        # Average pool the last hidden states
        return last_hidden_states.mean(dim=seq_dim)
    else:
        # Mask out the padding tokens
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        # Average pool the last hidden states
        return last_hidden.sum(dim=seq_dim) / attention_mask.sum(dim=1)[..., None]


def weighted_average_pool(
        last_hidden_states: Tensor,
        attention_mask: Tensor | None = None,
) -> Tensor:
    """
    Weighted average pool the last hidden states of a transformer model. The attention_mask is  used to mask out the p
    adding tokens.

    :param last_hidden_states: last hidden states of the transformer model. Shape: (batch_size, seq_len, hidden_dim)
    :param attention_mask: attention mask of the transformer model. Shape: (batch_size, seq_len)
    :return: sentence embeddings. Shape: (batch_size, hidden_dim)

    """
    # Get weights of shape [bs, seq_len, hid_dim]
    weights = (
        torch.arange(start=1, end=last_hidden_states.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_states.size())
        .float().to(last_hidden_states.device)
    )

    # Get attn mask of shape [bs, seq_len, hid_dim]
    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(last_hidden_states.size())
        .float()
    )

    # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask

    return embeddings


def masked_mean_pooling(last_hidden_state, attention_mask):
    """
    Perform mean pooling along the sequence dimension with attention mask.

    Parameters:
    - last_hidden_state (torch.Tensor): Tensor of shape (batch_size, sequence_length, embedding_dim)
                                       representing the last hidden state of a Transformer model.
    - attention_mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) representing attention mask.

    Returns:
    - torch.Tensor: Mean-pooled representation for each sequence in the batch.
    """
    # Apply attention mask to set weights for padded tokens to zero
    masked_last_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1).expand(last_hidden_state.size())

    # Sum along the sequence dimension
    sum_pooled = torch.sum(masked_last_hidden_state, dim=1)

    # Normalize by the sum of attention mask values
    mean_pooled_representation = sum_pooled / attention_mask.sum(dim=1, keepdim=True)

    return mean_pooled_representation


def tests():
    batch_size = 1
    seq_len = 3
    hidden_dim = 4

    tensor = torch.arange(batch_size * seq_len * hidden_dim).reshape(batch_size, seq_len, hidden_dim)
    # tensor = torch.randn(batch_size, seq_len, hidden_dim)
    print(tensor)

    attention_mask = torch.tensor([[1, 1, 0]])
    print(attention_mask)

    print("Shape of tensor: ", tensor.shape)
    print("Shape of attention_mask: ", attention_mask.shape)

    # test if a average_pooling == masked_mean_pooling
    print("Average pooling: ")
    pool1 = average_pooling(tensor, attention_mask)
    pool2 = masked_mean_pooling(tensor, attention_mask)
    pool3 = weighted_average_pool(tensor, attention_mask)

    print(pool1)
    print(pool2)
    print(pool3)

    assert torch.allclose(pool1, pool2)


if __name__ == '__main__':
    tests()
