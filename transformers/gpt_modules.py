"""
This is a didactic implementation of the Transformer model, adapted from the original paper
"Attention is all you need" (https://arxiv.org/abs/1706.03762).
The purpose is to show how the model works, and not to be used for training or inference.
"""
import torch
from torch import nn
import torch.nn.functional as F

# set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            max_seq_len: int = 1024,
            dropout: float = 0.
    ):
        """
        Positional encoding layer.
        Args:
            embed_dim: embedding dimensionality
            max_seq_len: maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        # initialize dropout layer
        self.dropout = nn.Dropout(dropout)

        # create the positional encoding matrix
        positional_encoding = torch.zeros(max_seq_len, embed_dim)

        # create the position tensor
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # create the div_term tensor
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))

        # apply the cosine to even indices in the positional encoding
        positional_encoding[:, 0::2] = torch.sin(position * div_term)

        # apply the sine to odd indices in the positional encoding
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # add a batch dimension
        positional_encoding = positional_encoding.unsqueeze(0)

        # register the buffer
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the positional encoding layer.
        Args:
            x: input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            output: output tensor of shape (batch_size, seq_len, embed_dim)
        """

        # add the positional encoding
        output = x + self.positional_encoding[:, :x.size(1)]

        # apply the dropout
        output = self.dropout(output)

        return output


class Attention(nn.Module):

    def __init__(self):
        """
        Scaled dot product attention layer.
        """
        super(Attention, self).__init__()

    @staticmethod
    def forward(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor | None = None
    ):
        """
        Compute the scaled dot product attention given the query, key, and value tensors.
        Usually num_heads * head_dim = embed_dim.
        Args:
            query: query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key: key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            value: value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            mask: mask tensor of shape (seq_len, seq_len)
        Returns:
            output: output tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """

        # get the dimension of the query
        head_dim = query.size(-1)

        # compute the scaled dot product. QK^T / sqrt(dim)
        # (.., seq_len, head_dim) * (.., embed_dim, head_dim) -> (.., seq_len, seq_len)
        scaled_dot_product = torch.matmul(query, key.transpose(-1, -2)) / (head_dim ** 0.5)

        # apply the mask
        if mask is not None:
            # apply the mask to the scaled dot product. mask is broadcastable
            # (.., seq_len, seq_len) + (.., seq_len, seq_len) -> (.., seq_len, seq_len)
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -float('Inf'))

        # apply the softmax. shape: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(scaled_dot_product, dim=-1)

        # apply the attention weights to the value. A * V.
        # (.., seq_len, seq_len) * (.., seq_len, head_dim) -> (.., seq_len, head_dim)
        output = torch.matmul(attention_weights, value)

        # final output shape: (batch_size, num_heads, seq_len, head_dim)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.
    ):
        """
        Multi-head attention layer.
        Args:
            embed_dim: embedding dimensionality
            num_heads: number of heads
            dropout: dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        # check if the input dimensionality is divisible by the number of heads
        assert embed_dim % num_heads == 0

        # number of features per head
        self.head_dim = embed_dim // num_heads

        # number of heads
        self.num_heads = num_heads

        # linear layers for the query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # attention layer
        self.attention = Attention()

        # linear layer for the output
        self.output_layer = nn.Linear(embed_dim, embed_dim)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor | None = None,
    ):
        """
        Forward pass of the multi-head attention layer.
        Args:
            query: query tensor of shape (batch_size, seq_len, embed_dim)
            key: key tensor of shape (batch_size, seq_len, embed_dim)
            value: value tensor of shape (batch_size, seq_len, embed_dim)
            mask: mask tensor of shape (seq_len, seq_len)
        Returns:
            output: output tensor of shape (batch_size, seq_len, embed_dim)
        """

        # get batch size and sequence length
        batch_size, seq_len, _ = query.size()

        # apply the linear layers for the query, key, and value
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # reshape the query, key, and value tensors to (batch_size, seq_len, num_heads, head_dim)
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, num_heads, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # reshape the value tensor to (batch_size, num_heads, seq_len, head_dim) for the attention computation
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # compute the scaled dot product attention, attention is applied to all heads in parallel
        # attention is computed on the last two dimensions, seq_len and head_dim, the rest are broadcast
        # 3 x (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        attention = self.attention(query, key, value, mask)

        # reshape the attention tensor to (batch_size, seq_len, num_heads, head_dim) for the concatenation
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        attention = attention.transpose(1, 2)

        # reshape the attention tensor to (batch_size, seq_len, embed_dim)
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, embed_dim)
        attention = attention.reshape(batch_size, seq_len, -1)

        # apply the dropout
        attention = self.dropout(attention)

        # apply the output layer
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        output = self.output_layer(attention)

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            dropout: float = 0.
    ):
        """
        Position-wise feed-forward layer.
        Args:
            embed_dim: embedding dimensionality
            hidden_dim: hidden dimensionality
            dropout: dropout rate
        """
        super(PositionWiseFeedForward, self).__init__()

        # linear layers
        self.linear_layer1 = nn.Linear(embed_dim, hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim, embed_dim)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the position-wise feed-forward layer.
        Args:
            x: input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            output: output tensor of shape (batch_size, seq_len, embed_dim)
        """

        # apply the first linear layer. layer only applied to the last dimension, the rest are broadcast
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, hidden_dim)
        output = self.linear_layer1(x)

        # apply the Swish activation function
        output = F.silu(output)

        # apply the dropout
        output = self.dropout(output)

        # apply the second linear layer
        # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, embed_dim)
        output = self.linear_layer2(output)

        return output


class DecoderBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            hidden_dim: int,
            dropout: float = 0.
    ):
        """
        DecoderBlock of the Transformer model.
        Args:
            embed_dim: embedding dimensionality
            num_heads: number of heads
            hidden_dim: hidden dimensionality
            dropout: dropout rate
        """
        super(DecoderBlock, self).__init__()

        # multi-head attention layer
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        # position-wise feed-forward layer
        self.position_wise_feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim, dropout)

        # layer normalization layers
        self.layer_norm_pre_attention = nn.LayerNorm(embed_dim)
        self.layer_norm_post_attention = nn.LayerNorm(embed_dim)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
    ):
        """
        Forward pass of the decoder layer.
        Args:
            x: input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: attention mask tensor of shape (seq_len, seq_len)
        Returns:
            output: output tensor of shape (batch_size, seq_len, embed_dim)

        """

        # apply the layer normalization
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        normalized = self.layer_norm_pre_attention(x)

        # apply the multi-head attention layer for self-attention
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        attention = self.multi_head_attention(normalized, normalized, normalized, attention_mask)

        # add the residual connection
        # (batch_size, seq_len, embed_dim) + (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        x = x + attention

        # apply the layer normalization
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        normalized = self.layer_norm_post_attention(x)

        # apply the position-wise feed-forward layer
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        mlp_output = self.position_wise_feed_forward(normalized)

        # add the residual connection
        # (batch_size, seq_len, embed_dim) + (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        x = x + mlp_output

        return x


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            num_layers: int,
            embed_dim: int,
            num_heads: int,
            hidden_dim: int,
            dropout: float = 0.
    ):
        """
        Transformer decoder.
        Args:
            num_layers: number of decoder layers
            embed_dim: embedding dimensionality
            num_heads: number of heads
            hidden_dim: hidden dimensionality
            dropout: dropout rate
        """
        super(TransformerDecoder, self).__init__()

        # decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
    ):
        """
        Forward pass of the transformer decoder.
        Args:
            x: input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: attention mask tensor of shape (seq_len, seq_len)
        Returns:
            output: output tensor of shape (batch_size, seq_len, embed_dim)
        """

        # apply the decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, attention_mask)

        return x


class GPT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            num_layers: int,
            embed_dim: int,
            num_heads: int,
            hidden_dim: int,
            dropout: float = 0.,
    ):
        """
        GPT model.
        Args:
            vocab_size: vocabulary size
            max_seq_len: maximum sequence length
            num_layers: number of decoder layers
            embed_dim: embedding dimensionality
            num_heads: number of heads
            hidden_dim: hidden dimensionality
            dropout: dropout rate
        """
        super(GPT, self).__init__()

        self.max_seq_len = max_seq_len

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # positional encoding layer
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=max_seq_len, dropout=dropout)

        # transformer decoder
        self.decoder = TransformerDecoder(num_layers, embed_dim, num_heads, hidden_dim, dropout)

        # linear layer for the output
        self.output_linear_layer = nn.Linear(embed_dim, vocab_size, bias=False)

        self.layer_norm_final = nn.LayerNorm(embed_dim)

    def load_pretrained_embeddings(self, embeddings: torch.Tensor):
        """
        Load the pretrained embeddings.
        Args:
            embeddings: pretrained embeddings tensor of shape (vocab_size, embed_dim)
        """
        self.embedding.weight.data.copy_(embeddings)

    def freeze_embeddings(self):
        """
        Freeze the embedding layer.
        """
        self.embedding.weight.requires_grad = False

    def unfreeze_embeddings(self):
        """
        Unfreeze the embedding layer.
        """
        self.embedding.weight.requires_grad = True

    @staticmethod
    def get_attention_mask(x: torch.Tensor):
        """
        Get the attention mask tensor.
        Args:
            x: input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            attention_mask: attention mask tensor of shape (batch_size, seq_len, seq_len)
        """

        # get a mask of size (seq_len, seq_len). 1's in the lower triangle, 0's in the upper triangle
        attention_mask = torch.tril(torch.ones(x.size(-2), x.size(-2)))

        return attention_mask

    def forward(
            self,
            x: torch.Tensor,
            targets: torch.Tensor | None = None,
    ):
        """
        Forward pass of the GPT model.
        Args:
            x: input tensor of shape (batch_size, seq_len)
            targets: target tensor of next tokens of shape (batch_size, seq_len)
        Returns:
            output: output tensor of shape (batch_size, seq_len, vocab_size)
        """

        # apply the embedding layer
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        output = self.embedding(x)

        # apply the positional encoding
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        output = self.positional_encoding(output)

        # get the attention mask
        attention_mask = GPT.get_attention_mask(output).to(output.device)

        # apply the transformer decoder
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        output = self.decoder(output, attention_mask)

        # apply the layer normalization
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        output = self.layer_norm_final(output)

        # apply the output layer
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, vocab_size)
        logits = self.output_linear_layer(output)

        loss = None
        if targets is not None:
            # compute the loss
            # (batch_size, seq_len, vocab_size) -> (batch_size, seq_len)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def generate(
            self,
            input_ids: torch.Tensor,
            max_length: int = 100,
            temperature: float = 1.,
            do_sample: bool = False,
            top_k: int | None = None,
    ):
        """
        Generate text using the GPT model.
        :param input_ids: input tensor (batch_size, seq_len) containing the token ids
        :param max_length: maximum length of the generated text
        :param temperature: temperature parameter for sampling
        :param do_sample: whether to sample or use the greedy approach
        :param top_k: top_k parameter for sampling

        """

        for _ in range(max_length):
            # get the logits
            logits, _ = self.forward(input_ids)

            # get the last token logits. shape: (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size)
            logits = logits[:, -1, :] / temperature

            # crop the logits to only the best tokens
            if top_k is not None:
                # apply the top_k sampling
                v, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < v[:, -1:], -float('Inf'))

            # get the last token probabilities. shape: (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # either sample from the distribution or take the most likely element
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            # concatenate the next token to the input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


def main():
    # test attention mask
    batch_size = 10
    seq_len = 22
    num_heads = 8
    embed_dim = 1024

    num_layers = 32
    vocab_size = 8000
    max_seq_len = 1024

    print("Device: ", DEVICE)

    print("Test attention mask")
    # create the input tensor of shape (batch_size, seq_len, embed_dim) as (1, 4, 6)
    x_mask = torch.randn((batch_size, 4, embed_dim))
    show_mask = GPT.get_attention_mask(x_mask)

    print("input shape: ", x_mask.shape)
    print("mask:")
    print(show_mask)
    print("mask shape: ", show_mask.shape)
    print("__________________________________________________________")

    # test scaled dot product attention
    print("Test scaled dot product attention")
    # create the query, key, and value tensors of shape (batch_size, num_heads, seq_len, dim)
    shape = (batch_size, num_heads, seq_len, embed_dim)
    query = torch.randn(shape)
    key = torch.randn(shape)
    value = torch.randn(shape)

    # create the mask tensor of shape (seq_len, seq_len)
    x = torch.randn((seq_len, seq_len))
    mask = GPT.get_attention_mask(x)
    output = Attention()(query, key, value, mask)

    print("mask shape: ", mask.shape)
    print("query shape: ", query.shape)
    print("key shape: ", key.shape)
    print("value shape: ", value.shape)
    print(output.shape)
    print("__________________________________________________________")

    # test multi-head attention
    print("Test multi-head attention")
    # create the input tensor of shape (batch_size, seq_len, embed_dim)
    x = torch.randn((batch_size, seq_len, embed_dim))
    mask = GPT.get_attention_mask(x)
    multi_head_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    output = multi_head_attention(x, x, x, mask)

    print(multi_head_attention)
    print("mask shape: ", mask.shape)
    print("input shape: ", x.shape)
    print("output shape: ", output.shape)
    print("__________________________________________________________")

    # test position-wise feed-forward layer
    print("Test position-wise feed-forward layer")
    # create the input tensor of shape (batch_size, seq_len, embed_dim)
    x = torch.randn((batch_size, seq_len, embed_dim))

    mlp = PositionWiseFeedForward(embed_dim=embed_dim, hidden_dim=embed_dim * 2)
    output = mlp(x)

    print(mlp)
    print("input shape: ", x.shape)
    print("output shape: ", output.shape)
    print("__________________________________________________________")

    # test decoder block
    print("Test decoder block")

    # create the input tensor of shape (batch_size, seq_len, embed_dim)
    x = torch.randn((batch_size, seq_len, embed_dim))
    mask = GPT.get_attention_mask(x)
    decoder_block = DecoderBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=embed_dim * 2)
    output = decoder_block(x, mask)

    print(decoder_block)
    print("mask shape: ", mask.shape)
    print("input shape: ", x.shape)
    print("output shape: ", output.shape)
    print("__________________________________________________________")

    # test transformer decoder
    print("Test transformer decoder")

    # create the input tensor of shape (batch_size, seq_len, embed_dim)
    x = torch.randn((batch_size, seq_len, embed_dim))
    mask = GPT.get_attention_mask(x)

    transformer_decoder = TransformerDecoder(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=embed_dim * 2,
    )

    output = transformer_decoder(x, mask)

    print(transformer_decoder)
    print("mask shape: ", mask.shape)
    print("input shape: ", x.shape)
    print("output shape: ", output.shape)
    print("__________________________________________________________")

    # test GPT
    print("Test GPT")

    # create the input tensor of shape (batch_size, seq_len)
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len)).to(DEVICE)

    print("input: ")
    print(x)
    print("input shape: ", x.shape)

    gpt = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=embed_dim * 2,
    ).to(DEVICE)

    output, _ = gpt(x)

    print(gpt)
    print("output shape: ", output.shape)
    print("__________________________________________________________")

    # test GPT generate
    print("Test GPT generate")

    # create the input tensor of shape (batch_size, seq_len)
    output = gpt.generate(x, max_length=10, do_sample=False, top_k=None)

    print("input shape: ", x.shape)
    print("output shape: ", output.shape)
    print("output: ")
    print(output)
    print("__________________________________________________________")


if __name__ == '__main__':
    main()
