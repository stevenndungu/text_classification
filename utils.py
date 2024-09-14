
# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random
import os, re
import itertools
from math import sqrt
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import sample
import pickle
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#print('Device: ', device)

# Define hyperparameter space
# hyperparameter_space = {
#     'lr': [1e-4, 5e-5, 1e-5],
#     'batch_size': [16, 32],
#     'num_hidden_layers': [4, 8],
#     'num_attention_heads': [4, 8],
#     'hidden_size': [128, 256],
#     'intermediate_size': [256, 512],
#     'hidden_dropout_prob': [0.1, 0.2],
#     'activation_function': ['gelu', 'relu'],
# }
hyperparameter_space = {
    'lr': [1e-3, 1e-5, 1e-8],
    'num_hidden_layers': [6,8],
    'num_attention_heads': [8,12],
    'hidden_size': [128,256],
    'intermediate_size': [256,512],
    'hidden_dropout_prob': [0.1,0.2],
    'activation_function': ['gelu'],
}
# Create a list of all combinations
all_combinations = list(itertools.product(
    hyperparameter_space['lr'],
    hyperparameter_space['num_hidden_layers'],
    hyperparameter_space['num_attention_heads'],
    hyperparameter_space['hidden_size'],
    hyperparameter_space['intermediate_size'],
    hyperparameter_space['hidden_dropout_prob'],
    hyperparameter_space['activation_function'],
))

# For random search, randomly sample a subset
num_samples = 30  # Adjust based on computational resources
sampled_combinations = sample(all_combinations, num_samples)
#print('hyperparameter set ...',sampled_combinations)



# # Configuration class for the model
class TransformerConfig:
    def __init__(self, 
                    vocab_size,
                    hidden_size,             # Hidden size (embeddings)
                    num_attention_heads,     # Number of attention heads
                    num_hidden_layers,       # Number of transformer encoder layers
                    intermediate_size,       # Intermediate size for feed-forward layers
                    hidden_dropout_prob,     # Dropout probability
                    max_position_embeddings, # Max sequence length
                    num_labels,              # Number of labels for classification (positive/negative)
                    activation_function):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.num_labels = num_labels
        self.activation_function = activation_function
      
  

#For Reproducibility
def reproducibility_requirements(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

reproducibility_requirements()

# Function to clean the review text
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers (optional, since BERT can handle this)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower().strip()
    return text
    
# Scaled Dot-Product Attention
def scaled_dot_product_attention(query, key, value):
    """
    Compute scaled dot product attention.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, dim_q).
        key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, dim_k).
        value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, dim_v).

    Returns:
        torch.Tensor: Output tensor after applying scaled dot product attention of shape (batch_size, seq_len_q, dim_v).

    """
    #first calculates the dimension of the key tensor (dim_k).
    dim_k = key.size(-1)
    # computes the attention scores by performing the dot product between the query and the transposed key tensor. The result is divided by the square root of dim_k.
    scores = torch.bmm(query, key.transpose(1, 2)) /sqrt(dim_k)
    # Next, the attention scores are normalized using the softmax function along the last dimension, which represents the sequence length (seq_len_k).
    weights = F.softmax(scores, dim=-1)
    
    """ Finally, the attention weights are applied to the value tensor by performing a batch matrix multiplication. The resulting tensor is the output of the scaled dot product attention and has the shape (batch_size, seq_len_q, dim_v).

    The output tensor represents the attended values corresponding to each query element based on their similarity to the key elements. """
    return torch.bmm(weights, value)

# Attention Head
class AttentionHead(nn.Module):

    """
    Attention head module for the Transformer model. Encapsulates the operations required to compute attention within a single attention head of the Transformer model.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        head_dim (int): Dimensionality of the attention head.

    """
    def __init__(self, embed_dim, head_dim):

        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

        
    def forward(self, hidden_state):

        """
        Perform forward pass through the attention head.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor after applying scaled dot product attention of shape (batch_size, seq_len, head_dim).

        """
        q = self.q(hidden_state)

        k = self.k(hidden_state)

        v = self.v(hidden_state)
        
        return scaled_dot_product_attention(q, k, v)

# Multi-Head Attention
class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads # Num of Multiple Heads
        
        # self.heads = nn.ModuleList(
        #     [AttentionHead(embed_dim, embed_dim)]
        # )
        self.heads = nn.ModuleList(
        [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        #self.output_linear = nn.Linear(embed_dim, embed_dim)
        self.output_linear = nn.Linear(head_dim * num_heads, embed_dim)

        
    def forward(self, hidden_state):
        
        """
        Perform forward pass through the multi-head attention module.
        
        For each attention head, the input tensor is passed through the corresponding AttentionHead instance, and the outputs are concatenated along the last dimension. The concatenated output is then passed through the output_linear layer to obtain the final output tensor.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor after applying multi-head attention and linear transformation
                of shape (batch_size, seq_len, embed_dim).

        """
        concatenated_output = torch.cat([h(hidden_state) for h in self.heads], dim=-1 )
        concatenated_output = self.output_linear(concatenated_output)

        return concatenated_output



# FeedForward Network
class FeedForward(nn.Module):
    """
    This class implements the Feed Forward neural network layer within the Transformer model.
    
    Feed Forward layer is a crucial part of the Transformer's architecture, responsible for the actual 
    transformation of the input data. It consists of two linear layers with a GELU activation function 
    in between, followed by a dropout layer for regularization.

    Parameters
    ----------
    config : object
        The configuration object containing model parameters. It should have the following attributes:
        - hidden_size: The size of the hidden layer in the transformer model.
        - intermediate_size: The size of the intermediate layer in the Feed Forward network.
        - hidden_dropout_prob: The dropout probability for the hidden layer.

    Attributes
    ----------
    linear1 : torch.nn.Module
        The first linear transformation layer.
    linear2 : torch.nn.Module
        The second linear transformation layer.
    gelu : torch.nn.Module
        The Gaussian Error Linear Unit (GELU) activation function.
    dropout : torch.nn.Module
        The dropout layer for regularization.
    """
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the Feed Forward network layer.

        Returns
        -------
        x : torch.Tensor
            The output tensor after passing through the Feed Forward network layer.
        """
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    """
    This class implements the Transformer Encoder Layer as part of the Transformer model.
    
    Each encoder layer consists of a Multi-Head Attention mechanism followed by a Position-wise 
    Feed Forward neural network. Additionally, residual connections around each of the two 
    sub-layers are employed, followed by layer normalization.

    Parameters
    ----------
    config : object
        The configuration object containing model parameters. It should have the following attributes:
        - hidden_size: The size of the hidden layer in the transformer model.

    Attributes
    ----------
    layer_norm_1 : torch.nn.Module
        The first layer normalization.
    layer_norm_2 : torch.nn.Module
        The second layer normalization.
    attention : MultiHeadAttention
        The MultiHeadAttention mechanism in the encoder layer.
    feed_forward : FeedForward
        The FeedForward neural network in the encoder layer.
    """
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the Transformer Encoder Layer.

        Returns
        -------
        x : torch.Tensor
            The output tensor after passing through the Transformer Encoder Layer.
        """
        hidden_state = self.layer_norm_1(x)
        attention_output = self.attention(hidden_state)
        x = x + attention_output  # Residual connection

        # Feed-forward
        hidden_state = self.layer_norm_2(x)
        feed_forward_output = self.feed_forward(hidden_state)
        x = x + feed_forward_output  # Residual connection
        return x





class Embeddings(nn.Module):
    """
    This class implements the Embeddings layer as part of the Transformer model.
    
    The Embeddings layer is responsible for converting input tokens and their corresponding positions 
    into dense vectors of fixed size. The token embeddings and position embeddings are summed up 
    and subsequently layer-normalized and passed through a dropout layer for regularization.

    Parameters
    ----------
    config : object
        The configuration object containing model parameters. It should have the following attributes:
        - vocab_size: The size of the vocabulary.
        - hidden_size: The size of the hidden layer in the transformer model.
        - max_position_embeddings: The maximum number of positions that the model can accept.

    Attributes
    ----------
    token_embeddings : torch.nn.Module
        The embedding layer for the tokens.
    position_embeddings : torch.nn.Module
        The embedding layer for the positions.
    layer_norm : torch.nn.Module
        The layer normalization.
    dropout : torch.nn.Module
        The dropout layer for regularization.
    """
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input tensor to the Embeddings layer, typically the token ids.

        Returns
        -------
        embeddings : torch.Tensor
            The output tensor after passing through the Embeddings layer.
        """
        # Ensure input_ids is on the same device as the embeddings
        device = input_ids.device

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)

        token_embeddings = self.token_embeddings(input_ids)  # This should be on the same device as input_ids
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings




class TransformerEncode(nn.Module):
    """
    This class implements the Transformer Encoder as part of the Transformer model.
    
    The Transformer Encoder consists of a series of identical layers, each with a self-attention mechanism 
    and a position-wise fully connected feed-forward network. The input to each layer is first processed by 
    the Embeddings layer which converts input tokens and their corresponding positions into dense vectors of 
    fixed size.

    Parameters
    ----------
    config : object
        The configuration object containing model parameters. It should have the following attributes:
        - num_hidden_layer: The number of hidden layers in the encoder.

    Attributes
    ----------
    embeddings : Embeddings
        The embedding layer which converts input tokens and positions into dense vectors.
    layers : torch.nn.ModuleList
        The list of Transformer Encoder Layers.
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        # Initialize a list of Transformer Encoder Layers. The number of layers is defined by config.num_hidden_layer
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers) ])
        
    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the Transformer Encoder.

        Returns
        -------
        x : torch.Tensor
            The output tensor after passing through the Transformer Encoder.
        """
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x

# Transformer for Sequence Classification
class TransformerForSequenceClassification(nn.Module):
    """
    This class implements the Transformer model for sequence classification tasks.
    
    The model architecture consists of a Transformer encoder, followed by a dropout layer for regularization, 
    and a linear layer for classification. The output from the [CLS] token's embedding is used for the classification task.

    Parameters
    ----------
    config : object
        The configuration object containing model parameters. It should have the following attributes:
        - hidden_size: The size of the hidden layer in the transformer model.
        - hidden_dropout_prob: The dropout probability for the hidden layer.
        - num_labels: The number of labels in the classification task.

    Attributes
    ----------
    encoder : TransformerEncode
        The Transformer Encoder.
    dropout : torch.nn.Module
        The dropout layer for regularization.
    classifier : torch.nn.Module
        The classification layer.
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncode(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, input_ids):
        encoder_output = self.encoder(input_ids)
        cls_output = encoder_output[:, 0, :]  # Use the [CLS] token representation
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits
