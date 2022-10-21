import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_model):
        """
        Inputs
            max_len - Maximum length of a sequence to expect.
            d_model - Hidden dimensionality of the input.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return x


class BertEmbeddings(nn.Module):
    """Create embeddings from word, position embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.l_hidden_size
        )
        
        if config.pos_embed == 'learned':
            self.pos_embed = 'learned'
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.l_hidden_size
            )
        elif config.pos_embed == 'sincos':
            self.pos_embed = 'sincos'
            self.position_embeddings = PositionalEncoding(
                config.max_position_embeddings, config.l_hidden_size
            )

        self.LayerNorm = nn.LayerNorm(config.l_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.l_hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):

        words_embeddings = self.word_embeddings(input_ids)
        
        if self.pos_embed == 'learned':
            seq_length = input_ids.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
            
        elif self.pos_embed == 'sincos':
            embeddings = self.position_embeddings(words_embeddings)
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    

class BertOutputEmbeddings(nn.Module):
    """Create embeddings from word, position embeddings.
    """

    def __init__(self, config):
        super(BertOutputEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            config.target_vocab_size, config.l_hidden_size
        )
        
        if config.pos_embed == 'learned':
            self.pos_embed = 'learned'
            self.position_embeddings = nn.Embedding(
                config.target_max_position_embeddings, config.l_hidden_size
            )
        elif config.pos_embed == 'sincos':
            self.pos_embed = 'sincos'
            self.position_embeddings = PositionalEncoding(
                config.target_max_position_embeddings, config.l_hidden_size
            )

        self.LayerNorm = nn.LayerNorm(config.l_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.l_hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        
        words_embeddings = self.word_embeddings(input_ids)
        
        if self.pos_embed == 'learned':
            seq_length = input_ids.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
            
        elif self.pos_embed == 'sincos':
            embeddings = self.position_embeddings(words_embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertImageEmbeddings(nn.Module):
    """Create embeddings from image, spatial location.
    """

    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(config.v_loc_size, config.v_hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, input_ids, input_loc):

        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)
        embeddings = img_embeddings + loc_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    
    
class SimpleImageEmbeddings(nn.Module):
    """Create embeddings from image, spatial location.
    """

    def __init__(self, config):
        super(SimpleImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.simple_embed_size)
        self.image_location_embeddings = nn.Linear(config.v_loc_size, config.simple_embed_size)
        
    def forward(self, input_ids, input_loc):

        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)
        embeddings = img_embeddings + loc_embeddings
        
        return embeddings
    
    
class SimpleEmbeddings(nn.Module):
    """Create embeddings from word, position embeddings.
    """

    def __init__(self, config):
        super(SimpleEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.simple_embed_size
        )
        
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.simple_embed_size
        )

    def forward(self, input_ids, position_ids=None):

        words_embeddings = self.word_embeddings(input_ids)
        
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        return embeddings