import copy
import torch
import torch.nn as nn

from src.components.utils import *


class MultiModalModel_Single(nn.Module):
    def __init__(self, config):
        super(MultiModalModel_Single, self).__init__()

        # initialize command word embedding
        self.embeddings = BertEmbeddings(config)
            
        # initialize action word embedding
        self.output_action_embeddings = BertOutputEmbeddings(config)

        # initialize grid world features from cell embeddings and locations
        self.v_embeddings = BertImageEmbeddings(config)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.encoder_hidden_size, 
                                                        nhead=config.encoder_num_attention_heads, 
                                                        dim_feedforward=config.encoder_intermediate_size, 
                                                        dropout=config.encoder_hidden_dropout_prob, 
                                                        activation=config.encoder_hidden_act,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        
        self.decoder = Decoder(config)
        
#         self._reset_parameters()
        
    def forward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        target_action,
        attention_mask,
        image_attention_mask,
        target_mask,
    ):
        
#         Embeddings for input command and input world state
        embedding_output = self.embeddings(input_txt)
        v_embedding_output = self.v_embeddings(input_imgs, image_loc)
#         print(embedding_output.shape)
#         print(v_embedding_output.shape)
        
        vl_embedding_output = torch.cat((embedding_output, v_embedding_output), 1)
        
#         print(vl_embedding_output.shape)
        
#         print(attention_mask)
#         print(attention_mask.shape)
#         print(image_attention_mask)
#         print(image_attention_mask.shape)
        
        vl_mask = torch.cat((attention_mask, image_attention_mask), 1)

#         print(vl_mask)
#         print(torch.logical_not(vl_mask))
#         print(vl_mask.shape)
        
#         Embedding for target action sequence
        action_embeddings_output = self.output_action_embeddings(target_action)
        
#         Forward prop through encoder
        memory = self.encoder(
            vl_embedding_output,
            src_key_padding_mask=vl_mask
        )
            
#         Forward through decoder
        action_prediction = self.decoder(action_embeddings_output, memory, target_mask)

        return (
            action_prediction,
            embedding_output,
            v_embedding_output,
            vl_embedding_output,
            memory,
        )


class Decoder(nn.Module):
    """
    Decoder
    
    Args:
        target: Target action
        memory: Encoder Output
        target_pad_mask: Mask for target action generation

    Returns:
        out: Prob Distribution over action tokens
    """
    def __init__(self, config):
        super(Decoder, self).__init__()
        
        self.device = config.device
        
#         Decoder Layer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=config.decoder_hidden_size, 
                                                        nhead=config.decoder_num_attention_heads, 
                                                        dim_feedforward=config.decoder_intermediate_size, 
                                                        dropout=config.decoder_hidden_dropout_prob, 
                                                        activation=config.decoder_hidden_act, 
                                                        layer_norm_eps=1e-12, 
                                                        batch_first=True)
    
#         Decoder
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=config.num_decoder_layers)
        
#         Generator to map decoder output to target vocab
        self.generator = nn.Linear(config.decoder_hidden_size, config.target_vocab_size)
        
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, target, memory, target_pad_mask, target_mask=None):

        self.target_decode_mask = self.generate_square_subsequent_mask(target.size()[1])
        
        if target_mask == None:#For training
            out = self.transformer_decoder(target, memory, 
                                           tgt_mask=self.target_decode_mask, 
                                           tgt_key_padding_mask=torch.logical_not(target_pad_mask.bool()))
        else:#While testing
            out = self.transformer_decoder(target, memory, 
                                           tgt_mask=target_mask)
        
        return self.generator(out)