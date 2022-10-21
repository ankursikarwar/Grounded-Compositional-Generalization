import copy
import torch
import torch.nn as nn
from src.components.utils import *

class Simple_Target_Predictor(nn.Module):

    def __init__(self, config):
        super(Simple_Target_Predictor, self).__init__()
                
        # initialize command word embedding
        self.embeddings = SimpleEmbeddings(config)

        # initialize grid world features from cell embeddings and locations
        self.v_embeddings = SimpleImageEmbeddings(config)
        
        self.num_layers = config.simple_num_layers
        self.num_heads = config.simple_num_heads
        self.embed_size = config.simple_embed_size
        self.seq_length = config.max_position_embeddings + 37
        
        multihead_attention = nn.MultiheadAttention(self.embed_size, self.num_heads, bias=False, batch_first=True)
    
        self.multihead_attention = nn.ModuleList(
            [copy.deepcopy(multihead_attention) for _ in range(self.num_layers)]
        )
        
        self.classifier = nn.Linear(self.embed_size*self.seq_length, 36)
#         self.fc1 =  nn.Linear(self.embed_size*self.seq_length, 1024)
#         self.act = nn.ReLU()
#         self.fc2 = nn.Linear(1024, 36)
        
        
    def forward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        attention_mask,
        image_attention_mask,
    ):

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_image_attention_mask = extended_image_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

#         Embeddings for input command and input world state
        embedding_output = self.embeddings(input_txt)
        v_embedding_output = self.v_embeddings(input_imgs, image_loc)
        
        residual_stream = torch.cat((embedding_output, v_embedding_output), 1)
        
        for idx in range(0, self.num_layers):
            attn_out, atten_out_wts = self.multihead_attention[idx](residual_stream, residual_stream, residual_stream)
            residual_stream += attn_out
            
        residual_stream = residual_stream.reshape(-1, self.embed_size*self.seq_length)
            
#         Forward through decoder
        target_prediction = self.classifier(residual_stream)
        # target_prediction = self.fc2(self.act(self.fc1(memory)))
        
        return target_prediction