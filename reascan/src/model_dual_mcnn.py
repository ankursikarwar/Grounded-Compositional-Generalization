import copy
import torch
import torch.nn as nn

from src.components.visual import *
from src.components.language import *
from src.components.connect import *
from src.components.utils import *


class MultiModalModel_Dual_MCNN(nn.Module):
    """
    Complete MultiModal Transformer with Encoder and Decoder 
    (For Training)
    
    Args:
        input_txt: Input command
        input_imgs: Input world state
        target_action: Target action
        attention_mask: Input command mask
        image_attention_mask: Input world state mask
        target_mask: Target action mask
            
    Returns:
        action_prediction: Output prob distribution over action tokens 
        encoded_layers_t: Last encoded layer (Language Stream)
        encoded_layers_v: Last encoded layer (Visual Stream)
        all_attention_wts: Attention weights
        all_attention_wts_b4_dropout: Attention weights (before dropout)
    """
    def __init__(self, config):
        super(MultiModalModel_Dual_MCNN, self).__init__()

        # initialize command word embedding
        self.embeddings = BertEmbeddings(config)
        
        self.mcnn_encoder = MCNN_Encoder(config)
            
        # initialize action word embedding
        self.output_action_embeddings = BertOutputEmbeddings(config)

        # initialize grid world features from cell embeddings and locations
        
        self.encoder = MultiModalEncoder_Dual_MCNN(config)
        self.decoder = Decoder_MCNN(config)
        
#         self._reset_parameters()
        
    def forward(
        self,
        input_txt,
        input_imgs,
        target_action,
        attention_mask,
        image_attention_mask,
        target_mask,
        output_all_encoded_layers=False,
        output_all_attention_wts=False,
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
        v_embedding_output = self.mcnn_encoder(input_imgs)
#         print(embedding_output.shape)
#         print(v_embedding_output.shape)
#         Embedding for target action sequence
        action_embeddings_output = self.output_action_embeddings(target_action)
        
#         Forward prop through encoder
        encoded_layers_t, encoded_layers_v, all_attention_wts, all_attention_wts_b4_dropout = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_wts=output_all_attention_wts,
        )
        
#         Select last encoder layer
        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]
            
#         Concatenating encoder output for language and visual stream
        memory = torch.cat((sequence_output_t, sequence_output_v), 1)

#         Forward through decoder
        action_prediction = self.decoder(action_embeddings_output, memory, target_mask)

        return (
            action_prediction,
            encoded_layers_t,
            encoded_layers_v,
            all_attention_wts,
            all_attention_wts_b4_dropout,
            embedding_output,
            v_embedding_output,
        )
    
#     def _reset_parameters(self):
#         r"""Initiate parameters in the transformer model."""

#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)


class MultiModalEncoder_Dual_MCNN(nn.Module):
    """
    MultiModal Encoder
    
    Args:
        lang_embedding: Input embeddings for language stream
        vis_embedding: Input embeddings for visual stream
        lang_attention_mask: Mask for language stream
        vis_attention_mask: Mask for visual stream

    Returns:
        all_attention_wts_b4_dropout: Attention weights (before dropout)
        all_encoder_layers_l: Last encoded layer (Language Stream)
        all_encoder_layers_v: Last encoded layer (Visual Stream)
        (all_attention_wts_l, all_attention_wts_v, all_attention_wts_x): Attention weights
        all_attention_wts_b4_dropout: Attention weights (before dropout)
    """
    def __init__(self, config):
        super(MultiModalEncoder_Dual_MCNN, self).__init__()
        
        self.interleave_self_attn = config.interleave_self_attn
        
#         Whether to allow co attention
        self.with_coattention = config.with_coattention
#         Which language layers use co attention
        self.t_biattention_id = config.t_biattention_id
#         Which visual layers use co attention
        self.v_biattention_id = config.v_biattention_id
        
        
#         Language Stream Layer
        l_layer = BertLangLayer(config)        
#         Visual Stream Layer
        v_layer = BertVisLayer(config)    
#         Co Attention Layer
        x_layer = BertConnectLayer(config)            
    
        
#         Make copy of all layers
        self.l_layer = nn.ModuleList(
            [copy.deepcopy(l_layer) for _ in range(config.num_lang_layers)]
        )
        
        self.v_layer = nn.ModuleList(
            [copy.deepcopy(v_layer) for _ in range(config.num_vis_layers)]
        )
        
        self.x_layer = nn.ModuleList(
            [copy.deepcopy(x_layer) for _ in range(len(config.v_biattention_id))]
        )
            
    def forward(
        self,
        lang_embedding,
        vis_embedding,
        lang_attention_mask,
        vis_attention_mask,
        output_all_encoded_layers=False,
        output_all_attention_wts=False,
    ):
        
        v_start = 0
        l_start = 0
        count = 0
        all_encoder_layers_l = []
        all_encoder_layers_v = []
        
        all_attention_wts_l = []
        all_attention_wts_v = []
        all_attention_wts_x = []
        all_attention_wts_b4_dropout = []
        
        batch_size, num_words, l_hidden_size = lang_embedding.size()
        _, num_regions, v_hidden_size = vis_embedding.size()
        
#         Begin forward through both streams
        if self.with_coattention == 'True':
            if self.interleave_self_attn == 'True':
                for v_layer_id, l_layer_id in zip(self.v_biattention_id, self.t_biattention_id):

                    v_end = v_layer_id
                    l_end = l_layer_id

                    for idx in range(l_start, l_end):
                        lang_embedding, lang_attention_probs = self.l_layer[idx](
                            lang_embedding, lang_attention_mask
                        )
#                         print("Language Self Attention: ", idx)

                        if output_all_encoded_layers:
                            all_encoder_layers_l.append(lang_embedding)

                        if output_all_attention_wts:
                            all_attention_wts_l.append(lang_attention_probs)

                    for idx in range(v_start, v_end):
                        vis_embedding, vis_attention_probs = self.v_layer[idx](
                            vis_embedding, vis_attention_mask
                        )
#                         print("Visual Self Attention: ", idx)

                        if output_all_encoded_layers:
                            all_encoder_layers_v.append(vis_embedding)

                        if output_all_attention_wts:
                            all_attention_wts_v.append(vis_attention_probs)    

                    vis_embedding, lang_embedding, co_attention_probs, co_attention_probs_b4_dropout = self.x_layer[count](
                        vis_embedding,
                        vis_attention_mask,
                        lang_embedding,
                        lang_attention_mask,
                    )
#                     print("Visual Language Cross Attention: ", count)

                    if output_all_encoded_layers:
                        all_encoder_layers_l.append(lang_embedding)
                        all_encoder_layers_v.append(vis_embedding)

                    if output_all_attention_wts:
                        all_attention_wts_x.append(co_attention_probs)
                        all_attention_wts_b4_dropout.append(co_attention_probs_b4_dropout)                

                    v_start = v_end
                    l_start = l_end
                    count += 1


                for idx in range(l_start, len(self.l_layer)):
                    lang_embedding, lang_attention_probs = self.l_layer[idx](
                        lang_embedding, lang_attention_mask
                    )
#                     print("Language Self Attention: ", idx)

                    if output_all_encoded_layers:
                        all_encoder_layers_l.append(lang_embedding)

                    if output_all_attention_wts:
                        all_attention_wts_l.append(lang_attention_probs)

                for idx in range(v_start, len(self.v_layer)):
                    vis_embedding, vis_attention_probs = self.v_layer[idx](
                        vis_embedding, vis_attention_mask
                    )
#                     print("Visual Self Attention: ", idx)

                    if output_all_encoded_layers:
                        all_encoder_layers_v.append(vis_embedding)

                    if output_all_attention_wts:
                        all_attention_wts_v.append(vis_attention_probs) 


            else:
    #             for idx in range(l_start, self.t_biattention_id[0]):
    #                 lang_embedding, lang_attention_probs = self.l_layer[idx](
    #                     lang_embedding, lang_attention_mask
    #                 )
    #                 print("Language Self Attention: ", idx)

    #                 if output_all_encoded_layers:
    #                     all_encoder_layers_l.append(lang_embedding)

    #                 if output_all_attention_wts:
    #                     all_attention_wts_l.append(lang_attention_probs)

    #             for idx in range(v_start, self.v_biattention_id[0]):
    #                 vis_embedding, vis_attention_probs = self.v_layer[idx](
    #                     vis_embedding, vis_attention_mask
    #                 )
    #                 print("Visual Self Attention: ", idx)

    #                 if output_all_encoded_layers:
    #                     all_encoder_layers_v.append(vis_embedding)

    #                 if output_all_attention_wts:
    #                     all_attention_wts_v.append(vis_attention_probs)    

                for v_layer_id, l_layer_id in zip(self.v_biattention_id, self.t_biattention_id):

                    vis_embedding, lang_embedding, co_attention_probs, co_attention_probs_b4_dropout = self.x_layer[count](
                        vis_embedding,
                        vis_attention_mask,
                        lang_embedding,
                        lang_attention_mask,
                    )
#                     print("Visual Language Cross Attention: ", count)

                    if output_all_attention_wts:
                        all_attention_wts_x.append(co_attention_probs)
                        all_attention_wts_b4_dropout.append(co_attention_probs_b4_dropout)  

                    if output_all_encoded_layers:
                        all_encoder_layers_l.append(lang_embedding)
                        all_encoder_layers_v.append(vis_embedding)

                    count += 1


                if not output_all_encoded_layers:
                    all_encoder_layers_l.append(lang_embedding)
                    all_encoder_layers_v.append(vis_embedding) 
                
        else:
            
            for idx in range(0, len(self.l_layer)):
                    lang_embedding, lang_attention_probs = self.l_layer[idx](
                        lang_embedding, lang_attention_mask
                    )
#                     print("Language Self Attention: ", idx)

                    if output_all_encoded_layers:
                        all_encoder_layers_l.append(lang_embedding)

                    if output_all_attention_wts:
                        all_attention_wts_l.append(lang_attention_probs)

            for idx in range(0, len(self.v_layer)):
                vis_embedding, vis_attention_probs = self.v_layer[idx](
                    vis_embedding, vis_attention_mask
                )
#                 print("Visual Self Attention: ", idx)

                if output_all_encoded_layers:
                    all_encoder_layers_v.append(vis_embedding)

                if output_all_attention_wts:
                    all_attention_wts_v.append(vis_attention_probs)
            
        
        if not output_all_encoded_layers:
            all_encoder_layers_l.append(lang_embedding)
            all_encoder_layers_v.append(vis_embedding)
            
#         print('Language Stream Encodings (Inclusive of Co-Attention Layers): ', len(all_encoder_layers_l))
#         print('Visual Stream Encodings (Inclusive of Co-Attention Layers): ', len(all_encoder_layers_v))
            
        return (
            all_encoder_layers_l,
            all_encoder_layers_v,
            (all_attention_wts_l, all_attention_wts_v, all_attention_wts_x),
            all_attention_wts_b4_dropout,
        )
    
    
class Decoder_MCNN(nn.Module):
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
        super(Decoder_MCNN, self).__init__()
        
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
    
    
class MCNN_Encoder(nn.Module):
    def __init__(self, config):
        super(MCNN_Encoder, self).__init__()
        
        self.device = config.device

        self.conv_1 = nn.Conv2d(config.v_feature_size, 50, 1, padding='same').to('cuda:0')
        self.conv_5 = nn.Conv2d(config.v_feature_size, 50, 5, padding='same').to('cuda:0')
        self.conv_7 = nn.Conv2d(config.v_feature_size, 50, 7, padding='same').to('cuda:0')

        self.lin = nn.Linear(150, config.v_hidden_size)
        
        self.dropout = nn.Dropout(p=0.1)
        
        
    def forward(self, batch_world):

        conv_1_out = self.conv_1(batch_world).permute(0, 2, 3, 1)
        conv_5_out = self.conv_5(batch_world).permute(0, 2, 3, 1)
        conv_7_out = self.conv_7(batch_world).permute(0, 2, 3, 1)
        
        conv_out = [conv_1_out, conv_5_out, conv_7_out]
        conv_out = torch.cat(conv_out, dim=-1)
        
        conv_out = conv_out.reshape(conv_out.shape[0], -1, conv_out.shape[-1])
        conv_out = self.lin(conv_out)
        conv_out = nn.functional.relu(conv_out)
        
        return self.dropout(conv_out)