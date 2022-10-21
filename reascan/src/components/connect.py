import math
import torch.nn as nn

from src.components.utils import *
from src.components.visual import *
from src.components.language import *

class BertConnectLayer(nn.Module):
    """Co Attention layer (Allows attention across visual and language streams)
    Follows Co Attention layer figure shown in https://arxiv.org/pdf/1908.02265.pdf
    """
    def __init__(self, config):
        super(BertConnectLayer, self).__init__()
        self.biAttention = BertBiAttention(config)
        self.biOutput = BertBiOutput(config)
        
        self.v_intermediate = BertVisIntermediate(config)
        self.v_output = BertVisOutput(config)
        
        self.l_intermediate = BertLangIntermediate(config)
        self.l_output = BertLangOutput(config)
        
    def forward(
        self,
        vis_embedding,
        vis_attention_mask,
        lang_embedding,
        lang_attention_mask,
    ):
        
        bi_output1, bi_output2, co_attention_probs, co_attention_probs_b4_dropout = self.biAttention(
            vis_embedding,
            vis_attention_mask,
            lang_embedding,
            lang_attention_mask,
        )
        #Context in Language Stream (bi_output1)
        #Context in Visual Stream (bi_output2)
        
        attention_output1, attention_output2 = self.biOutput(
            bi_output2, vis_embedding, bi_output1, lang_embedding
        )
        
        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.l_intermediate(attention_output2)
        layer_output2 = self.l_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs, co_attention_probs_b4_dropout       
        
        
class BertBiAttention(nn.Module):
    """Bi Attention in Co Attention layers
    """
    def __init__(self, config):
        super(BertBiAttention, self).__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.bi_hidden_size, config.bi_num_attention_heads)
            )
            
        self.visualization = config.visualization
        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(
            config.bi_hidden_size / config.bi_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        
        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)
        

        self.query2 = nn.Linear(config.l_hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.l_hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.l_hidden_size, self.all_head_size)
        
        self.dropout2 = nn.Dropout(config.l_attention_probs_dropout_prob)
        
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
#         Shape (batch_size, number_of_heads, seq_length, head_siz)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
    ):
        
        #For Vision Input
        #Vision Embedding (input_tensor1) Size:  torch.Size([1, 37, 1024])
        query_layer1 = self.query1(input_tensor1)
        key_layer1 = self.key1(input_tensor1)
        value_layer1 = self.value1(input_tensor1)
        #Vision Query Size:  torch.Size([1, 37, 1024])
        #Vision Key Size:  torch.Size([1, 37, 1024])
        #Vision Value Size:  torch.Size([1, 37, 1024])

        query_layer1 = self.transpose_for_scores(query_layer1)
        key_layer1 = self.transpose_for_scores(key_layer1)
        value_layer1 = self.transpose_for_scores(value_layer1)
        #Vision Query Size (After transpose):  torch.Size([1, 8, 37, 128])
        #Vision Key Size (After transpose):  torch.Size([1, 8, 37, 128])
        #Vision Value Size (After transpose):  torch.Size([1, 8, 37, 128])

        
        # for Language input:
        query_layer2 = self.query2(input_tensor2)
        key_layer2 = self.key2(input_tensor2)
        value_layer2 = self.value2(input_tensor2)

        query_layer2 = self.transpose_for_scores(query_layer2)
        key_layer2 = self.transpose_for_scores(key_layer2)
        value_layer2 = self.transpose_for_scores(value_layer2)
        
        
        # Dot product between "query2" and "key1"
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1
        #Keys from Visual Stream (key_layer1):  torch.Size([1, 8, 37, 128])
        #Keys from Visual Stream (Transposed b4 matmul):  torch.Size([1, 8, 128, 37])
        #Queries from Language Stream (query_layer2):  torch.Size([1, 8, 38, 128])
        #Attention Score in Language Stream (attention_scores1):  torch.Size([1, 8, 38, 37])
        #Attention Mask in Language Stream (attention_mask1):  torch.Size([1, 1, 1, 37])
        #Attention Score in Language Stream (After masking):  torch.Size([1, 8, 38, 37])


        #Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
        #Attention Probs in Language Stream (attention_probs1):  torch.Size([1, 8, 38, 37])
        attention_probs1_b4_dropout = attention_probs1

        attention_probs1 = self.dropout1(attention_probs1)
        #Attention Probs in Language Stream (after dropout):  torch.Size([1, 8, 38, 37])
        

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)
        #Values from Visual Stream (value_layer1):  torch.Size([1, 8, 37, 128])
        #Context in Language Stream:  torch.Size([1, 8, 38, 128])
        #Context in Language Stream (After Permute):  torch.Size([1, 38, 8, 128])
        #Context in Language Stream (What shape to use):  torch.Size([1, 38, 1024])
        #Context in Language Stream (After Choosing shape):  torch.Size([1, 38, 1024])
        
        
        
        # Dot product between "query1" and "key2"
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        attention_scores2 = attention_scores2 + attention_mask2

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)
        attention_probs2_b4_dropout = attention_probs2
        
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        attn_data_b4_dropout = {
            "attn1": attention_probs1_b4_dropout,
            "attn2": attention_probs2_b4_dropout
        }
         
        
        if self.visualization == 'True':
            attn_data = {
                "attn1": attention_probs1,
                "queries1": query_layer1,
                "keys1": key_layer1,
                "values1": value_layer1, 
                "context1": context_layer1,
                "attn2": attention_probs2,
                "queries2": query_layer2,
                "keys2": key_layer2, 
                "values2": value_layer2,
                "context2": context_layer2
            }
        else:
            attn_data = None

        return context_layer1, context_layer2, attn_data, attn_data_b4_dropout   
    
    
class BertBiOutput(nn.Module):
    """Bi Attention in Co Attention layers
    """
    def __init__(self, config):
        super(BertBiOutput, self).__init__()
        
        self.dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.LayerNorm1 = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.dense2 = nn.Linear(config.bi_hidden_size, config.l_hidden_size)
        self.LayerNorm2 = nn.LayerNorm(config.l_hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.l_hidden_dropout_prob)
        
        
    def forward(self, context_v, vis_embedding, context_l, lang_embedding):
        
        context_state1 = self.dense1(context_v)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(context_l)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + vis_embedding)
        hidden_states2 = self.LayerNorm2(context_state2 + lang_embedding)

        return hidden_states1, hidden_states2