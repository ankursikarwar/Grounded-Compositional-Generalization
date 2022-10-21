import math
import torch
import torch.nn as nn

from src.components.utils import *


ACT2FN = {"gelu": torch.nn.functional.relu, 
          "relu": torch.nn.functional.relu, 
          "swish": torch.nn.functional.silu}


class BertLangLayer(nn.Module):
    def __init__(self, config):
        super(BertLangLayer, self).__init__()
        self.attention = BertLangAttention(config)
        self.intermediate = BertLangIntermediate(config)
        self.output = BertLangOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertLangAttention(nn.Module):
    def __init__(self, config):
        super(BertLangAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs
    
    
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.l_hidden_size % config.l_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.l_hidden_size, config.l_num_attention_heads)
            )
        self.num_attention_heads = config.l_num_attention_heads
        self.attention_head_size = int(config.l_hidden_size / config.l_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.visualization = config.visualization

        self.query = nn.Linear(config.l_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.l_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.l_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.l_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer, 
                "values": value_layer, 
                "context": context_layer
            }
        else:
            attn_data = None

        return context_layer, attn_data


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.l_hidden_size, config.l_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.l_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.l_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    
class BertLangIntermediate(nn.Module):
    def __init__(self, config):
        super(BertLangIntermediate, self).__init__()
        self.dense = nn.Linear(config.l_hidden_size, config.l_intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.l_hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertLangOutput(nn.Module):
    def __init__(self, config):
        super(BertLangOutput, self).__init__()
        self.dense = nn.Linear(config.l_intermediate_size, config.l_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.l_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.l_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states