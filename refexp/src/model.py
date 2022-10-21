import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, task_lang_data, device, config):
        super(SimpleTransformer, self).__init__()

        self.device = device
        self.task = config.task
        self.include_pos = config.include_pos
        
        self.vocab_size = len(task_lang_data['lang'])
        self.embed_dim = len(task_lang_data['embedding_matrix'][0])
        
        self.num_layers = config.num_layers
        
        # initialize word embedding
        embedding_matrix = task_lang_data['embedding_matrix']
        self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embeddings.weight = nn.parameter.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float, device=device), requires_grad=False)
        
        if self.task == 'ThreeAttr_RefExp_Rel':
            if self.include_pos:
                self.pos_embeddings = nn.Embedding(8, self.embed_dim, sparse=config.sparse)
            
        multihead_attention = nn.MultiheadAttention(self.embed_dim, config.num_heads, bias=False, batch_first=True)
    
        self.multihead_attention = nn.ModuleList(
            [copy.deepcopy(multihead_attention) for _ in range(self.num_layers)]
        )
        
        self._reset_parameters()
                
    def forward(
        self,
        input_tensor,
        expression_length
    ):
        
#         Embeddings for input command and input world state
        embedding_output = self.embeddings(input_tensor)
#         print(embedding_output.shape)
        
        if self.task == 'ThreeAttr_RefExp_Rel':
            if self.include_pos:
                seq_length = expression_length
                position_ids = torch.arange(
                    seq_length, dtype=torch.long, device=self.device
                )
                position_ids = position_ids.unsqueeze(0).expand_as(input_tensor[:, :expression_length])
                position_embeddings = self.pos_embeddings(position_ids)
                position_embeddings = F.pad(input=position_embeddings, pad=(0, 0, 0, 36), mode='constant', value=0)
#                 print(position_embeddings)
                embedding_output = embedding_output + position_embeddings
#                 print(embedding_output)

        
        residual_stream = embedding_output
    
        residual_stream_data = [embedding_output.clone().detach()]
        attn_out_data = []
        
        for idx in range(0, self.num_layers):
            attn_out, atten_out_wts = self.multihead_attention[idx](residual_stream, residual_stream, residual_stream)
            residual_stream += attn_out
            attn_out_data.append(attn_out.clone().detach())
            residual_stream_data.append(residual_stream.clone().detach())
        
        world_residual_stream = residual_stream[:,expression_length:,:]
        
        world_logits = torch.sum(world_residual_stream, dim=2)
        
        forward_data = {'input_tensor': input_tensor.clone().detach(), 
                        'residual_stream_data': residual_stream_data, 
                        'attn_out_data': attn_out_data, 
                        'world_residual_stream': world_residual_stream.clone().detach(), 
                        'world_logits': world_logits.clone().detach()}

        return world_logits, forward_data
    
    
    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad==True:
                nn.init.xavier_uniform_(p)