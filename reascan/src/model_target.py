import torch
import torch.nn as nn

class Target_Predictor(nn.Module):

    def __init__(self, model, config, target_layer=-1, random_layer=False):
        super(Target_Predictor, self).__init__()
        
        self.target_layer = target_layer
        self.random_layer = random_layer
        
        # initialize command word embedding
        self.embeddings = model.embeddings

        # initialize grid world features from cell embeddings and locations
        self.v_embeddings = model.v_embeddings
        
        self.encoder = model.encoder
        
        self.embed_size = config.l_hidden_size
        self.seq_length = config.max_position_embeddings + 37
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
        v_embedding_output = self.v_embeddings(input_imgs, image_loc)
#         print(embedding_output.shape)
#         print(v_embedding_output.shape)
        
#         Forward prop through encoder
        encoded_layers_t, encoded_layers_v, all_attention_wts, all_attention_wts_b4_dropout = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_wts=output_all_attention_wts,
        )
        
        
        if self.target_layer in [-20, -100]:
            sequence_output_t = embedding_output
            sequence_output_v = v_embedding_output
        else:
            sequence_output_t = encoded_layers_t[self.target_layer]
            sequence_output_v = encoded_layers_v[self.target_layer]            

            
        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]
            
#         Concatenating encoder output for language and visual stream
        memory = torch.cat((sequence_output_t, sequence_output_v), 1)
        memory = memory.reshape(-1, self.embed_size*self.seq_length)
            
        if self.random_layer:
            memory = torch.rand(memory.shape).to('cuda:0')
            
#         Forward through decoder
        target_prediction = self.classifier(memory)
        # target_prediction = self.fc2(self.act(self.fc1(memory)))
        
        return target_prediction