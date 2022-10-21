import torch
from tqdm import tqdm
from src.utils.utils import *

def train_one_epoch_mcnn(model, train_dataloader, command_lang, action_lang, 
                    optimizer, epoch, device, config, logging, writer=None, scheduler=None):
    """Train one epoch
    """
    
    model.train()
    otg_loss = 0
    losses = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    for batch_index, data in tqdm(enumerate(train_dataloader)):
        batch_text, batch_text_mask = tensorFromBatch(command_lang, data.input_command, 
                                                      device, max_length=config.max_position_embeddings, 
                                                      target_data=data.target_location, include_target=config.include_target)
        
        batch_world_cnn, batch_world_mask = worldFromBatch_4_conv(data.situation, config.v_feature_size, device)
        batch_world_cnn = batch_world_cnn.permute(0, 3, 1, 2)
        
        batch_target, batch_target_mask = tensorFromBatch(action_lang, data.target_sequence, 
                                                          device, max_length=config.target_max_position_embeddings)
        batch_target_input = batch_target[:, :-1]
        batch_target_mask_input = batch_target_mask[:, :-1]

        if config.num_of_streams == 'dual':
            action_pred, _, _, _, _, _, _ = model(batch_text, batch_world_cnn, 
                                                  batch_target_input, batch_text_mask, batch_world_mask, 
                                                  batch_target_mask_input, output_all_encoded_layers=False, 
                                                  output_all_attention_wts=False,)

        batch_target_output = batch_target[:, 1:]
        
        optimizer.zero_grad()

        loss = loss_fn(action_pred.reshape(-1, action_pred.shape[-1]), batch_target_output.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        
        otg_loss += loss.item()
        
        if ((batch_index+1) % config.display_freq == 0):
            display_freq = float(config.display_freq)
            avg_otg_loss = otg_loss / display_freq
            otg_loss = 0
            
            logging.info('Trained on {} batches | For last {} batches ---- Loss: {:.4f}'.format(
                batch_index+1, config.display_freq, avg_otg_loss))
            writer.add_scalar('Epoch_'+str(epoch)+'_10000_loss/train_loss', avg_otg_loss, batch_index+1)
            
#         scheduler.step()
            
    return losses / len(train_dataloader)


def validate_one_epoch_mcnn(model, val_dataloader, command_lang, 
                       action_lang, device, config, logger, exact_match=False, errors=None):
    """Validate one epoch
    """
    
    model.eval()
    losses = 0
    accuracy = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for batch_index, data in tqdm(enumerate(val_dataloader)):
            batch_text, batch_text_mask = tensorFromBatch(command_lang, data.input_command, 
                                                          device, max_length=config.max_position_embeddings, 
                                                          target_data=data.target_location, include_target=config.include_target)
            
            batch_world_cnn, batch_world_mask = worldFromBatch_4_conv(data.situation, config.v_feature_size, device)
            batch_world_cnn = batch_world_cnn.permute(0, 3, 1, 2)

            batch_target, batch_target_mask = tensorFromBatch(action_lang, data.target_sequence, 
                                                              device, max_length=config.target_max_position_embeddings)
            batch_target_no_pad = tensorFromBatch(action_lang, data.target_sequence, 
                                                  device, max_length=config.target_max_position_embeddings, no_pad=True)

            if not exact_match:
                batch_target_input = batch_target[:, :-1]
                batch_target_mask_input = batch_target_mask[:, :-1]

                if config.num_of_streams == 'dual':
                    action_pred, _, _, _, _, _, _ = model(batch_text, batch_world_cnn, 
                                                          batch_target_input, batch_text_mask, batch_world_mask, 
                                                          batch_target_mask_input, output_all_encoded_layers=False, 
                                                          output_all_attention_wts=False)
                    

                batch_target_output = batch_target[:, 1:]

                loss = loss_fn(action_pred.reshape(-1, action_pred.shape[-1]), batch_target_output.reshape(-1))
                losses += loss.item()      
                
                if config.num_of_streams == 'dual':
                    decoded_batch, _ = greedy_decode_batch_dual_mcnn(model, batch_text, batch_text_mask, 
                                                                batch_world_cnn, batch_world_mask, 
                                                                config, device, max_len=config.target_max_position_embeddings)
                
                acc, _, _ = calc_accuracy(decoded_batch, batch_target_no_pad, batch_index, data.input_command, logger, config, errors)
                accuracy += acc

                
            if exact_match:
                if config.num_of_streams == 'dual':
                    decoded_batch, _ = greedy_decode_batch_dual_mcnn(model, batch_text, batch_text_mask, 
                                                                batch_world_cnn, batch_world_mask, 
                                                                config, device, max_len=config.target_max_position_embeddings)
                    
                acc, _, errors = calc_accuracy(decoded_batch, batch_target_no_pad, batch_index, data.input_command, logger, config, errors)
                accuracy += acc
  
    if exact_match:
        return accuracy / len(val_dataloader), errors
    else:
        return losses / len(val_dataloader), accuracy / len(val_dataloader)


def greedy_decode_batch_dual_mcnn(model, batch_text, batch_text_mask, batch_world_cnn, 
                             batch_world_mask, config, 
                             device, max_len=105, start_symbol=1):
    """Greedy decode batch for testing
    """
    
    extended_attention_mask = batch_text_mask.unsqueeze(1).unsqueeze(2)
    extended_image_attention_mask = batch_world_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0


    embedding_output = model.embeddings(batch_text)
    v_embedding_output = model.mcnn_encoder(batch_world_cnn)

    encoded_layers_t, encoded_layers_v, all_attention_wts, all_attention_wts_b4_dropout = model.encoder(
        embedding_output,
        v_embedding_output,
        extended_attention_mask,
        extended_image_attention_mask,
        output_all_attention_wts=True
    )

    sequence_output_t = encoded_layers_t[-1]
    sequence_output_v = encoded_layers_v[-1]

    memory = torch.cat((sequence_output_t, sequence_output_v), 1)

    target_sentences_tokens = [[start_symbol] for _ in range(memory.shape[0])]
    trg_token_ids_batch = torch.tensor([[tokens[0]] for tokens in target_sentences_tokens], device=device)
    is_decoded = [False] * memory.shape[0]

    lengths = [config.target_max_position_embeddings] * config.batch_size

    while True:
        trg_embeddings = model.output_action_embeddings(trg_token_ids_batch)
        tgt_mask = (generate_square_subsequent_mask(trg_token_ids_batch.size(1), device).type(torch.bool)).to(device)
        out = model.decoder.transformer_decoder(trg_embeddings, memory, tgt_mask)    
        prob = model.decoder.generator(out[:, -1])
        most_probable_last_token_indices = torch.argmax(prob, dim=1).cpu().numpy()

        for idx, predicted_word in enumerate(most_probable_last_token_indices):
            if predicted_word == 2:  # once we find EOS token for a particular sentence we flag it
                is_decoded[idx] = True
                if lengths[idx] >= len(trg_token_ids_batch[0]):
                    lengths[idx] = len(trg_token_ids_batch[0])
   	 
        trg_token_ids_batch = torch.cat((trg_token_ids_batch, 
                                         torch.unsqueeze(
                                             torch.tensor(
                                             most_probable_last_token_indices, device=device), 1)), 1)
               	 
        if all(is_decoded) or len(trg_token_ids_batch[0]) == config.target_max_position_embeddings:
            break
   	 
    batch_prediction = [trg_pred[:lengths[idx]+1] for idx, trg_pred in enumerate(trg_token_ids_batch)]

    return batch_prediction, all_attention_wts_b4_dropout

def calc_accuracy(decoded_batch, batch_target_no_pad, batch_index, data, logger, config, errors):
    correct = 0
    total = 0
    accuracies = []
    for index, sample in enumerate(decoded_batch):
        total += 1
        if torch.equal(sample, batch_target_no_pad[index]):
            correct += 1
            # accuracies.append(1)
        else:
            if errors is not None:
                errors.append((batch_index*config.batch_size)+index)
            # accuracies.append(0)
                
#         else:
#             command = " ".join(data[index])
#             with open(config.outputs_path + '/output_error_examples.txt', 'a') as f_out:
#                 f_out.write("{},".format((int(batch_index)*64)+int(index)))
#                 f_out.write("Example Number: {}".format(index))
#                 f_out.write("Command: {}".format(command))
#                 f_out.write("\n")
            
    return correct/total, accuracies, errors


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask