import torch
from tqdm import tqdm
from src.utils.utils import *

def train_one_epoch_target(model, train_dataloader, command_lang, action_lang, 
                    optimizer, epoch, device, config, logging, writer=None, scheduler=None):
    """Train one epoch
    """
    
    model.train()
    otg_loss = 0
    losses = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    for batch_index, data in tqdm(enumerate(train_dataloader)):
        batch_text, batch_text_mask = tensorFromBatch(command_lang, data.input_command, 
                                                      device, max_length=config.max_position_embeddings)
        batch_world, batch_world_mask, batch_world_loc = worldFromBatch(data.situation, 
                                                                        config.v_feature_size, device)

        batch_target_loc = locFromBatch(data.target_location, device)
        
        target_prediction = model(batch_text, batch_world, batch_world_loc, 
                                              batch_text_mask, batch_world_mask, 
                                              output_all_encoded_layers=True, 
                                              output_all_attention_wts=False,)
        
        optimizer.zero_grad()

        loss = loss_fn(target_prediction, batch_target_loc)
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
                        
    return losses / len(train_dataloader)


def validate_one_epoch_target(model, val_dataloader, command_lang, 
                       action_lang, device, config, logger, errors=None):
    """Validate one epoch
    """
    
    model.eval()
    losses = 0
    accuracy = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for batch_index, data in tqdm(enumerate(val_dataloader)):
            batch_text, batch_text_mask = tensorFromBatch(command_lang, data.input_command, 
                                                          device, max_length=config.max_position_embeddings)
            batch_world, batch_world_mask, batch_world_loc = worldFromBatch(data.situation, 
                                                                            config.v_feature_size, device)

            batch_target_loc = locFromBatch(data.target_location, device)
            
            target_prediction = model(batch_text, batch_world, batch_world_loc, 
                                              batch_text_mask, batch_world_mask, 
                                              output_all_encoded_layers=True, 
                                              output_all_attention_wts=False,)


            loss = loss_fn(target_prediction, batch_target_loc)
            losses += loss.item()      
                
            acc, errors = calc_target_accuracy(target_prediction, batch_target_loc, batch_index, config, errors)
            accuracy += acc

    return losses / len(val_dataloader), accuracy / len(val_dataloader), errors


def calc_target_accuracy(world_logits, target_gridnum, batch_index, config, errors):
    correct = 0
    total = 0
    for index, sample in enumerate(world_logits):
        total += 1
        if torch.equal(torch.argmax(sample), target_gridnum[index]):
            correct += 1
        else:
            if errors is not None:
                errors.append((batch_index*config.batch_size)+index)
            
    return correct/total, errors