import torch
from tqdm import tqdm
from src.utils.utils import *

def train_one_epoch(model, train_dataloader, task_lang_data, 
                    optimizer, epoch, device, config, logging, wandb, writer=None, scheduler=None):
    """Train one epoch
    """
    
    model.train()
    otg_loss = 0
    losses = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    for batch_index, data in tqdm(enumerate(train_dataloader)):
        expression, world, target = data
        
        if config.task == 'ThreeAttr_RefExp_Rel_Sep_Ref':
            expression_tokens = map_expression_to_tokens(expression, separate_ref=True)
        else:
            expression_tokens = map_expression_to_tokens(expression)
        
        if config.task == 'TwoAttr_RefExp':
            world_tokens = map_two_attr_world_to_tokens(world)
        elif config.task in ['ThreeAttr_RefExp', 'ThreeAttr_RefExp_Rel', 'ThreeAttr_RefExp_Rel_Sep_Ref']:
            world_tokens = map_three_attr_world_to_tokens(world)
            
        expression_tensor = map_to_tensor(expression_tokens, task_lang_data, device)
        expression_length = expression_tensor.size()[1]
        world_tensor = map_to_tensor(world_tokens, task_lang_data, device)
            
        input_tensor = torch.cat((expression_tensor, world_tensor), dim=1)
        
        target_gridnum = target_to_gridnum(target, device)
        
        world_logits, _ = model(input_tensor, expression_length)
        
        optimizer.zero_grad()

        loss = loss_fn(world_logits, target_gridnum)
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
            
        scheduler.step()
                        
    return losses / len(train_dataloader)


def validate_one_epoch(model, val_dataloader, task_lang_data, 
                       device, config, logger):
    """Validate one epoch
    """
    
    model.eval()
    losses = 0
    accuracy = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for batch_index, data in tqdm(enumerate(val_dataloader)):
            expression, world, target = data
            
            if config.task == 'ThreeAttr_RefExp_Rel_Sep_Ref':
                expression_tokens = map_expression_to_tokens(expression, separate_ref=True)
            else:
                expression_tokens = map_expression_to_tokens(expression)

            if config.task == 'TwoAttr_RefExp':
                world_tokens = map_two_attr_world_to_tokens(world)
            elif config.task in ['ThreeAttr_RefExp', 'ThreeAttr_RefExp_Rel', 'ThreeAttr_RefExp_Rel_Sep_Ref']:
                world_tokens = map_three_attr_world_to_tokens(world)

            expression_tensor = map_to_tensor(expression_tokens, task_lang_data, device)
            expression_length = expression_tensor.size()[1]
            world_tensor = map_to_tensor(world_tokens, task_lang_data, device)

            input_tensor = torch.cat((expression_tensor, world_tensor), dim=1)
            
            target_gridnum = target_to_gridnum(target, device)
        
            world_logits, _ = model(input_tensor, expression_length)

            loss = loss_fn(world_logits, target_gridnum)
            losses += loss.item()      
                
            acc = calc_accuracy(world_logits, target_gridnum)
            accuracy += acc

    return losses / len(val_dataloader), accuracy / len(val_dataloader)


def calc_accuracy(world_logits, target_gridnum):
    correct = 0
    total = 0
    for index, sample in enumerate(world_logits):
        total += 1
        if torch.equal(torch.argmax(sample), target_gridnum[index]):
            correct += 1
            
    return correct/total