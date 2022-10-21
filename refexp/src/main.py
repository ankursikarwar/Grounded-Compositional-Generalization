import os
import sys
import wandb
import random
import logging
import argparse
import numpy as np
from attrdict import AttrDict
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter
try:
	import cPickle as pickle
except ImportError:
	import pickle

from torch.utils.data import DataLoader
    
from src.args import build_parser
from src.utils.helper import *
from src.utils.logger import *
from src.dataloader import *

from src.train import *
from src.model import *

# torch.autograd.set_detect_anomaly(True)

def main():
    
    parser = build_parser()
    args = parser.parse_args()      
    
#     with open('./src/config/'+args.load_config, 'rt') as f:
#         print('Loading config from: ./src/config/'+args.load_config)
#         t_args = argparse.Namespace()
#         t_args.__dict__.update(json.load(f))
#         args = parser.parse_args(namespace=t_args)
        
    config = args 
              
    ''' Set seed for reproducibility'''
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    '''GPU initialization'''
    device = gpu_init_pytorch(config.gpu)
    config.device = device
    
    data_path = config.data_path
    
    run_name = config.run_name
    
    config.log_path = os.path.join(config.log_folder, run_name)
    config.model_path = os.path.join(config.model_folder, run_name)
    config.board_path = os.path.join(config.board_folder, run_name)
    config.outputs_path = os.path.join(config.outputs_folder, run_name)

    config_file = os.path.join(config.model_path, 'config.p')
      
#     Create directories    
    if config.mode in ['train']:
        create_save_directories(config.log_path)
        create_save_directories(config.model_path)
        create_save_directories(config.outputs_path)
        
#     Training
    if config.mode == 'train':       
        log_file = os.path.join(config.log_path, 'log_train.txt')
        logger = get_logger(run_name, log_file, logging.DEBUG)
        writer = SummaryWriter(config.board_path)

        command = "python -m src.main" + f"{' '.join(sys.argv)}".split('py')[1]
        logger.info("The command launched : {}".format(command))
        config.command = command
        
        logger.info('Experiment Name: {}'.format(config.run_name))
        logger.debug('Device: {}'.format(device))
        
        data_fname = config.data_fname  

#         Loading train and val dataloaders
        logger.debug('Train and Val Data Path: {}'.format(os.path.join(data_path, data_fname)))
        logger.info('Loading Train Data ..')
        train_dataset = RefExp_Dataset(os.path.join(data_path, data_fname), 'Train')
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)         

        logger.info('Loading Val Data ..')
        val_dataset = RefExp_Dataset(os.path.join(data_path, data_fname), 'Val_Comp')
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
        
#         Load lang and embedding matrix
        with open('./src/config/'+config.load_config, 'r') as f:
            task_lang_data = json.load(f)
        logger.debug('Language: {}'.format(task_lang_data['lang']))
        
#         Building Model
        logger.info('Building Model ..')
        model = SimpleTransformer(task_lang_data, device, config).to(device)
            
        config.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
        out_model_archi(model, config.num_params, config)
        
        print_log(logger, vars(config))
        
#         Choosing optimizer
        parameters = filter(lambda p: p.requires_grad, model.parameters())
    
        if config.opt == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=config.lr)
        elif config.opt == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=config.lr)
        elif config.opt == 'adadelta':
            optimizer = torch.optim.Adadelta(parameters, lr=config.lr)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, steps_per_epoch=len(train_dataloader), epochs=config.epochs)
            
#         Save config file
        with open(config_file, 'wb') as f:
            pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug('Config File Saved')
        
#         Initialize wandb project
        wandb.init(project=config.wandb_project_name, 
                   entity="sikarwarank", 
                   config=config)
    
#         Initialize loss and accuracy
        min_val_loss = 10000.0
        max_val_acc = 0.0
        best_epoch = 0
        
#         Train and Validation Loop
        for epoch in range(1, config.epochs+1):
            
#             Train one epoch
            logger.info('-------------- Epoch {}/{} --------------'.format(epoch, config.epochs))
            logger.info('-------------- Train {}/{} --------------'.format(epoch, config.epochs))
            logger.info('Learning Rate: {}'.format(optimizer.param_groups[0]['lr']))
            
            epoch_train_loss = train_one_epoch(model, train_dataloader, task_lang_data,
                                               optimizer, epoch, device, config, logger, wandb, writer, scheduler)
            
            logger.info('-------------- Train Epoch Loss {} --------------'.format(epoch_train_loss))
            writer.add_scalar('epoch_loss/train_loss', epoch_train_loss, epoch)           
            wandb.log({"train_loss": epoch_train_loss, "epoch": epoch})         

            
#             Validate one epoch
            logger.info('\n')
            logger.info('-------------- Validate {}/{} --------------'.format(epoch, config.epochs))                                   
            
            epoch_val_loss, epoch_val_acc = validate_one_epoch(model, val_dataloader, task_lang_data, device, config, logger)

            logger.info('-------------- Val Epoch Loss {} --------------'.format(epoch_val_loss))
            writer.add_scalar('epoch_loss/val_loss', epoch_val_loss, epoch)
            wandb.log({"val_loss": epoch_val_loss, "epoch": epoch})

            logger.info('-------------- Val Epoch Acc {} --------------'.format(epoch_val_acc))
            writer.add_scalar('epoch_acc/val_acc', epoch_val_acc, epoch)
            wandb.log({"val_acc": epoch_val_acc, "epoch": epoch})

            
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss

#             Saving best model yet (based on max val acc)
            if epoch_val_acc > max_val_acc:
                max_val_acc = epoch_val_acc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_acc': epoch_val_acc,
                    'max_val_acc': max_val_acc,
                    'min_val_loss': min_val_loss,
                    'best_epoch': best_epoch
                }, os.path.join(config.model_path, 'model.pt'))
                
            
            if config.save_all_epochs:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch_train_loss': epoch_train_loss,
                        'epoch_val_loss': epoch_val_loss,
                        'epoch_val_acc': epoch_val_acc,
                        'max_val_acc': max_val_acc,
                        'min_val_loss': min_val_loss,
                        'best_epoch': best_epoch
                    }, os.path.join(config.model_path, 'model_epoch_'+str(epoch)+'.pt'))    
                
                
            logger.info('Best Epoch: {}'.format(best_epoch))
            logger.info('Min Val Loss: {}'.format(min_val_loss))
            logger.info('Max Val Acc: {}'.format(max_val_acc))
            logger.info('\n\n\n')
                        
        
        wandb.log({'Best_Epoch': best_epoch})
        wandb.log({'Min_Val_Loss': min_val_loss})
        wandb.log({'Max_Val_Acc': max_val_acc})
            
        writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars.json'))
        writer.close()
        
#         Test along with Train
        if config.train_test:
            
            logger.info('-------------- Testing --------------')
            logger.info('\n\nExperiment Name: {}'.format(config.run_name))
            
            logger.debug('Device: {}'.format(device))
            logger.info('\n')
            
#             Loading model
            logger.info('Loading Model From: {}'.format(os.path.join(config.model_path, 'model.pt')))
            model = SimpleTransformer(task_lang_data, device, config).to(device)
            checkpoint = torch.load(os.path.join(config.model_path, 'model.pt'), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            plt.rcParams["figure.figsize"] = (8,8)
            if config.task == 'TwoAttr_RefExp':
                plt.rcParams["font.size"] = "50"
            elif config.task in ['ThreeAttr_RefExp', 'ThreeAttr_RefExp_Rel', 'ThreeAttr_RefExp_Rel_Sep_Ref']:
                plt.rcParams["font.size"] = "20"
            
            for layer in range(0, config.num_layers):
                embed_weight = checkpoint['model_state_dict']['embeddings.weight']
                q, k, v = checkpoint['model_state_dict']['multihead_attention.'+str(layer)+'.in_proj_weight'].chunk(3, dim=0)
                q_embed = torch.matmul(embed_weight, torch.transpose(q, 0, 1))
                k_embed = torch.matmul(k, torch.transpose(embed_weight, 0, 1))
                qk_matrix = torch.matmul(q_embed, k_embed)
                
                o = checkpoint['model_state_dict']['multihead_attention.'+str(layer)+'.out_proj.weight']
                ov_matrix = torch.matmul(torch.matmul(embed_weight, torch.transpose(v, 0, 1)), torch.transpose(o, 0, 1))
                ov_matrix_sum = ov_matrix.sum(dim=1)
                qk_matrix_normalized = torch.mul(qk_matrix, ov_matrix_sum)
                
                y_labels = task_lang_data['lang'].keys()
                x_labels = task_lang_data['lang'].keys()
                fig, ax = plt.subplots(figsize=(40, 40))
                pos = ax.matshow(qk_matrix.detach().cpu(), cmap='Blues')

                for (i, j), z in np.ndenumerate(qk_matrix.detach().cpu()):
                    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

                plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
                plt.yticks(np.arange(len(x_labels)), y_labels)
                plt.savefig(os.path.join(config.log_path, 'qk_matrix_layer_'+str(layer)+'.png'))
                
                qk_image = wandb.Image(os.path.join(config.log_path, 'qk_matrix_layer_'+str(layer)+'.png'), 
                                       caption="QK Matrix Layer "+str(layer))
                wandb.log({"qk_matrix": qk_image})
                
                
                fig, ax = plt.subplots(figsize=(40, 40))
                pos = ax.matshow(qk_matrix_normalized.detach().cpu(), cmap='Blues')

                for (i, j), z in np.ndenumerate(qk_matrix_normalized.detach().cpu()):
                    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

                plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
                plt.yticks(np.arange(len(x_labels)), y_labels)
                plt.savefig(os.path.join(config.log_path, 'normalized_qk_matrix_layer_'+str(layer)+'.png'))
                
                normalized_qk_image = wandb.Image(os.path.join(config.log_path, 'normalized_qk_matrix_layer_'+str(layer)+'.png'), 
                                       caption="Normalized QK Matrix Layer "+str(layer))
                wandb.log({"normalized_qk_matrix": normalized_qk_image})
                
            

            if config.task == 'TwoAttr_RefExp':
                test_data_splits = ['Test_A1', 'Test_A2', 'Test_B1', 'Test_B2', 'Test_B3']
                wandb_label = ['A1', 'A2', 'B1', 'B2', 'B3']
            elif config.task == 'ThreeAttr_RefExp':
                test_data_splits = ['Test_A1', 'Test_A2', 'Test_A3', 'Test_A4', 'Test_B1', 'Test_B2', 'Test_B3']
                wandb_label = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3']
            elif config.task in ['ThreeAttr_RefExp_Rel', 'ThreeAttr_RefExp_Rel_Sep_Ref']:
                test_data_splits = ['Train', 'Test', 'Test_A1', 'Test_A2', 'Test_B1', 'Test_B2', 'Test_B3']
                wandb_label = ['Train', 'Test', 'A1', 'A2', 'B1', 'B2', 'B3']
                    
            for index, test_split in enumerate(test_data_splits):
#                 Load test data
                logger.info('\n')
                logger.debug('Test split: {}'.format(test_split))
                logger.info('Loading Test Data ..')
                test_dataset = RefExp_Dataset(os.path.join(data_path, data_fname), test_split)
                test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

#                 Test
                _, test_acc = validate_one_epoch(model, test_dataloader, task_lang_data, device, config, logger)

                logger.info('-------------- Test Acc {} --------------'.format(test_acc))
                wandb.log({wandb_label[index]: test_acc})
                    
        
if __name__ == '__main__':
    main()