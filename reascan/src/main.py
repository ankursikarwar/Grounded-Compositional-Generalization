import os
import sys
# import wandb
import random
import logging
import argparse
import numpy as np
from attrdict import AttrDict

import torch
from tensorboardX import SummaryWriter
try:
	import cPickle as pickle
except ImportError:
	import pickle

from src.args import build_parser
from src.utils.utils import *
from src.utils.helper import *
from src.utils.logger import *
from src.dataloader import *

from src.train import *
from src.model_dual import *
from src.model_single import *

from src.train_mcnn import *
from src.model_dual_mcnn import *

from src.train_target import *
from src.model_target import *

from src.train_target_simple import *
from src.model_target_simple import *

def main():
    
    parser = build_parser()
    args = parser.parse_args()      
    
    with open('./src/config/'+args.load_config, 'rt') as f:
        print('Loading config from: ./src/config/'+args.load_config)
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
        
    config = args 
              
    ''' Set seed for reproducibility'''
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    '''GPU initialization'''
    device = gpu_init_pytorch(config.gpu)
    config.device = device
    
    data_path = config.data_path
    reascan_data_path = os.path.join(data_path, 'ReaSCAN-v1.1')
    gscan_data_path = os.path.join(data_path, 'ReaSCAN-v1.1')
    google_data_path = os.path.join(data_path, 'spatial_relation_splits')
    
    run_name = config.run_name
    
    config.log_path = os.path.join(config.log_folder, run_name)
    config.model_path = os.path.join(config.model_folder, run_name)
    config.board_path = os.path.join(config.board_folder, run_name)
    config.outputs_path = os.path.join(config.outputs_folder, run_name)

    config_file = os.path.join(config.model_path, 'config.p')
      
#     Create directories    
    if config.mode in ['train', 'train_target_simple', 'train_mcnn']:
        create_save_directories(config.log_path)
        create_save_directories(config.model_path)
        create_save_directories(config.outputs_path)
        
#     Training (Action Sequence Generation)
    if config.mode == 'train':       
        log_file = os.path.join(config.log_path, 'log_train.txt')
        logger = get_logger(run_name, log_file, logging.DEBUG)
        writer = SummaryWriter(config.board_path)

        command = "python -m src.main" + f"{' '.join(sys.argv)}".split('py')[1]
        logger.info("The command launched : {}".format(command))
        config.command = command
        
        logger.info('Experiment Name: {}'.format(config.run_name))
        logger.debug('Device: {}'.format(device))
        
        train_fname = config.train_fname
        val_fname = config.val_fname
        
#         Choose training split
        if config.dataset == 'reascan':         
            if config.train_split == 'all':
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                train_data_path = os.path.join(split_path, train_fname)
                val_data_path = os.path.join(split_path, val_fname)
                
            if config.train_split == 'custom':
                train_data_path = train_fname
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional')
                val_data_path = os.path.join(split_path, val_fname)

            if config.train_split in ['p1', 'p2', 'p3', 'p3-rd']: 
                #Change train_fname and val_fname accordingly
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + config.train_split) 
                train_data_path = os.path.join(split_path, train_fname)
                val_data_path = os.path.join(split_path, val_fname)
                
        if config.dataset == 'gscan':            
            if config.train_split == 'all':
                split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                train_data_path = os.path.join(split_path, train_fname)
                val_data_path = os.path.join(split_path, val_fname)
                
        if config.dataset == 'google':            
            if config.train_split == 'all':
                train_data_path = os.path.join(google_data_path, train_fname)
                val_data_path = os.path.join(google_data_path, val_fname)

                            
#         Loading train and val dataloaders
        logger.debug('Train Data Path: {}'.format(train_data_path))
        logger.info('Loading Train Data ..')
        train_dataloader = dataloader(train_data_path, device, batch_size=config.batch_size)
        
        logger.debug('Val Data Path: {}'.format(val_data_path))
        logger.info('Loading Val Data ..')
        val_dataloader = dataloader(val_data_path, device, batch_size=config.batch_size)
        
        # For 'include target info' experiment
        if config.include_target:
            config.vocab_size += 36
        
#         Building Model
        logger.info('Building Model ..')
        if config.num_of_streams == 'dual':
            model = MultiModalModel_Dual(config).to(device)
        elif config.num_of_streams == 'single':
            model = MultiModalModel_Single(config).to(device)
                 
        config.num_params = sum(p.numel() for p in model.parameters())
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
            
#         Setting up lr scheduler (No LR Decay by default)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_step, gamma=config.lr_decay)
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=len(train_dataloader), epochs=config.epochs)

    
#         Save generated vocab for command and action language
        command_lang, action_lang = prepareData('command', 'action', train_dataloader, include_target=config.include_target)
        torch.save(command_lang, os.path.join(config.model_path, 'command_lang.pth'))
        torch.save(action_lang, os.path.join(config.model_path, 'action_lang.pth'))
        logger.debug('Command Language: {}'.format(command_lang.word2index))
        logger.debug('Action Language: {}'.format(action_lang.word2index))
        
#         Save config file
        with open(config_file, 'wb') as f:
            pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug('Config File Saved')
        
#         Initialize wandb project
#         wandb.init(project=config.wandb_project_name, 
#                    entity="", 
#                    config=config)
    
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
            
            epoch_train_loss = train_one_epoch(model, train_dataloader, command_lang, action_lang, 
                                               optimizer, epoch, device, config, logger, writer)
            
            logger.info('-------------- Train Epoch Loss {} --------------'.format(epoch_train_loss))
            writer.add_scalar('epoch_loss/train_loss', epoch_train_loss, epoch)           
#             wandb.log({"train_loss": epoch_train_loss, "epoch": epoch})         

            
#             Validate one epoch
            logger.info('\n')
            logger.info('-------------- Validate {}/{} --------------'.format(epoch, config.epochs))                                   
            
            epoch_val_loss, epoch_val_acc = validate_one_epoch(model, val_dataloader, command_lang, 
                                                               action_lang, device, config, logger)

            logger.info('-------------- Val Epoch Loss {} --------------'.format(epoch_val_loss))
            writer.add_scalar('epoch_loss/val_loss', epoch_val_loss, epoch)
#             wandb.log({"val_loss": epoch_val_loss, "epoch": epoch})

            logger.info('-------------- Val Epoch Acc {} --------------'.format(epoch_val_acc))
            writer.add_scalar('epoch_acc/val_acc', epoch_val_acc, epoch)
#             wandb.log({"val_acc": epoch_val_acc, "epoch": epoch})

            
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
            
#             scheduler.step()
            
        
#         wandb.log({'Best_Epoch': best_epoch})
#         wandb.log({'Min_Val_Loss': min_val_loss})
#         wandb.log({'Max_Val_Acc': max_val_acc})
            
        writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars.json'))
        writer.close()
        
#         Test along with Train
        if config.train_test:
            
            # log_file = os.path.join(config.log_path, 'log_test_'+config.test_split+'.txt')
            # logger = get_logger(run_name, log_file, logging.DEBUG)
            logger.info('-------------- Testing --------------')
            logger.info('\n\nExperiment Name: {}'.format(config.run_name))
            
            logger.debug('Device: {}'.format(device))
            logger.info('\n')
            
#             Loading model
            if config.test_epoch == -1:
                logger.info('Loading Model From: {}'.format(os.path.join(config.model_path, 'model.pt')))
                if config.num_of_streams == 'dual':
                    model = MultiModalModel_Dual(config).to(device)
                elif config.num_of_streams == 'single':
                    model = MultiModalModel_Single(config).to(device)
                checkpoint = torch.load(os.path.join(config.model_path, 'model.pt'), map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.info('Loading Model From: {}'.format(os.path.join(config.model_path, 'model_epoch_'+str(config.test_epoch)+'.pt')))
                if config.num_of_streams == 'dual':
                    model = MultiModalModel_Dual(config).to(device)
                elif config.num_of_streams == 'single':
                    model = MultiModalModel_Single(config).to(device)
                checkpoint = torch.load(os.path.join(config.model_path, 'model_epoch_'+str(config.test_epoch)+'.pt'), map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])

            test_data_paths = []

            if config.embedding == 'modified':
                train_fname = 'train.json'
                val_fname = 'dev.json'
                test_fname = 'test.json'
            elif config.embedding == 'default':
                train_fname = 'train_default_embedding.json'
                val_fname = 'dev_default_embedding.json'
                test_fname = 'test_default_embedding.json'      

    #         Choosing test split    
            if config.dataset == 'reascan':
                if True: #Always test on random test
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label = ['All_Test']
            
                if config.test_split == 'all_train':
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, train_fname))
                    wandb_label += ['All_Train']

                if config.test_split == 'all_val':
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, val_fname))
                    wandb_label += ['All_Val']

                if config.test_split in ['p1', 'p2', 'p3', 'p3-rd']:
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + config.test_split + '-test') 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label += [config.test_split+'_Test']

                if config.test_split == 'comp':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2', 'c1', 'c2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2']

                if config.test_split == 'custom_comp':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2', 'c1', 'c2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))

                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c2.json'))

                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 
                                   'New_C1', 'New_C2']
                    
                if config.test_split == 'custom_comp_add_c1c2':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))
                        
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c1_'+config.data_version+'.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c2_'+config.data_version+'.json'))
                        
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c2.json'))

                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1_Rem_'+config.data_version, 'C2_Rem_'+config.data_version, 
                                   'New_C1', 'New_C2']
                    
                if config.test_split == 'custom_comp_add_c1c2_random':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))
                        
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c1_v1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c1_v2.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c1_v3.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c2_v1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c2_v2.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c2_v3.json'))
                        
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c2.json'))

                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1_Rem_v1', 'C1_Rem_v2', 'C1_Rem_v3', 'C2_Rem_v1', 'C2_Rem_v2', 'C2_Rem_v3', 
                                   'New_C1', 'New_C2']

                if config.test_split == 'custom_train':
                    train_fname = config.train_fname
                    test_data_paths.append(train_fname)

                    wandb_label += ['Custom_Train']


            if config.dataset == 'gscan':
                if True:
                    split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label = ['All_Test']
                    
                if config.test_split == 'all_train':
                    split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                    test_data_paths.append(os.path.join(split_path, train_fname))
                    wandb_label += ['All_Train']

                if config.test_split == 'all_val':
                    split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                    test_data_paths.append(os.path.join(split_path, val_fname))
                    wandb_label += ['All_Val']

                if config.test_split == 'comp':
                    for split in ['visual', 'visual_easier', 
                                  'situational_1', 'situational_2',
                                  'adverb_1', 'adverb_2', 'contextual']:
                        split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                        test_data_paths.append(os.path.join(split_path, split+'.json'))

                    wandb_label += ['Visual', 'Visual_Easier', 'Situational_1', 'Situational_2', 
                                   'Adverb_1', 'Adverb_2', 'Contextual']
                    
                    
            if config.dataset == 'google':
                if config.test_split == 'all_train':
                    test_data_paths.append(os.path.join(google_data_path, train_fname))
                    wandb_label = ['All_Train']

                if config.test_split == 'all_val':
                    test_data_paths.append(os.path.join(google_data_path, val_fname))
                    wandb_label = ['All_Val']

                if config.test_split == 'comp':
                    for split in ['train', 'dev', 'test', 'visual', 'relation', 
                                  'relative_position_1', 'relative_position_2',
                                  'referent']:
                        test_data_paths.append(os.path.join(google_data_path, split+'.json'))

                    wandb_label = ['All_Train', 'All_Val', 'All_Test', 'Visual', 'Relation', 'Relative_Position_1',
                                   'Relative_Position_2', 'Referent']
                    
                    
            for index, test_path in enumerate(test_data_paths):
#                 Load test data
                logger.info('\n')
                logger.debug('Test Data Path: {}'.format(test_path))
                logger.info('Loading Test Data ..')
                test_dataloader = dataloader(test_path, device, batch_size=config.batch_size, random_shuffle=False)                       

#                 Loading vocab
                command_lang = torch.load(os.path.join(config.model_path, 'command_lang.pth'))
                action_lang = torch.load(os.path.join(config.model_path, 'action_lang.pth'))

#                 Test
                test_acc, _ = validate_one_epoch(model, test_dataloader, command_lang, 
                                              action_lang, device, config, logger, exact_match=True)

                logger.info('-------------- Test Acc {} --------------'.format(test_acc))
#                 wandb.log({wandb_label[index]: test_acc})
        

#     Testing (Action Sequence Generation) (Don't Use Yet, Minor Fixes Remaining)
    if config.mode == 'test':
        
#         Load config for the run
        mode = config.mode
        test_epoch = config.test_epoch
        test_split = config.test_split
        run_name = config.run_name
#         wandb_run_name = config.wandb_run_name
        log_folder = config.log_folder
        model_folder = config.model_folder
        board_folder = config.board_folder
        outputs_folder = config.outputs_folder
        
        with open(config_file, 'rb') as f:
            config = AttrDict(pickle.load(f))
            assert config.run_name == run_name
            config.mode = mode
            config.test_split = test_split
            config.test_epoch = test_epoch
#             config.wandb_run_name = wandb_run_name
            
        config.log_path = os.path.join(log_folder, run_name)
        config.model_path = os.path.join(model_folder, run_name)
        config.board_path = os.path.join(board_folder, run_name)
        config.outputs_path = os.path.join(outputs_folder, run_name)
                      
        log_file = os.path.join(config.log_path, 'log_test_'+config.test_split+'.txt')
        logger = get_logger(run_name, log_file, logging.DEBUG)    
        logger.info('Experiment Name: {}'.format(config.run_name))
        
        device = gpu_init_pytorch(config.gpu)
        logger.debug('Device: {}'.format(device))
        logger.info('\n')
            
#         Loading model
        if config.test_epoch == -1:
            logger.info('Loading Model From: {}'.format(os.path.join(config.model_path, 'model.pt')))
            if config.num_of_streams == 'dual':
                model = MultiModalModel_Dual(config).to(device)
            elif config.num_of_streams == 'single':
                model = MultiModalModel_Single(config).to(device)
            checkpoint = torch.load(os.path.join(config.model_path, 'model.pt'), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.info('Loading Model From: {}'.format(os.path.join(config.model_path, 'model_epoch_'+str(config.test_epoch)+'.pt')))
            if config.num_of_streams == 'dual':
                model = MultiModalModel_Dual(config).to(device)
            elif config.num_of_streams == 'single':
                model = MultiModalModel_Single(config).to(device)
            checkpoint = torch.load(os.path.join(config.model_path, 'model_epoch_'+str(config.test_epoch)+'.pt'), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        test_data_paths = []
        
        if config.embedding == 'modified':
            train_fname = 'train.json'
            val_fname = 'dev.json'
            test_fname = 'test.json'
        elif config.embedding == 'default':
            train_fname = 'train_default_embedding.json'
            val_fname = 'dev_default_embedding.json'
            test_fname = 'test_default_embedding.json'
        
#         Choosing test split    
        if config.dataset == 'reascan':
            if True:
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                test_data_paths.append(os.path.join(split_path, test_fname))
                wandb_label = ['All_Test']
        
            if config.test_split == 'all_train':
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                test_data_paths.append(os.path.join(split_path, train_fname))
                wandb_label += ['All_Train']
                
            if config.test_split == 'all_val':
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                test_data_paths.append(os.path.join(split_path, val_fname))
                wandb_label += ['All_Val']
                
            if config.test_split in ['p1', 'p2', 'p3', 'p3-rd']:
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + config.test_split + '-test') 
                test_data_paths.append(os.path.join(split_path, test_fname))
                wandb_label += [config.test_split+'_Test']
                
            if config.test_split == 'comp':
                for split in ['a1', 'a2', 'a3', 'b1', 'b2', 'c1', 'c2']:
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2']
                    
            if config.test_split == 'custom_comp':
                for split in ['a1', 'a2', 'a3', 'b1', 'b2', 'c1', 'c2']:
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                
                test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c1.json'))
                test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c2.json'))

                wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 
                               'New_C1', 'New_C2']

            if config.test_split == 'custom_train':
                train_fname = config.train_fname
                test_data_paths.append(train_fname)

                wandb_label += ['Custom_Train']
                         
                
        if config.dataset == 'gscan':
            if True:
                split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                test_data_paths.append(os.path.join(split_path, test_fname))
                wandb_label = ['All_Test']
                
            if config.test_split == 'all_train':
                split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                test_data_paths.append(os.path.join(split_path, train_fname))
                wandb_label += ['All_Train']

            if config.test_split == 'all_val':
                split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                test_data_paths.append(os.path.join(split_path, val_fname))
                wandb_label += ['All_Val']
                
            if config.test_split == 'comp':
                for split in ['visual', 'visual_easier', 
                              'situational_1', 'situational_2',
                              'adverb_1', 'adverb_2', 'contextual']:
                    split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                    test_data_paths.append(os.path.join(split_path, split+'.json'))
                    
                wandb_label += ['Visual', 'Visual_Easier', 'Situational_1', 'Situational_2', 
                               'Adverb_1', 'Adverb_2', 'Contextual']
        
        
        if config.dataset == 'google':
            if config.test_split == 'all_train':
                test_data_paths.append(os.path.join(google_data_path, train_fname))
                wandb_label = ['All_Train']

            if config.test_split == 'all_val':
                test_data_paths.append(os.path.join(google_data_path, val_fname))
                wandb_label = ['All_Val']

            if config.test_split == 'all_test':
                test_data_paths.append(os.path.join(google_data_path, test_fname))
                wandb_label = ['All_Test']

            if config.test_split == 'comp':
                for split in ['train', 'dev', 'test', 'visual', 'relation', 
                              'relative_position_1', 'relative_position_2',
                              'referent']:
                    test_data_paths.append(os.path.join(google_data_path, split+'.json'))

                wandb_label = ['All_Train', 'All_Val', 'All_Test', 'Visual', 'Relation', 'Relative_Position_1', 
                               'Relative_Position_2', 'Referent']
        
        
#         api = wandb.Api()
#         run = api.run(config.wandb_run_name)
        
        for index, test_path in enumerate(test_data_paths):
#             Load test data
            logger.info('\n')
            logger.debug('Test Data Path: {}'.format(test_path))
            logger.info('Loading Test Data ..')
            test_dataloader = dataloader(test_path, device, batch_size=config.batch_size, random_shuffle=False)                       

#             Loading vocab
            command_lang = torch.load('./models/'+config.run_name+'/command_lang.pth')
            action_lang = torch.load('./models/'+config.run_name+'/action_lang.pth')
                        
#             Test
            test_acc, _ = validate_one_epoch(model, test_dataloader, command_lang, 
                                          action_lang, device, config, logger, exact_match=True)
            
            logger.info('-------------- Test Acc {} --------------'.format(test_acc))
#             run.summary[wandb_label[index]] = test_acc
        
#         run.summary.update()
                    
        
    # Training Target Predictor
    if config.mode == 'train_target_predictor':       
        log_file = os.path.join(config.log_path, 'log_train_target_'+str(config.target_layer)+'.txt')
        logger = get_logger(run_name, log_file, logging.DEBUG)
        writer = SummaryWriter(config.board_path)

        command = "python -m src.main" + f"{' '.join(sys.argv)}".split('py')[1]
        logger.info("The command launched : {}".format(command))
        config.command = command
        
        logger.info('Experiment Name: {}'.format(config.run_name))
        logger.debug('Device: {}'.format(device))
        
        train_fname = config.train_fname
        val_fname = config.val_fname #Which validation to use
        
#         Choose training split
        if config.dataset == 'reascan':         
            if config.train_split == 'all':
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                train_data_path = os.path.join(split_path, train_fname)
                val_data_path = os.path.join(split_path, val_fname)
        
            if config.train_split == 'custom':
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                train_data_path = train_fname
                val_data_path = os.path.join(split_path, val_fname)
                
        
        if config.dataset == 'gscan':            
            if config.train_split == 'all':
                split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                train_data_path = os.path.join(split_path, train_fname)
                val_data_path = os.path.join(split_path, val_fname)
                
        if config.dataset == 'google':            
            if config.train_split == 'all':
                train_data_path = os.path.join(google_data_path, train_fname)
                val_data_path = os.path.join(google_data_path, val_fname)
        

#         Loading train and val dataloaders
        logger.debug('Train Data Path: {}'.format(train_data_path))
        logger.info('Loading Train Data ..')
        train_dataloader = dataloader(train_data_path, device, batch_size=config.batch_size)
        
        logger.debug('Val Data Path: {}'.format(val_data_path))
        logger.info('Loading Val Data ..')
        val_dataloader = dataloader(val_data_path, device, batch_size=config.batch_size)
        
        
        with open(config_file, 'rb') as f:
            model_config = AttrDict(pickle.load(f))
        
#         Loading Model
        logger.info('Loading Model ..')
        if config.num_of_streams == 'dual':
            model = MultiModalModel_Dual(model_config).to(device)
        elif config.num_of_streams == 'single':
            model = MultiModalModel_Single(model_config).to(device)
            
        checkpoint = torch.load(os.path.join(config.model_path, 'model.pt'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
#        Building Target Predictor
        target_predictor = Target_Predictor(model, model_config, 
                                            target_layer=config.target_layer, 
                                            random_layer=config.random_layer).to(device)
        
        for name, param in target_predictor.named_parameters():
            # if name in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']:
            if name in ['classifier.weight', 'classifier.bias']:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        config.num_params = sum(p.numel() for p in target_predictor.parameters())
        out_model_archi(target_predictor, config.num_params, config, target_predictor=True)
        
        print_log(logger, vars(config))
        
#         Choosing optimizer
        parameters = filter(lambda p: p.requires_grad, target_predictor.parameters())
    
        if config.opt == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=config.lr)
        elif config.opt == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=config.lr)
        elif config.opt == 'adadelta':
            optimizer = torch.optim.Adadelta(parameters, lr=config.lr)
            
#         Setting up lr scheduler (No LR Decay by default)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_step, gamma=config.lr_decay)
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=len(train_dataloader), epochs=config.epochs)

    
#         Load command and action language
        command_lang = torch.load(os.path.join(config.model_path, 'command_lang.pth'))
        action_lang = torch.load(os.path.join(config.model_path, 'action_lang.pth'))
        
        
        config_file = os.path.join(config.model_path, 'config_target_predictor_'+str(config.target_layer)+'.p')
        
#         Save config file
        with open(config_file, 'wb') as f:
            pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug('Config File Saved')
        
#         Initialize wandb project
#         wandb.init(project=config.wandb_project_name, 
#                    entity="", 
#                    config=config)
    
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
            
            epoch_train_loss = train_one_epoch_target(target_predictor, train_dataloader, command_lang, action_lang, 
                                               optimizer, epoch, device, config, logger, writer)
            
            logger.info('-------------- Train Epoch Loss {} --------------'.format(epoch_train_loss))
            writer.add_scalar('epoch_loss/train_loss', epoch_train_loss, epoch)           
#             wandb.log({"train_loss": epoch_train_loss, "epoch": epoch})         

            
#             Validate one epoch
            logger.info('\n')
            logger.info('-------------- Validate {}/{} --------------'.format(epoch, config.epochs))                                   
            
            epoch_val_loss, epoch_val_acc, _ = validate_one_epoch_target(target_predictor, val_dataloader, command_lang, 
                                                               action_lang, device, config, logger)

            logger.info('-------------- Val Epoch Loss {} --------------'.format(epoch_val_loss))
            writer.add_scalar('epoch_loss/val_loss', epoch_val_loss, epoch)
#             wandb.log({"val_loss": epoch_val_loss, "epoch": epoch})

            logger.info('-------------- Val Epoch Acc {} --------------'.format(epoch_val_acc))
            writer.add_scalar('epoch_acc/val_acc', epoch_val_acc, epoch)
#             wandb.log({"val_acc": epoch_val_acc, "epoch": epoch})

            
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss

#             Saving best model yet (based on max val acc)
            if epoch_val_acc > max_val_acc:
                max_val_acc = epoch_val_acc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': target_predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_acc': epoch_val_acc,
                    'max_val_acc': max_val_acc,
                    'min_val_loss': min_val_loss,
                    'best_epoch': best_epoch
                }, os.path.join(config.model_path, 'target_predictor_'+str(config.target_layer)+'.pt'))
                
            
            if config.save_all_epochs:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': target_predictor.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch_train_loss': epoch_train_loss,
                        'epoch_val_loss': epoch_val_loss,
                        'epoch_val_acc': epoch_val_acc,
                        'max_val_acc': max_val_acc,
                        'min_val_loss': min_val_loss,
                        'best_epoch': best_epoch
                    }, os.path.join(config.model_path, 'target_predictor_epoch_'+str(epoch)+'_'+str(config.target_layer)+'.pt'))    
                
                
            logger.info('Best Epoch: {}'.format(best_epoch))
            logger.info('Min Val Loss: {}'.format(min_val_loss))
            logger.info('Max Val Acc: {}'.format(max_val_acc))
            logger.info('\n\n\n')
            
#             scheduler.step()
            
        
#         wandb.log({'Best_Epoch': best_epoch})
#         wandb.log({'Min_Val_Loss': min_val_loss})
#         wandb.log({'Max_Val_Acc': max_val_acc})
            
        writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars_target_predictor_'+str(config.target_layer)+'.json'))
        writer.close()
        
#         Test along with Train
        if config.train_test:
            
            # log_file = os.path.join(config.log_path, 'log_test_'+config.test_split+'.txt')
            # logger = get_logger(run_name, log_file, logging.DEBUG)
            logger.info('-------------- Testing --------------')
            logger.info('\n\nExperiment Name: {}'.format(config.run_name))
            
            logger.debug('Device: {}'.format(device))
            logger.info('\n')
            
#             Loading model
            if config.test_epoch == -1:
                logger.info('Loading Model From: {}'.format(os.path.join(config.model_path, 'target_predictor_'+str(config.target_layer)+'.pt')))
                target_predictor = Target_Predictor(model, model_config, target_layer=config.target_layer).to(device)
                checkpoint = torch.load(os.path.join(config.model_path, 'target_predictor_'+str(config.target_layer)+'.pt'), map_location=device)
                target_predictor.load_state_dict(checkpoint['model_state_dict'])

            test_data_paths = []

            train_fname = 'train.json'
            val_fname = 'dev.json'
            test_fname = 'test.json'

    #         Choosing test split    
            if config.dataset == 'reascan':
                if True:
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label = ['All_Test']
                    
                if config.test_split == 'all_train':
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, train_fname))
                    wandb_label += ['All_Train']

                if config.test_split == 'all_val':
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, val_fname))
                    wandb_label += ['All_Val']

                if config.test_split in ['p1', 'p2', 'p3', 'p3-rd']:
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + config.test_split + '-test') 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label += [config.test_split+'_Test']

                if config.test_split == 'comp':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2', 'c1', 'c2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2']

                if config.test_split == 'custom_comp':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2', 'c1', 'c2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))

                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c2.json'))

                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 
                                   'New_C1', 'New_C2']

                if config.test_split == 'custom_train':
                    train_fname = config.train_fname
                    test_data_paths.append(train_fname)

                    wandb_label += ['Custom_Train']
                    
                    
            if config.dataset == 'gscan':
                if True:
                    split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label = ['All_Test']

                if config.test_split == 'all_train':
                    split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                    test_data_paths.append(os.path.join(split_path, train_fname))
                    wandb_label += ['All_Train']

                if config.test_split == 'all_val':
                    split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                    test_data_paths.append(os.path.join(split_path, val_fname))
                    wandb_label += ['All_Val']

                if config.test_split == 'comp':
                    for split in ['visual', 'visual_easier', 
                                  'situational_1', 'situational_2',
                                  'adverb_1', 'adverb_2', 'contextual']:
                        split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                        test_data_paths.append(os.path.join(split_path, split+'.json'))

                    wandb_label += ['Visual', 'Visual_Easier', 'Situational_1', 'Situational_2', 
                                   'Adverb_1', 'Adverb_2', 'Contextual']
        
        
            if config.dataset == 'google':
                if config.test_split == 'all_train':
                    test_data_paths.append(os.path.join(google_data_path, train_fname))
                    wandb_label = ['All_Train']

                if config.test_split == 'all_val':
                    test_data_paths.append(os.path.join(google_data_path, val_fname))
                    wandb_label = ['All_Val']

                if config.test_split == 'all_test':
                    test_data_paths.append(os.path.join(google_data_path, test_fname))
                    wandb_label = ['All_Test']

                if config.test_split == 'comp':
                    for split in ['train', 'dev', 'test', 'visual', 'relation', 
                                  'relative_position_1', 'relative_position_2',
                                  'referent']:
                        test_data_paths.append(os.path.join(google_data_path, split+'.json'))

                    wandb_label = ['All_Train', 'All_Val', 'All_Test', 'Visual', 'Relation', 'Relative_Position_1', 
                                   'Relative_Position_2', 'Referent']
                    
                    
            for index, test_path in enumerate(test_data_paths):
#                 Load test data
                logger.info('\n')
                logger.debug('Test Data Path: {}'.format(test_path))
                logger.info('Loading Test Data ..')
                test_dataloader = dataloader(test_path, device, batch_size=config.batch_size, random_shuffle=False)                       

#                 Loading vocab
                command_lang = torch.load(os.path.join(config.model_path, 'command_lang.pth'))
                action_lang = torch.load(os.path.join(config.model_path, 'action_lang.pth'))

#                 Test
                _, test_acc, _ = validate_one_epoch_target(target_predictor, test_dataloader, command_lang, 
                                              action_lang, device, config, logger)

                logger.info('-------------- Test Acc {} --------------'.format(test_acc))
#                 wandb.log({wandb_label[index]: test_acc})

            
    if config.mode == 'train_target_simple':       
        log_file = os.path.join(config.log_path, 'log_train_target_simple.txt')
        logger = get_logger(run_name, log_file, logging.DEBUG)
        writer = SummaryWriter(config.board_path)

        command = "python -m src.main" + f"{' '.join(sys.argv)}".split('py')[1]
        logger.info("The command launched : {}".format(command))
        config.command = command
        
        logger.info('Experiment Name: {}'.format(config.run_name))
        logger.debug('Device: {}'.format(device))
        
        train_fname = config.train_fname
        val_fname = config.val_fname
        
#         Choose training split
        if config.dataset == 'reascan':         
            if config.train_split == 'all':
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                train_data_path = os.path.join(split_path, train_fname)
                val_data_path = os.path.join(split_path, val_fname)
                
            if config.train_split == 'custom':
                train_data_path = train_fname
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional')
                val_data_path = os.path.join(split_path, val_fname)

                            
#         Loading train and val dataloaders
        logger.debug('Train Data Path: {}'.format(train_data_path))
        logger.info('Loading Train Data ..')
        train_dataloader = dataloader(train_data_path, device, batch_size=config.batch_size)
        
        logger.debug('Val Data Path: {}'.format(val_data_path))
        logger.info('Loading Val Data ..')
        val_dataloader = dataloader(val_data_path, device, batch_size=config.batch_size)
        
        
#         Building Model
        logger.info('Building Model ..')
        model = Simple_Target_Predictor(config).to(device)
                 
        config.num_params = sum(p.numel() for p in model.parameters())
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
            
#         Setting up lr scheduler (No LR Decay by default)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_step, gamma=config.lr_decay)
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=len(train_dataloader), epochs=config.epochs)
    
#         Save generated vocab for command and action language
        command_lang, _ = prepareData('command', 'action', train_dataloader)
        torch.save(command_lang, os.path.join(config.model_path, 'command_lang.pth'))
        logger.debug('Command Language: {}'.format(command_lang.word2index))
        
#         Save config file
        with open(config_file, 'wb') as f:
            pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug('Config File Saved')
        
#         Initialize wandb project
#         wandb.init(project=config.wandb_project_name, 
#                    entity="", 
#                    config=config)
    
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
            
            epoch_train_loss = train_one_epoch_target_simple(model, train_dataloader, command_lang, 
                                               optimizer, epoch, device, config, logger, writer)
            
            logger.info('-------------- Train Epoch Loss {} --------------'.format(epoch_train_loss))
            writer.add_scalar('epoch_loss/train_loss', epoch_train_loss, epoch)           
#             wandb.log({"train_loss": epoch_train_loss, "epoch": epoch})         

            
#             Validate one epoch
            logger.info('\n')
            logger.info('-------------- Validate {}/{} --------------'.format(epoch, config.epochs))                                   
            
            epoch_val_loss, epoch_val_acc, _ = validate_one_epoch_target_simple(model, val_dataloader, command_lang, 
                                                                          device, config, logger)

            logger.info('-------------- Val Epoch Loss {} --------------'.format(epoch_val_loss))
            writer.add_scalar('epoch_loss/val_loss', epoch_val_loss, epoch)
#             wandb.log({"val_loss": epoch_val_loss, "epoch": epoch})

            logger.info('-------------- Val Epoch Acc {} --------------'.format(epoch_val_acc))
            writer.add_scalar('epoch_acc/val_acc', epoch_val_acc, epoch)
#             wandb.log({"val_acc": epoch_val_acc, "epoch": epoch})

            
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
            
#             scheduler.step()
            
        
#         wandb.log({'Best_Epoch': best_epoch})
#         wandb.log({'Min_Val_Loss': min_val_loss})
#         wandb.log({'Max_Val_Acc': max_val_acc})
            
        writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars.json'))
        writer.close()
        
#         Test along with Train
        if config.train_test:
            
            # log_file = os.path.join(config.log_path, 'log_test_'+config.test_split+'.txt')
            # logger = get_logger(run_name, log_file, logging.DEBUG)
            logger.info('-------------- Testing --------------')
            logger.info('\n\nExperiment Name: {}'.format(config.run_name))
            
            logger.debug('Device: {}'.format(device))
            logger.info('\n')
            
#             Loading model
            if config.test_epoch == -1:
                logger.info('Loading Model From: {}'.format(os.path.join(config.model_path, 'model.pt')))
                model = Simple_Target_Predictor(config).to(device)
                checkpoint = torch.load(os.path.join(config.model_path, 'model.pt'), map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.info('Loading Model From: {}'.format(os.path.join(config.model_path, 'model_epoch_'+str(config.test_epoch)+'.pt')))
                model = Simple_Target_Predictor(config).to(device)
                checkpoint = torch.load(os.path.join(config.model_path, 'model_epoch_'+str(config.test_epoch)+'.pt'), map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])

            test_data_paths = []

            train_fname = 'train.json'
            val_fname = 'dev.json'
            test_fname = 'test.json'

    #         Choosing test split    
            if config.dataset == 'reascan':
                if True:
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label = ['All_Test']
                    
                if config.test_split == 'all_train':
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, train_fname))
                    wandb_label += ['All_Train']

                if config.test_split == 'all_val':
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, val_fname))
                    wandb_label += ['All_Val']

                if config.test_split == 'comp':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2', 'c1', 'c2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2']

                if config.test_split == 'custom_comp':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2', 'c1', 'c2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))

                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c2.json'))

                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 
                                   'New_C1', 'New_C2']

                if config.test_split == 'custom_train':
                    train_fname = config.train_fname
                    test_data_paths.append(train_fname)

                    wandb_label += ['Custom_Train']
                    
                    
            for index, test_path in enumerate(test_data_paths):
#                 Load test data
                logger.info('\n')
                logger.debug('Test Data Path: {}'.format(test_path))
                logger.info('Loading Test Data ..')
                test_dataloader = dataloader(test_path, device, batch_size=config.batch_size, random_shuffle=False)                       

#                 Loading vocab
                command_lang = torch.load(os.path.join(config.model_path, 'command_lang.pth'))

#                 Test
                _, test_acc, _ = validate_one_epoch_target_simple(model, test_dataloader, command_lang, 
                                              device, config, logger)

                logger.info('-------------- Test Acc {} --------------'.format(test_acc))
#                 wandb.log({wandb_label[index]: test_acc})

            
    if config.mode == 'train_mcnn':       
        log_file = os.path.join(config.log_path, 'log_train_mcnn.txt')
        logger = get_logger(run_name, log_file, logging.DEBUG)
        writer = SummaryWriter(config.board_path)

        command = "python -m src.main" + f"{' '.join(sys.argv)}".split('py')[1]
        logger.info("The command launched : {}".format(command))
        config.command = command
        
        logger.info('Experiment Name: {}'.format(config.run_name))
        logger.debug('Device: {}'.format(device))
        
        train_fname = config.train_fname
        val_fname = config.val_fname
        
#         Choose training split
        if config.dataset == 'reascan':         
            if config.train_split == 'all':
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                train_data_path = os.path.join(split_path, train_fname)
                val_data_path = os.path.join(split_path, val_fname)
                
            if config.train_split == 'custom':
                train_data_path = train_fname
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional')
                val_data_path = os.path.join(split_path, val_fname)

            if config.train_split in ['p1', 'p2', 'p3', 'p3-rd']: 
                #Change train_fname and val_fname accordingly
                split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + config.train_split) 
                train_data_path = os.path.join(split_path, train_fname)
                val_data_path = os.path.join(split_path, val_fname)
                
        if config.dataset == 'gscan':            
            if config.train_split == 'all':
                split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                train_data_path = os.path.join(split_path, train_fname)
                val_data_path = os.path.join(split_path, val_fname)
                
        if config.dataset == 'google':            
            if config.train_split == 'all':
                train_data_path = os.path.join(google_data_path, train_fname)
                val_data_path = os.path.join(google_data_path, val_fname)

                            
#         Loading train and val dataloaders
        logger.debug('Train Data Path: {}'.format(train_data_path))
        logger.info('Loading Train Data ..')
        train_dataloader = dataloader(train_data_path, device, batch_size=config.batch_size)
        
        logger.debug('Val Data Path: {}'.format(val_data_path))
        logger.info('Loading Val Data ..')
        val_dataloader = dataloader(val_data_path, device, batch_size=config.batch_size)
        
        # For 'include target info' experiment
        if config.include_target:
            config.vocab_size += 36
        
#         Building Model
        logger.info('Building Model ..')
        if config.num_of_streams == 'dual':
            model = MultiModalModel_Dual_MCNN(config).to(device)
                 
        config.num_params = sum(p.numel() for p in model.parameters())
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
            
#         Setting up lr scheduler (No LR Decay by default)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_step, gamma=config.lr_decay)
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, steps_per_epoch=len(train_dataloader), epochs=config.epochs)

    
#         Save generated vocab for command and action language
        command_lang, action_lang = prepareData('command', 'action', train_dataloader, include_target=config.include_target)
        torch.save(command_lang, os.path.join(config.model_path, 'command_lang.pth'))
        torch.save(action_lang, os.path.join(config.model_path, 'action_lang.pth'))
        logger.debug('Command Language: {}'.format(command_lang.word2index))
        logger.debug('Action Language: {}'.format(action_lang.word2index))
        
#         Save config file
        with open(config_file, 'wb') as f:
            pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug('Config File Saved')
        
#         Initialize wandb project
#         wandb.init(project=config.wandb_project_name, 
#                    entity="", 
#                    config=config)
    
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
            
            epoch_train_loss = train_one_epoch_mcnn(model, train_dataloader, command_lang, action_lang, 
                                               optimizer, epoch, device, config, logger, writer)
            
            logger.info('-------------- Train Epoch Loss {} --------------'.format(epoch_train_loss))
            writer.add_scalar('epoch_loss/train_loss', epoch_train_loss, epoch)           
#             wandb.log({"train_loss": epoch_train_loss, "epoch": epoch})         

            
#             Validate one epoch
            logger.info('\n')
            logger.info('-------------- Validate {}/{} --------------'.format(epoch, config.epochs))                                   
            
            epoch_val_loss, epoch_val_acc = validate_one_epoch_mcnn(model, val_dataloader, command_lang, 
                                                               action_lang, device, config, logger)

            logger.info('-------------- Val Epoch Loss {} --------------'.format(epoch_val_loss))
            writer.add_scalar('epoch_loss/val_loss', epoch_val_loss, epoch)
#             wandb.log({"val_loss": epoch_val_loss, "epoch": epoch})

            logger.info('-------------- Val Epoch Acc {} --------------'.format(epoch_val_acc))
            writer.add_scalar('epoch_acc/val_acc', epoch_val_acc, epoch)
#             wandb.log({"val_acc": epoch_val_acc, "epoch": epoch})

            
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
            
#             scheduler.step()
            
        
#         wandb.log({'Best_Epoch': best_epoch})
#         wandb.log({'Min_Val_Loss': min_val_loss})
#         wandb.log({'Max_Val_Acc': max_val_acc})
            
        writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars.json'))
        writer.close()
        
#         Test along with Train
        if config.train_test:
            
            # log_file = os.path.join(config.log_path, 'log_test_'+config.test_split+'.txt')
            # logger = get_logger(run_name, log_file, logging.DEBUG)
            logger.info('-------------- Testing --------------')
            logger.info('\n\nExperiment Name: {}'.format(config.run_name))
            
            logger.debug('Device: {}'.format(device))
            logger.info('\n')
            
#             Loading model
            if config.test_epoch == -1:
                logger.info('Loading Model From: {}'.format(os.path.join(config.model_path, 'model.pt')))
                if config.num_of_streams == 'dual':
                    model = MultiModalModel_Dual_MCNN(config).to(device)
                checkpoint = torch.load(os.path.join(config.model_path, 'model.pt'), map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.info('Loading Model From: {}'.format(os.path.join(config.model_path, 'model_epoch_'+str(config.test_epoch)+'.pt')))
                if config.num_of_streams == 'dual':
                    model = MultiModalModel_Dual_MCNN(config).to(device)
                checkpoint = torch.load(os.path.join(config.model_path, 'model_epoch_'+str(config.test_epoch)+'.pt'), map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])

            test_data_paths = []

            if config.embedding == 'modified':
                train_fname = 'train.json'
                val_fname = 'dev.json'
                test_fname = 'test.json'
            elif config.embedding == 'default':
                train_fname = 'train_default_embedding.json'
                val_fname = 'dev_default_embedding.json'
                test_fname = 'test_default_embedding.json'      

    #         Choosing test split    
            if config.dataset == 'reascan':
                if True: #Always test on random test
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label = ['All_Test']
            
                if config.test_split == 'all_train':
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, train_fname))
                    wandb_label += ['All_Train']

                if config.test_split == 'all_val':
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional') 
                    test_data_paths.append(os.path.join(split_path, val_fname))
                    wandb_label += ['All_Val']

                if config.test_split in ['p1', 'p2', 'p3', 'p3-rd']:
                    split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + config.test_split + '-test') 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label += [config.test_split+'_Test']

                if config.test_split == 'comp':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2', 'c1', 'c2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2']

                if config.test_split == 'custom_comp':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2', 'c1', 'c2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))

                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c2.json'))

                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 
                                   'New_C1', 'New_C2']
                    
                if config.test_split == 'custom_comp_add_c1c2':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))
                        
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c1_'+config.data_version+'.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c2_'+config.data_version+'.json'))
                        
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c2.json'))

                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1_Rem_'+config.data_version, 'C2_Rem_'+config.data_version, 
                                   'New_C1', 'New_C2']
                    
                if config.test_split == 'custom_comp_add_c1c2_random':
                    for split in ['a1', 'a2', 'a3', 'b1', 'b2']:
                        split_path = os.path.join(reascan_data_path, 'ReaSCAN-compositional-' + split) 
                        test_data_paths.append(os.path.join(split_path, test_fname))
                        
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c1_v1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c1_v2.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c1_v3.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c2_v1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c2_v2.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/add_c1c2_exp/test_c2_v3.json'))
                        
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c1.json'))
                    test_data_paths.append(os.path.join(data_path, 'custom/new_c1_c2_splits/new_c2.json'))

                    wandb_label += ['A1', 'A2', 'A3', 'B1', 'B2', 'C1_Rem_v1', 'C1_Rem_v2', 'C1_Rem_v3', 'C2_Rem_v1', 'C2_Rem_v2', 'C2_Rem_v3', 
                                   'New_C1', 'New_C2']

                if config.test_split == 'custom_train':
                    train_fname = config.train_fname
                    test_data_paths.append(train_fname)

                    wandb_label += ['Custom_Train']


            if config.dataset == 'gscan':
                if True:
                    split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                    test_data_paths.append(os.path.join(split_path, test_fname))
                    wandb_label = ['All_Test']
                    
                if config.test_split == 'all_train':
                    split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                    test_data_paths.append(os.path.join(split_path, train_fname))
                    wandb_label += ['All_Train']

                if config.test_split == 'all_val':
                    split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                    test_data_paths.append(os.path.join(split_path, val_fname))
                    wandb_label += ['All_Val']

                if config.test_split == 'comp':
                    for split in ['visual', 'visual_easier', 
                                  'situational_1', 'situational_2',
                                  'adverb_1', 'adverb_2', 'contextual']:
                        split_path = os.path.join(gscan_data_path, 'gSCAN-compositional_splits') 
                        test_data_paths.append(os.path.join(split_path, split+'.json'))

                    wandb_label += ['Visual', 'Visual_Easier', 'Situational_1', 'Situational_2', 
                                   'Adverb_1', 'Adverb_2', 'Contextual']
                    
                    
            if config.dataset == 'google':
                if config.test_split == 'all_train':
                    test_data_paths.append(os.path.join(google_data_path, train_fname))
                    wandb_label = ['All_Train']

                if config.test_split == 'all_val':
                    test_data_paths.append(os.path.join(google_data_path, val_fname))
                    wandb_label = ['All_Val']

                if config.test_split == 'comp':
                    for split in ['train', 'dev', 'test', 'visual', 'relation', 
                                  'relative_position_1', 'relative_position_2',
                                  'referent']:
                        test_data_paths.append(os.path.join(google_data_path, split+'.json'))

                    wandb_label = ['All_Train', 'All_Val', 'All_Test', 'Visual', 'Relation', 'Relative_Position_1',
                                   'Relative_Position_2', 'Referent']
                    
                    
            for index, test_path in enumerate(test_data_paths):
#                 Load test data
                logger.info('\n')
                logger.debug('Test Data Path: {}'.format(test_path))
                logger.info('Loading Test Data ..')
                test_dataloader = dataloader(test_path, device, batch_size=config.batch_size, random_shuffle=False)                       

#                 Loading vocab
                command_lang = torch.load(os.path.join(config.model_path, 'command_lang.pth'))
                action_lang = torch.load(os.path.join(config.model_path, 'action_lang.pth'))

#                 Test
                test_acc, _ = validate_one_epoch_mcnn(model, test_dataloader, command_lang, 
                                              action_lang, device, config, logger, exact_match=True)

                logger.info('-------------- Test Acc {} --------------'.format(test_acc))
#                 wandb.log({wandb_label[index]: test_acc})
        

        
if __name__ == '__main__':
    main()