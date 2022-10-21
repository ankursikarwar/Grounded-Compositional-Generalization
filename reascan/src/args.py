import argparse

def build_parser():
    
    parser = argparse.ArgumentParser(description='Run Model')
    
    # Global arguments
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'train_mcnn', 'test', 'train_target_predictor', 'train_target_simple'], help='Modes: train, test')
    parser.add_argument('--display_freq', type=int, default= 200, help='Number of examples after which to display Loss and Accuracy')
    
    parser.add_argument('--log_folder', type=str, default='./logs/', help='Logs folder')
    parser.add_argument('--model_folder', type=str, default='./models/', help='Models folder')
    parser.add_argument('--outputs_folder', type=str, default='./outputs/', help='Outputs folder')
    parser.add_argument('--board_folder', type=str, default='./runs/', help='Runs folder')
    
    parser.add_argument('--train_test', dest='train_test', 
                        action='store_true', help='Test along with Train')
    parser.add_argument('--no-train_test', dest='train_test', 
                        action='store_false', help='Test along with Train')
    parser.set_defaults(train_test=True)
    
    parser.add_argument('--save_all_epochs', dest='save_all_epochs', 
                        action='store_true', help='Save models for all epochs for analysis')
    parser.add_argument('--no-save_all_epochs', dest='save_all_epochs', 
                        action='store_false', help='Save models for all epochs for analysis')
    parser.set_defaults(save_all_epochs=False)
    
    # Test arguments
    parser.add_argument('--test_split', type=str, required=True, choices=['all_train', 'all_val', 'all_test', 
                                                                          'p1', 'p2', 'p3', 'p3-rd', 
                                                                          'comp', 'custom_comp', 'custom_train', 
                                                                          'custom_comp_add_c1c2', 'custom_comp_add_c1c2_random'])
    parser.add_argument('--test_epoch', type=int, default=-1, help='Which epoch to test')
    parser.add_argument('--wandb_run_name', type=str, default='debug', help='Specify the wandb run name')
    
    # Train arguments
    parser.add_argument('--gpu', type=int, required=True, help='Specify the gpu to use')
    parser.add_argument('--seed', type=int, default=6174, help='Default seed to set')
    parser.add_argument('--data_path', type=str, default='./data/', help='Specify the path for ReaSCAN data')
    
    parser.add_argument('--run_name', type=str, required=True, help='Run name for logs')
    parser.add_argument('--wandb_project_name', type=str, required=False, help='Specify the wandb project name')
    parser.add_argument('--dataset', type=str, default='reascan', choices=['reascan', 'gscan', 'google'], help='Choose dataset to train on')
    parser.add_argument('--train_split', type=str, default='all', choices=['all', 'custom', 'p1', 'p2', 'p3', 'p3-rd'])
    parser.add_argument('--train_fname', type=str, required=True)
    parser.add_argument('--val_fname', type=str, required=True)
    
    parser.add_argument('--load_config', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
                        
    parser.add_argument('--lr_decay', type=float, default=1, help='Gamma for Learning rate scheduler')
    parser.add_argument('--decay_step', type=int, default=10000, help='Step size for Learning rate scheduler')
    
    parser.add_argument('--batch_size', type=int, required=True, help='Batch Size')
    parser.add_argument('--epochs', type=int, required=True, help='Maximum # of training epochs')
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'adadelta', 'sgd'], help='Optimizer for training')
    
    parser.add_argument('--embedding', type=str, default='modified', choices=['modified', 'default'], help='Embedding variant to use')
                        
    # Model arguments (Remaining in config files)
    parser.add_argument('--num_of_streams', type=str, default='dual', choices=['dual', 'single'], help='Num of streams in transformer')
    parser.add_argument('--pos_embed', type=str, default='learned', choices=['learned', 'sincos'], help='Position Embedding to use')
    
    # Experiment arguments
    parser.add_argument('--data_version', type=str, default='v1', choices=['v1', 'v2', 'v3', 'v4'], help='Data Version for Add C1 C2 Experiment')
    
    # Target Localization Experiments
    parser.add_argument('--target_layer', type=int, default=-1, help='Layer to use for target prediction')
    parser.add_argument('--random_layer', dest='random_layer', 
                        action='store_true', help='Train Linear layer on random representation')
    parser.add_argument('--no-random_layer', dest='random_layer', 
                        action='store_false', help='Train Linear layer on random representation')
    parser.set_defaults(random_layer=False)
    
    # Include Target Information Experiment
    parser.add_argument('--include_target', dest='include_target', 
                        action='store_true', help='Include Target Information in input')
    parser.add_argument('--no-include_target', dest='include_target', 
                        action='store_false', help='Include Target Information in input')
    parser.set_defaults(include_target=False)
    
    # Simple Transformer Target Localization Experiment
    parser.add_argument('--simple_num_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--simple_num_heads', type=int, default=1, help='Number of heads')
    parser.add_argument('--simple_embed_size', type=int, default=64, help='Embedding dimension')
    
    return parser