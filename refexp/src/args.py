import argparse

def build_parser():
    
    parser = argparse.ArgumentParser(description='Run Model')
    
    # Global arguments
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Modes: train, test')
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
    parser.set_defaults(save_all_epochs=True)
    
    # Test arguments
    parser.add_argument('--wandb_run_name', type=str, default='debug', help='Specify the wandb run name')
    
    # Train arguments
    parser.add_argument('--gpu', type=int, required=True, help='Specify the gpu to use')
    parser.add_argument('--seed', type=int, default=6174, help='Default seed to set')
    parser.add_argument('--data_path', type=str, default='./data/', help='Specify the path for ReaSCAN data')
    
    parser.add_argument('--run_name', type=str, required=True, help='Run name for logs')
    parser.add_argument('--wandb_project_name', type=str, required=True, help='Specify the wandb project name')
    parser.add_argument('--dataset', type=str, default='reascan', choices=[], help='Choose dataset to train on')
    
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--data_fname', type=str, required=True)
    parser.add_argument('--load_config', type=str, required=True)
    
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch Size')
    parser.add_argument('--epochs', type=int, required=True, help='Maximum # of training epochs')
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'adadelta', 'sgd'], help='Optimizer for training')
    
    # Model arguments
    parser.add_argument('--num_layers', type=int, required=True, help='Number of attention layers')
    parser.add_argument('--num_heads', type=int, required=True, help='Number of attention heads')
    
    # Include position information for three attribute with rel experiment
    parser.add_argument('--include_pos', dest='include_pos', 
                        action='store_true', help='Include pos information in command part')
    parser.add_argument('--no-include_pos', dest='include_pos', 
                        action='store_false', help='Include pos information in command part')
    parser.set_defaults(include_pos=False)
    
    parser.add_argument('--sparse', dest='sparse', 
                        action='store_true', help='Make pos embedding matrix sparse')
    parser.add_argument('--no-sparse', dest='sparse', 
                        action='store_false', help='Make pos embedding matrix sparse')
    parser.set_defaults(sparse=False)
    
    return parser