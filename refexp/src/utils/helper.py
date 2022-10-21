import os 
import torch

def gpu_init_pytorch(gpu_num):
    torch.cuda.set_device(int(gpu_num))
    device = torch.device("cuda:{}".format(
        gpu_num) if torch.cuda.is_available() else "cpu")
    return device

def create_save_directories(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise ValueError("Run_Name already used")
        
def out_model_archi(model, num_params, config):
    with open(config.model_path + '/architecture.txt', 'a') as f_out:
        f_out.write('---------------------------------------\n')
        f_out.write('Number of Params: ' + str(num_params) + '\n')
        f_out.write('Model: ' + str(model) + '\n')
        f_out.write('---------------------------------------\n')             
        f_out.close()