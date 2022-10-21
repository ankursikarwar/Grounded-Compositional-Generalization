import torch
import torchtext as tt
from torchtext.legacy import data

def dataloader(data_path, device, batch_size=32, random_shuffle=True):

    TARGET_LOCATION_FIELD = data.RawField(postprocessing=lambda x: torch.DoubleTensor(x).to(device))
    SITUATION_FIELD = data.RawField(postprocessing=lambda x: torch.DoubleTensor(x).to(device))
    
    COMMAND_FIELD = data.RawField()
    TARGET_ACTION_FIELD = data.RawField()
    
    dataset = data.TabularDataset(path=data_path, format="json", 
                                  fields={'target_location': ('target_location', TARGET_LOCATION_FIELD), 
                                          'situation': ('situation', SITUATION_FIELD), 
                                          'input_command': ('input_command', COMMAND_FIELD), 
                                          'target_sequence': ('target_sequence', TARGET_ACTION_FIELD)})
        
    iterator = data.Iterator(dataset, batch_size=batch_size, 
                             device=device,shuffle=random_shuffle)
        
    return iterator