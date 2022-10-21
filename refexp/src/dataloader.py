import json
import numpy as np

from torch.utils.data import Dataset

class RefExp_Dataset(Dataset):
    def __init__(self, data_path, split, analysis=False):
        with open(data_path, 'r') as fp:
            refexp = json.load(fp)
        
        self.data = refexp[split]
        self.analysis = analysis
        
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        if self.analysis:
            return (' '.join(self.data[index]['Expression']), 
                    np.array(self.data[index]['Situation'], dtype=np.float32), 
                    np.array(self.data[index]['Target_Loc'], dtype=np.int64), 
                    np.array(self.data[index]['Referent_Loc'], dtype=np.int64))
        else:
            return (' '.join(self.data[index]['Expression']), 
                    np.array(self.data[index]['Situation'], dtype=np.float32), 
                    np.array(self.data[index]['Target_Loc'], dtype=np.int64))