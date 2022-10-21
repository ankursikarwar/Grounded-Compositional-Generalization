import torch

def map_expression_to_tokens(expression_batch, separate_ref=False):
    expression_batch_token = []
    
    if separate_ref:
        for expression in expression_batch:
            expression_split = expression.split(' ')
            expression_split[-2] += '_ref'
            expression_split[-3] += '_ref'
            expression_batch_token.append(expression_split)
    else:
        for expression in expression_batch:
            expression_batch_token.append(expression.split(' '))
    
    return expression_batch_token

def map_two_attr_world_to_tokens(world_batch):
#     [r, g, b, circle, square, cylinder]    
    world_batch = world_batch.reshape(-1, 36, 6)
    world_batch_token = []
    
    for world in world_batch:
        world_token = []
        for cell in world:
            cell_token = ''
            if torch.sum(cell) == 0:
                cell_token += 'empty'
            else:
                if cell[0]:
                    cell_token += 'red'
                if cell[1]:
                    cell_token += 'green'
                if cell[2]:
                    cell_token += 'blue'

                if cell[3]:
                    cell_token += '_circle'
                if cell[4]:
                    cell_token += '_square'
                if cell[5]:
                    cell_token += '_cylinder'
                    
            world_token.append(cell_token)
        world_batch_token.append(world_token)
    
    return world_batch_token

def map_three_attr_world_to_tokens(world_batch):
#     [r, g, b, circle, square, cylinder, 1, 2, 3, 4]    
    world_batch = world_batch.reshape(-1, 36, 10)
    world_batch_token = []
    
    for world in world_batch:
        world_token = []
        for cell in world:
            cell_token = ''
            if torch.sum(cell) == 0:
                cell_token += 'empty'
            else:
                if cell[6]:
                    cell_token += '1'
                if cell[7]:
                    cell_token += '2'
                if cell[8]:
                    cell_token += '3'
                if cell[9]:
                    cell_token += '4'
                
                if cell[0]:
                    cell_token += '_red'
                if cell[1]:
                    cell_token += '_green'
                if cell[2]:
                    cell_token += '_blue'

                if cell[3]:
                    cell_token += '_circle'
                if cell[4]:
                    cell_token += '_square'
                if cell[5]:
                    cell_token += '_cylinder'
                    
            world_token.append(cell_token)
        world_batch_token.append(world_token)
    
    return world_batch_token

def map_to_tensor(tokens_batch, task_lang_data, device):
    
    batch_ids = []
    for tokens in tokens_batch:
        ids = []
        for word in tokens:
            ids.append(task_lang_data['lang'][word])
        batch_ids.append(ids)
            
    batch_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device)
    
    return batch_tensor

def target_to_gridnum(target_batch, device):
    
    gridnum_batch = []
    for target in target_batch:
        gridnum = (target[0]*6) + target[1]
        gridnum_batch.append(gridnum)
        
    return torch.tensor(gridnum_batch, dtype=torch.long, device=device)