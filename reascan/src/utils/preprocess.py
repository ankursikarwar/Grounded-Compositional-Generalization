import json
import argparse
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Union

file_paths_reascan = ['../../data/ReaSCAN-v1.1/ReaSCAN-compositional/data-compositional-splits.txt', 
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-a1/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-a2/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-a3/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-b1/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-b2/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-c1/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-c2/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-p1/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-p1-test/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-p2/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-p2-test/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-p3/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-p3-test/data-compositional-splits.txt',
                      '../../data/ReaSCAN-v1.1/ReaSCAN-compositional-p3-rd/data-compositional-splits.txt']

file_paths_gscan = ['../../data/ReaSCAN-v1.1/gSCAN-compositional_splits/dataset.txt']

file_paths_google = ['../../data/spatial_relation_splits/dataset.txt']


def get_target_loc_vector(target_dict, grid_size):
    
    target_pos = target_dict['position']
    row = int(target_pos['row'])
    col = int(target_pos['column'])
    target_loc_vector = [[0] * 6 for i in range(grid_size)]
    target_loc_vector[row][col] = 1
    
    return target_loc_vector


def parse_sparse_situation_gscan(situation_representation: dict, grid_size: int) -> np.ndarray:
    """
    Each grid cell in a situation is fully specified by a vector:
    [_ _ _ _   _       _       _    _ _ _ _    _   _ _ _ _]
     1 2 3 4 square cylinder circle y g r b  agent E S W N
     _______  _________________________ _______ ______ _______
       size             shape            color  agent agent dir.
    :param situation_representation: data from dataset.txt at key "situation".
    :param grid_size: int determining row/column number.
    :return: grid to be parsed by computational models.
    """
    num_object_attributes = len([int(bit) for bit in situation_representation["target_object"]["vector"]])
    # Object representation + agent bit + agent direction bits (see docstring).
    num_grid_channels = num_object_attributes + 1 + 4

    # Initialize the grid.
    grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)

    # Place the agent.
    agent_row = int(situation_representation["agent_position"]["row"])
    agent_column = int(situation_representation["agent_position"]["column"])
    agent_direction = int(situation_representation["agent_direction"])
    agent_representation = np.zeros([num_grid_channels], dtype=np.int)
    agent_representation[-5] = 1
    agent_representation[-4 + agent_direction] = 1
    grid[agent_row, agent_column, :] = agent_representation

    # Loop over the objects in the world and place them.
    placed_position = set([])
    for placed_object in situation_representation["placed_objects"].values():
        object_vector = np.array([int(bit) for bit in placed_object["vector"]], dtype=np.int)
        object_row = int(placed_object["position"]["row"])
        object_column = int(placed_object["position"]["column"])
        placed_position.add((object_row, object_column))
        if (object_row, object_column) not in placed_position:
            grid[object_row, object_column, :] = np.concatenate([object_vector, np.zeros([5], dtype=np.int)])
        else:
            overlay = np.concatenate([object_vector, np.zeros([5], dtype=np.int)])
            grid[object_row, object_column, :] += overlay # simply add it.
    return grid


def parse_sparse_situation_reascan(situation_representation: dict, grid_size: int) -> np.ndarray:
    """
    Each grid cell in a situation is fully specified by a vector:
    [_ _ _ _   _       _       _     _  _ _ _ _ _ _ _ _ _ _ _ _    _   _ _ _ _]
     1 2 3 4 circle cylinder square box r b g y 1 2 3 4 r b g y  agent E S W N
     _______  _________________________ _______ _______ _______ ______ _______
       size             shape            color  box_size box_color agent agent dir.
    :param situation_representation: data from dataset.txt at key "situation".
    :param grid_size: int determining row/column number.
    :return: grid to be parsed by computational models.
    """
    num_object_attributes = len([int(bit) for bit in situation_representation["target_object"]["vector"]])
    num_box_attributes = 8
    # Object representation + agent bit + agent direction bits (see docstring).
    num_grid_channels = num_object_attributes + num_box_attributes + 1 + 4

    # Initialize the grid.
    grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)

    # Place the agent.
    agent_row = int(situation_representation["agent_position"]["row"])
    agent_column = int(situation_representation["agent_position"]["column"])
    agent_direction = int(situation_representation["agent_direction"])
    agent_representation = np.zeros([num_grid_channels], dtype=np.int)
    agent_representation[-5] = 1
    agent_representation[-4 + agent_direction] = 1
    grid[agent_row, agent_column, :] = agent_representation

    # Loop over the objects in the world and place them.
    placed_position = set([])
    for placed_object in situation_representation["placed_objects"].values():
        object_vector = np.array([int(bit) for bit in placed_object["vector"]], dtype=np.int)
        if placed_object["object"]["shape"] == "box":
            box_vec_1 = np.array([int(bit) for bit in placed_object["vector"][0:4]], dtype=np.int)
            box_vec_2 = np.array([int(bit) for bit in placed_object["vector"][8:12]], dtype=np.int)
            box_vector = np.concatenate([box_vec_1, box_vec_2])
            object_vector[0:4] = 0
            object_vector[8:12] = 0
        else:
            box_vector = np.zeros([8], dtype=np.int)
        object_row = int(placed_object["position"]["row"])
        object_column = int(placed_object["position"]["column"])
        placed_position.add((object_row, object_column))
        if (object_row, object_column) not in placed_position:
            grid[object_row, object_column, :] = np.concatenate([object_vector, box_vector, np.zeros([5], dtype=np.int)])
        else:
            overlay = np.concatenate([object_vector, box_vector, np.zeros([5], dtype=np.int)])
            grid[object_row, object_column, :] += overlay # simply add it.
    return grid


def data_loader(file_path: str, args) -> Dict[str, Union[List[str], np.ndarray]]:
    with open(file_path, 'r') as infile:
        all_data = json.load(infile)
        grid_size = int(all_data["grid_size"])
        splits = list(all_data["examples"].keys())
        loaded_data = {}
        for split in splits:
            loaded_data[split] = []
            print(split + ':')
            for data_example in tqdm(all_data["examples"][split]):
                if args.dataset == 'google':
                    # input_command = data_example["command"].split(',')
                    comm = data_example["command"].split(',')
                    input_command = []
                    for i in comm:
                        input_command+=i.split(' ')
                else:
                    input_command = data_example["command"].split(',')
                
                target_command = data_example["target_commands"].split(',')
                target_location = get_target_loc_vector(data_example["situation"]["target_object"],
                                                        grid_size)

                if args.dataset in ['gscan', 'google']:
                    situation = parse_sparse_situation_gscan(situation_representation=data_example["situation"],
                                                   grid_size=grid_size)
                if args.dataset == 'reascan':
                    if args.embedding == 'modified':
                        situation = parse_sparse_situation_reascan(situation_representation=data_example["situation"],
                                                       grid_size=grid_size)
                    elif args.embedding == 'default':
                        situation = parse_sparse_situation_gscan(situation_representation=data_example["situation"],
                                                       grid_size=grid_size)
                
                loaded_data[split].append({"input_command": input_command,
                                           "target_sequence": target_command,
                                           "target_location": target_location,
                                           "situation": situation.tolist()}) 
    return loaded_data


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='reascan', choices=['reascan', 'gscan', 'google'], help='Choose dataset for preprocessing')
    parser.add_argument('--embedding', type=str, default='modified', choices=['modified', 'default'], help='Which embedding to use')
        
    args = parser.parse_args()
    
    if args.dataset == 'reascan':
        file_paths = file_paths_reascan
    elif args.dataset == 'gscan':
        file_paths = file_paths_gscan
    elif args.dataset == 'google':
        file_paths = file_paths_google

    for file_path in file_paths:
        print('Processing {} ...'.format(file_path.split('/')[3]))
        data = data_loader(file_path, args)

        for split, dt in data.items():
            print('Dumping {} json ...'.format(split))
            if args.dataset == 'reascan':
                if args.embedding == 'modified':
                    with open(file_path.split('data-compositional')[0] + split + '.json', 'w') as f:
                        for line in dt:
                            f.write(json.dumps(line) + '\n')
                elif args.embedding == 'default':
                    with open(file_path.split('data-compositional')[0] + split + '_default_embedding.json', 'w') as f:
                        for line in dt:
                            f.write(json.dumps(line) + '\n')
            elif args.dataset == 'gscan':
                with open(file_path.split('dataset')[0] + split + '.json', 'w') as f:
                    for line in dt:
                        f.write(json.dumps(line) + '\n')
            elif args.dataset == 'google':
                with open(file_path.split('dataset')[0] + split + '.json', 'w') as f:
                    for line in dt:
                        f.write(json.dumps(line) + '\n')