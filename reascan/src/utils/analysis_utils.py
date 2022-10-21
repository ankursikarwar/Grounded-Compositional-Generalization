import numpy

def parse_command(command):

    that_is_one = False
    that_is_two = False

    command_parse = {'target_shape': None, 
                     'target_color': None, 
                     'target_size': None, 
                     'that_is_one_shape': None, 
                     'that_is_one_color': None, 
                     'that_is_one_size': None,
                     'that_is_two_shape': None, 
                     'that_is_two_color': None, 
                     'that_is_two_size': None,
                     'that_is_one_relation': None, 
                     'that_is_two_relation': None
                    }

    for word in command:
        if word == 'blue': 
            if that_is_one == False:
                command_parse['target_color'] = 'blue'
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_color'] = 'blue'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_color'] = 'blue'
        if word == 'green':
            if that_is_one == False:
                command_parse['target_color'] = 'green'
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_color'] = 'green'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_color'] = 'green'
        if word == 'red':
            if that_is_one == False:
                command_parse['target_color'] = 'red'
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_color'] = 'red'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_color'] = 'red'
        if word == 'yellow':
            if that_is_one == False:
                command_parse['target_color'] = 'yellow'
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_color'] = 'yellow'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_color'] = 'yellow'

        if word == 'square': 
            if that_is_one == False:
                command_parse['target_shape'] = 'square'
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_shape'] = 'square'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_shape'] = 'square'
        if word == 'cylinder':
            if that_is_one == False:
                command_parse['target_shape'] = 'cylinder'
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_shape'] = 'cylinder'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_shape'] = 'cylinder'
        if word == 'circle':
            if that_is_one == False:
                command_parse['target_shape'] = 'circle'
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_shape'] = 'circle'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_shape'] = 'circle'
        if word == 'box':
            if that_is_one == False:
                command_parse['target_shape'] = 'box'
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_shape'] = 'box'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_shape'] = 'box'

        if word == 'small': 
            if that_is_one == False:
                command_parse['target_size'] = 'small'
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_size'] = 'small'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_size'] = 'small'
        if word == 'big':
            if that_is_one == False:
                command_parse['target_size'] = 'big'
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_size'] = 'big'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_size'] = 'big'

        if word == 'row':
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_relation'] = 'same_row'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_relation'] = 'same_row'
        if word == 'column':
            if that_is_one == True and that_is_two == False:
                command_parse['that_is_one_relation'] = 'same_column'
            if that_is_one == True and that_is_two == True:
                command_parse['that_is_two_relation'] = 'same_column'

        if word == 'that':
            if that_is_one == True:
                that_is_two = True
            that_is_one = True
            
    return command_parse


def process_relative_clause(command_parse, situation, target_clause=False, first_clause=False, second_clause=False):
    if first_clause:
        first_clause_shape = command_parse['that_is_one_shape']
        first_clause_color = command_parse['that_is_one_color']
        first_clause_size = command_parse['that_is_one_size']
        
        shape_match = []
        color_match = []
        shape_color_match = []
        shape_color_match_size_val = []
        correct_match = []
        final_match = []
        
        for index, embedding in enumerate(situation):
            if first_clause_shape == 'circle':
                if embedding[4] == 1.0:
                    shape_match.append(index)
            if first_clause_shape == 'cylinder':
                if embedding[5] == 1.0:
                    shape_match.append(index)
            if first_clause_shape == 'square':
                if embedding[6] == 1.0:
                    shape_match.append(index)
            if first_clause_shape == None:
                if embedding[0] == 1 or embedding[1] == 1 or embedding[2] == 1 or embedding[3] == 1:
                    shape_match.append(index)

            if first_clause_color == 'red':
                if embedding[8] == 1.0:
                    color_match.append(index)
            if first_clause_color == 'blue':
                if embedding[9] == 1.0:
                    color_match.append(index)
            if first_clause_color == 'green':
                if embedding[10] == 1.0:
                    color_match.append(index)
            if first_clause_color == 'yellow':
                if embedding[11] == 1.0:
                    color_match.append(index)   
            if first_clause_color == None:
                if embedding[0] == 1 or embedding[1] == 1 or embedding[2] == 1 or embedding[3] == 1:
                    color_match.append(index)
                    
        for val in shape_match:
            if val in color_match:
                shape_color_match.append(val)
                if situation[val][0] == 1.0:
                    shape_color_match_size_val.append(1)
                if situation[val][1] == 1.0:
                    shape_color_match_size_val.append(2)
                if situation[val][2] == 1.0:
                    shape_color_match_size_val.append(3)
                if situation[val][3] == 1.0:
                    shape_color_match_size_val.append(4)
    
        
        if first_clause_size == 'small':
            curr_size = min(shape_color_match_size_val)
            for index, val in enumerate(shape_color_match):
                if shape_color_match_size_val[index] == curr_size:
                    correct_match.append(val)
        elif first_clause_size == 'big':
            curr_size = max(shape_color_match_size_val)
            for index, val in enumerate(shape_color_match):
                if shape_color_match_size_val[index] == curr_size:
                    correct_match.append(val)
        else:
            for index, val in enumerate(shape_color_match):
                correct_match.append(val)
            
        if command_parse['that_is_one_relation'] == 'same_row':
            for match in correct_match:
                curr_row = match // 6
                if curr_row == 0:
                    final_match.append(0)
                    final_match.append(1)
                    final_match.append(2)
                    final_match.append(3)
                    final_match.append(4)
                    final_match.append(5)
                if curr_row == 1:
                    final_match.append(6)
                    final_match.append(7)
                    final_match.append(8)
                    final_match.append(9)
                    final_match.append(10)
                    final_match.append(11)
                if curr_row == 2:
                    final_match.append(12)
                    final_match.append(13)
                    final_match.append(14)
                    final_match.append(15)
                    final_match.append(16)
                    final_match.append(17)
                if curr_row == 3:
                    final_match.append(18)
                    final_match.append(19)
                    final_match.append(20)
                    final_match.append(21)
                    final_match.append(22)
                    final_match.append(23)
                if curr_row == 4:
                    final_match.append(24)
                    final_match.append(25)
                    final_match.append(26)
                    final_match.append(27)
                    final_match.append(28)
                    final_match.append(29)
                if curr_row == 5:
                    final_match.append(30)
                    final_match.append(31)
                    final_match.append(32)
                    final_match.append(33)
                    final_match.append(34)
                    final_match.append(35)
                    
        if command_parse['that_is_one_relation'] == 'same_column':
            for match in correct_match:
                curr_column = match % 6
                if curr_column == 0:
                    final_match.append(0)
                    final_match.append(6)
                    final_match.append(12)
                    final_match.append(18)
                    final_match.append(24)
                    final_match.append(30)
                if curr_column == 1:
                    final_match.append(1)
                    final_match.append(7)
                    final_match.append(13)
                    final_match.append(19)
                    final_match.append(25)
                    final_match.append(31)
                if curr_column == 2:
                    final_match.append(2)
                    final_match.append(8)
                    final_match.append(14)
                    final_match.append(20)
                    final_match.append(26)
                    final_match.append(32)
                if curr_column == 3:
                    final_match.append(3)
                    final_match.append(9)
                    final_match.append(15)
                    final_match.append(21)
                    final_match.append(27)
                    final_match.append(33)
                if curr_column == 4:
                    final_match.append(4)
                    final_match.append(10)
                    final_match.append(16)
                    final_match.append(22)
                    final_match.append(28)
                    final_match.append(34)
                if curr_column == 5:
                    final_match.append(5)
                    final_match.append(11)
                    final_match.append(17)
                    final_match.append(23)
                    final_match.append(29)
                    final_match.append(35)
        
        final_match = list(set(final_match))
        
        return final_match
            
    if second_clause:
        second_clause_shape = command_parse['that_is_two_shape']
        second_clause_color = command_parse['that_is_two_color']
        second_clause_size = command_parse['that_is_two_size']
        
        shape_match = []
        color_match = []
        shape_color_match = []
        shape_color_match_size_val = []
        correct_match = []
        final_match = []
        
        for index, embedding in enumerate(situation):
            if second_clause_shape == 'circle':
                if embedding[4] == 1.0:
                    shape_match.append(index)
            if second_clause_shape == 'cylinder':
                if embedding[5] == 1.0:
                    shape_match.append(index)
            if second_clause_shape == 'square':
                if embedding[6] == 1.0:
                    shape_match.append(index)
            if second_clause_shape == None:
                if embedding[0] == 1 or embedding[1] == 1 or embedding[2] == 1 or embedding[3] == 1:
                    shape_match.append(index)

            if second_clause_color == 'red':
                if embedding[8] == 1.0:
                    color_match.append(index)
            if second_clause_color == 'blue':
                if embedding[9] == 1.0:
                    color_match.append(index)
            if second_clause_color == 'green':
                if embedding[10] == 1.0:
                    color_match.append(index)
            if second_clause_color == 'yellow':
                if embedding[11] == 1.0:
                    color_match.append(index)     
            if second_clause_color == None:
                if embedding[0] == 1 or embedding[1] == 1 or embedding[2] == 1 or embedding[3] == 1:
                    color_match.append(index)
                    
        for val in shape_match:
            if val in color_match:
                shape_color_match.append(val)
                if situation[val][0] == 1.0:
                    shape_color_match_size_val.append(1)
                if situation[val][1] == 1.0:
                    shape_color_match_size_val.append(2)
                if situation[val][2] == 1.0:
                    shape_color_match_size_val.append(3)
                if situation[val][3] == 1.0:
                    shape_color_match_size_val.append(4)
    
        
        if second_clause_size == 'small':
            curr_size = min(shape_color_match_size_val)
            for index, val in enumerate(shape_color_match):
                if shape_color_match_size_val[index] == curr_size:
                    correct_match.append(val)
        elif second_clause_size == 'big':
            curr_size = max(shape_color_match_size_val)
            for index, val in enumerate(shape_color_match):
                if shape_color_match_size_val[index] == curr_size:
                    correct_match.append(val)
        else:
            for index, val in enumerate(shape_color_match):
                correct_match.append(val)
        
        if command_parse['that_is_two_relation'] == 'same_row':
            for match in correct_match:
                curr_row = match // 6
                if curr_row == 0:
                    final_match.append(0)
                    final_match.append(1)
                    final_match.append(2)
                    final_match.append(3)
                    final_match.append(4)
                    final_match.append(5)
                if curr_row == 1:
                    final_match.append(6)
                    final_match.append(7)
                    final_match.append(8)
                    final_match.append(9)
                    final_match.append(10)
                    final_match.append(11)
                if curr_row == 2:
                    final_match.append(12)
                    final_match.append(13)
                    final_match.append(14)
                    final_match.append(15)
                    final_match.append(16)
                    final_match.append(17)
                if curr_row == 3:
                    final_match.append(18)
                    final_match.append(19)
                    final_match.append(20)
                    final_match.append(21)
                    final_match.append(22)
                    final_match.append(23)
                if curr_row == 4:
                    final_match.append(24)
                    final_match.append(25)
                    final_match.append(26)
                    final_match.append(27)
                    final_match.append(28)
                    final_match.append(29)
                if curr_row == 5:
                    final_match.append(30)
                    final_match.append(31)
                    final_match.append(32)
                    final_match.append(33)
                    final_match.append(34)
                    final_match.append(35)
                    
        if command_parse['that_is_two_relation'] == 'same_column':
            for match in correct_match:
                curr_column = match % 6
                if curr_column == 0:
                    final_match.append(0)
                    final_match.append(6)
                    final_match.append(12)
                    final_match.append(18)
                    final_match.append(24)
                    final_match.append(30)
                if curr_column == 1:
                    final_match.append(1)
                    final_match.append(7)
                    final_match.append(13)
                    final_match.append(19)
                    final_match.append(25)
                    final_match.append(31)
                if curr_column == 2:
                    final_match.append(2)
                    final_match.append(8)
                    final_match.append(14)
                    final_match.append(20)
                    final_match.append(26)
                    final_match.append(32)
                if curr_column == 3:
                    final_match.append(3)
                    final_match.append(9)
                    final_match.append(15)
                    final_match.append(21)
                    final_match.append(27)
                    final_match.append(33)
                if curr_column == 4:
                    final_match.append(4)
                    final_match.append(10)
                    final_match.append(16)
                    final_match.append(22)
                    final_match.append(28)
                    final_match.append(34)
                if curr_column == 5:
                    final_match.append(5)
                    final_match.append(11)
                    final_match.append(17)
                    final_match.append(23)
                    final_match.append(29)
                    final_match.append(35)
        
        final_match = list(set(final_match))
        
        return final_match
    
    if target_clause:
        target_clause_shape = command_parse['target_shape']
        target_clause_color = command_parse['target_color']
        target_clause_size = command_parse['target_size']
        
        shape_match = []
        color_match = []
        shape_color_match = []
        shape_color_match_size_val = []
        correct_match = []
        final_match = []
        
        for index, embedding in enumerate(situation):
            if target_clause_shape == 'circle':
                if embedding[4] == 1.0:
                    shape_match.append(index)
            if target_clause_shape == 'cylinder':
                if embedding[5] == 1.0:
                    shape_match.append(index)
            if target_clause_shape == 'square':
                if embedding[6] == 1.0:
                    shape_match.append(index)
            if target_clause_shape == None:
                if embedding[0] == 1 or embedding[1] == 1 or embedding[2] == 1 or embedding[3] == 1:
                    shape_match.append(index)

            if target_clause_color == 'red':
                if embedding[8] == 1.0:
                    color_match.append(index)
            if target_clause_color == 'blue':
                if embedding[9] == 1.0:
                    color_match.append(index)
            if target_clause_color == 'green':
                if embedding[10] == 1.0:
                    color_match.append(index)
            if target_clause_color == 'yellow':
                if embedding[11] == 1.0:
                    color_match.append(index)    
            if target_clause_color == None:
                if embedding[0] == 1 or embedding[1] == 1 or embedding[2] == 1 or embedding[3] == 1:
                    color_match.append(index)
                    
        for val in shape_match:
            if val in color_match:
                shape_color_match.append(val)
                if situation[val][0] == 1.0:
                    shape_color_match_size_val.append(1)
                if situation[val][1] == 1.0:
                    shape_color_match_size_val.append(2)
                if situation[val][2] == 1.0:
                    shape_color_match_size_val.append(3)
                if situation[val][3] == 1.0:
                    shape_color_match_size_val.append(4)
        
        final_match = list(set(shape_color_match))
        
        return final_match
    
    
def find_possible_target_c2(command, situation):
    
    command_parse = parse_command(command)
    situation = numpy.array(situation).reshape(36, -1)

    target_clause_grids = process_relative_clause(command_parse, situation, target_clause=True, first_clause=False, second_clause=False)
    first_clause_grids = process_relative_clause(command_parse, situation, target_clause=False, first_clause=True, second_clause=False)
    second_clause_grids = process_relative_clause(command_parse, situation, target_clause=False, first_clause=False, second_clause=True)

    first_second_clause_grids = []
    possible_targets = []
    possible_targets_size = []
    final_possible_targets = []
    for grid in first_clause_grids:
        if grid in second_clause_grids:
            if situation[grid][0] == 1 or situation[grid][1] == 1 or situation[grid][2] == 1 or situation[grid][3] == 1:
                first_second_clause_grids.append(grid)

    for grid in first_second_clause_grids:
        if grid in target_clause_grids:
            possible_targets.append(grid)
            
            
    for grid in possible_targets:
        if situation[grid][0] == 1:
            possible_targets_size.append(1)
        if situation[grid][1] == 1:
            possible_targets_size.append(2)
        if situation[grid][2] == 1:
            possible_targets_size.append(3)
        if situation[grid][3] == 1:
            possible_targets_size.append(4)
            
        
    if command_parse['target_size'] == 'small':
        if len(possible_targets_size) != 0: 
            curr_size = min(possible_targets_size)
            for index, grid in enumerate(possible_targets):
                if possible_targets_size[index] == curr_size:
                    final_possible_targets.append(grid)
            
    elif command_parse['target_size'] == 'big':
        if len(possible_targets_size) != 0: 
            curr_size = max(possible_targets_size)
            for index, grid in enumerate(possible_targets):
                if possible_targets_size[index] == curr_size:
                    final_possible_targets.append(grid)
        
    else:
        final_possible_targets = possible_targets
            
    return final_possible_targets