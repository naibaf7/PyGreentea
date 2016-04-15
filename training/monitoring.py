import os
from collections import OrderedDict

def get_losses_from_folder_with_nohup(folder_containing_nohup):
    try:
        file_path = os.path.join(folder_containing_nohup, 'nohup.err')
        return get_losses_from_file_path(file_path)
    except IOError:
        # file not found... try *.out
        file_path = os.path.join(folder_containing_nohup, 'nohup.out')
        return get_losses_from_file_path(file_path)


def get_losses_from_file_path(path_to_log_file):
    with open(path_to_log_file, 'r') as log_file:
        return get_losses_from_log_file(log_file)


def get_losses_from_log_file(log_file):
    iteration_line_count = 0
    losses = OrderedDict()
    for line in log_file.readlines():
        if 'loss = ' in line:
            iteration_line_count += 1
            iteration_number_str = line.split('Iteration ')[1].split(',')[0]
            splitted = line.split('loss = ')
            loss_str = splitted[1][0:11]
            try:
                loss = float(loss_str)
            except: 
                print('found weird loss value: {0}'.format(loss_str))
                continue
            if len(iteration_number_str) < 7:
                iteration_number = int(iteration_number_str)
                losses[iteration_number] = loss
            else:
                print('found weird iteration number string: {0}'.format(iteration_number_str))
    return losses

