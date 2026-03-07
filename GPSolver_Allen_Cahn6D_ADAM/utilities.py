import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import tensorly as tl

import math

import configparser
import time

FLOAT = torch.float64
Integer = torch.int64


def write_log(args):
    config = configparser.ConfigParser()

    config['DEFAULT'] = vars(args)
    with open(args.log_store_path + ".ini", 'w+') as configfile:
        config.write(configfile)

def write_records(args, rmse, mae, relativeL1, relativeL2, best_test_rmse, best_test_mae, best_test_relativeL1, best_test_relativeL2, best_test_epoch_rmse, best_test_epoch_mae, \
    best_test_epoch_relativeL1, best_test_epoch_relativeL2, epoch, best_total_running_time_L2, best_test_time_avg_L2, total_running_time_training, total_running_time_testing):
    log = "Training at epoch: [" + str(epoch) + "] spending {" + str(total_running_time_training) + "} seconds\n"
    log += "test RMSE-(" + str(rmse) + "), test L2 Relative mean error-(" + str(relativeL2) + "), test MAE error-(" + str(mae) + "), test L1 Relative error-(" + str(relativeL1) + ")\n"
    log += "best test RMSE: (" + str(best_test_rmse) + ") at epoch-[" + str(best_test_epoch_rmse) + "]\n"
    log += "best test L1 relative error: (" + str(best_test_relativeL1) + ") at epoch-[" + str(best_test_epoch_relativeL1) + "]\n"
    log += "best test L2 relative mean error: (" + str(best_test_relativeL2) + ") at epoch-[" + str(best_test_epoch_relativeL2) + "]\n"
    log += "best test MAE: (" + str(best_test_mae) + ") at epoch-[" + str(best_test_epoch_mae) + "]\n"
    log += "total training time until best Relative L2: {" + str(best_total_running_time_L2) + "} seconds\n"
    log += "avg testing time until best Relative L2: {" + str(best_test_time_avg_L2) + "} seconds\n"
    log += "total testing time until now: {" + str(total_running_time_testing) + "} seconds\n"
    with open(args.log_store_path + ".ini", 'a') as configfile:
        configfile.write(log)
        configfile.write("*************************************************************************\n\n")
        
def write_broken_info(args):
    log = "Algorithm fails due to overflow of loss!"
    with open(args.log_store_path + ".ini", 'a') as configfile:
        configfile.write(log)
        configfile.write("*************************************************************************\n\n")

    
def prepare_collocation_point(X_Traing_PDE, args):
    x_train_col = []
    for i in range(args.n_order):
        x_train_col.append(torch.tensor(X_Traing_PDE[:, i:(i+1)], dtype=FLOAT, device=args.device, requires_grad=True))
    return x_train_col

def get_collocation_points(data_gen, idx_col_pickup):
    # idx_pickup = np.random.choice(data_gen.args.n_train_collocation, size=min(data_gen.args.n_train_collocation, data_gen.args.train_batch_collocation), replace=False)
    x_train_col = prepare_collocation_point(data_gen.x_train_collocation[idx_col_pickup], data_gen.args)
    return x_train_col, torch.tensor(data_gen.F_train_PDE[idx_col_pickup], dtype=FLOAT, device=data_gen.args.device)

def get_idx_col(data_gen):
    return np.random.choice(data_gen.args.n_train_collocation, size=min(data_gen.args.n_train_collocation, data_gen.args.train_batch_collocation), replace=False)
    
def get_boundary_points(data_gen):
    idx_pickup = np.random.choice(data_gen.args.n_train_boundary, size=min(data_gen.args.n_train_boundary, data_gen.args.train_batch_boundary), replace=False)
    x_train_boundary = data_gen.x_train_boundary[idx_pickup]
    
    return [torch.tensor(x_train_boundary[:, i:(i+1)], dtype=FLOAT, device=data_gen.args.device) for i in range(data_gen.args.n_order)], \
        torch.tensor(data_gen.y_train_boundary[idx_pickup], dtype=FLOAT, device=data_gen.args.device)
        
# def get_idx_kernel(data_gen, Total=True):
#     idx_pickup = []
#     for i in range(data_gen.args.n_order):
#         if Total:
#             idx_pickup.append(torch.arange(data_gen.args.n_xtrain[i], dtype=Integer, device=data_gen.args.device))
#         else:
#             idx_pickup.append(torch.tensor(np.random.choice(data_gen.args.n_xtrain[i], int(data_gen.args.n_xtrain[i] * data_gen.args.pickup_rate[i]), replace=False), dtype=Integer, device=data_gen.args.device))
#     return idx_pickup

def load_ckp(model, optimizer):
    checkpoint = torch.load(model.args.checkpoint_dir)
    model.load_state(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'], checkpoint['count'], checkpoint['btr'], checkpoint['btm'], checkpoint['btr2'], \
        checkpoint['btr1'], checkpoint['bter'], checkpoint['btem'], checkpoint['bter2'], checkpoint['bter1']
    
def store_intermediate_result(args, train_time_col, test_relativeL2, loss_collection):
    processing_data = {'training_time' : np.array(train_time_col), 'test_L2' : np.array(test_relativeL2), 'training_loss' : np.array(loss_collection)}

    np.savez(args.process_store_path, **processing_data)