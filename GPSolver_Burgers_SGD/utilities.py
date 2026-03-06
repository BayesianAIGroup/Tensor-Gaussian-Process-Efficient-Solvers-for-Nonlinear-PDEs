import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
import numpy as np

import tensorly as tl

from functools import partial
import configparser
import time

k_mapping = {0:0.5, 1:1.5, 2:2.5, 3:3.5, 4:4.5, 5:jnp.inf}

def write_log(args):
    config = configparser.ConfigParser()

    config['DEFAULT'] = vars(args)
    with open(args.log_store_path + ".ini", 'w+') as configfile:
        config.write(configfile)

def write_records(args, rmse, mae, relativeL2, relativeL1, best_test_rmse, best_test_mae, best_test_relativeL2, best_test_relativeL1, best_test_epoch_rmse, best_test_epoch_mae, \
    best_test_epoch_relativeL2, best_test_epoch_relativeL1, epoch, best_total_running_time_L2, best_test_time_avg_L2, total_running_time_training, total_running_time_testing):
    log = "Training at epoch: [" + str(epoch) + "] spending {" + str(total_running_time_training) + "} seconds\n"
    log += "test RMSE-(" + str(rmse) + "), test L2 Relative mean error-(" + str(relativeL2) + "), test MAE error-(" + str(mae)+ "), test L1 Relative mean error-(" + str(relativeL1) + ")\n"
    log += "best test RMSE: (" + str(best_test_rmse) + ") at epoch-[" + str(best_test_epoch_rmse) + "]\n"
    log += "best test L2 relative mean error: (" + str(best_test_relativeL2) + ") at epoch-[" + str(best_test_epoch_relativeL2) + "]\n"
    log += "best test MAE: (" + str(best_test_mae) + ") at epoch-[" + str(best_test_epoch_mae) + "]\n"
    log += "best test L1 relative mean error: (" + str(best_test_relativeL1) + ") at epoch-[" + str(best_test_epoch_relativeL1) + "]\n"
    log += "total training time until best Relative L2: {" + str(best_total_running_time_L2) + "} seconds\n"
    log += "avg testing time until best Relative L2: {" + str(best_test_time_avg_L2) + "} seconds\n"
    log += "total testing time until now: {" + str(total_running_time_testing) + "} seconds\n"
    with open(args.log_store_path + ".ini", 'a') as configfile:
        configfile.write(log)
        configfile.write("*************************************************************************\n\n")

def get_collocation_points(data_gen):
    idx_pickup = np.random.choice(data_gen.args.n_train_collocation, size=min(data_gen.args.n_train_collocation, data_gen.args.train_batch_collocation), replace=False)
    x_train_col = data_gen.x_c[idx_pickup]
    return jnp.array(x_train_col)
    
def get_boundary_points(data_gen):
    idx_pickup = np.random.choice(data_gen.args.n_train_boundary, size=min(data_gen.args.n_train_boundary, data_gen.args.train_batch_boundary), replace=False)

    return jnp.array(data_gen.x_b[idx_pickup]), jnp.array(data_gen.y_b[idx_pickup])


def store_intermediate_result(args, train_time_col, test_relativeL2, loss_collection):
    processing_data = {'training_time' : np.array(train_time_col), 'test_L2' : np.array(test_relativeL2), 'training_loss' : np.array(loss_collection)}

    np.savez(args.process_store_path, **processing_data)