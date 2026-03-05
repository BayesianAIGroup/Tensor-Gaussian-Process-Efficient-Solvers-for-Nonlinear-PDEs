import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
import numpy as np

from functools import partial
import configparser

import time

k_mapping = {0:0.5, 1:1.5, 2:2.5, 3:3.5, 4:4.5, 5:jnp.inf}

@jit
def u_elliptic_zhitong(x, y):
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) + 4.0 * jnp.sin(4.0 * jnp.pi * x) * jnp.sin(4.0 * jnp.pi * y)

@jit
def pde_force_gen(x, y):
    uval = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) + 4.0 * jnp.sin(4.0 * jnp.pi * x) * jnp.sin(4.0 * jnp.pi * y)
    dudxx = - (jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) - 64.0 * (jnp.pi**2) * jnp.sin(4.0 * jnp.pi * x) * jnp.sin(4.0 * jnp.pi * y)
    return uval**3 - 2 * dudxx


def write_log(args):
    config = configparser.ConfigParser()

    config['DEFAULT'] = vars(args)
    with open(args.log_store_path + ".ini", 'w+') as configfile:
        config.write(configfile)

def write_records(args, err, err1, err2, test_min_err, test_min_err1, test_min_err2, min_epoch_test, min_epoch_test1, min_epoch_test2, log_lsx, \
epoch, best_total_running_time_L2, best_test_time_avg_L2, total_running_time_training, total_running_time_testing):
    log = "Training at epoch: [" + str(epoch) + "] spending {" + str(total_running_time_training) + "} seconds\n"
    log += "test RMSE-(" + str(err) + "), test L2 relative mean error-(" + str(err2) + "), test L1 relative mean error-(" + str(err1) + ")\n"
    log += "best test RMSE: (" + str(test_min_err) + ") at epoch-[" + str(min_epoch_test) + "]\n"
    log += "best test L2 relative mean error: (" + str(test_min_err2) + ") at epoch-[" + str(min_epoch_test2) + "]\n"
    log += "best test L1 relative mean error: (" + str(test_min_err1) + ") at epoch-[" + str(min_epoch_test1) + "]\n"
    log += "log_ls under best test error: (" + str(log_lsx) + ")\n"
    log += "total training time until best Relative L2: {" + str(best_total_running_time_L2) + "} seconds\n"
    log += "avg testing time until best Relative L2: {" + str(best_test_time_avg_L2) + "} seconds\n"
    log += "total testing time until now: {" + str(total_running_time_testing) + "} seconds\n"
    with open(args.log_store_path + ".ini", 'a') as configfile:
        configfile.write(log)
        configfile.write("*************************************************************************\n\n")

def store_intermediate_result(args, train_time_col, test_relativeL2, loss_collection):
    processing_data = {'training_time' : np.array(train_time_col), 'test_L2' : np.array(test_relativeL2), 'training_loss' : np.array(loss_collection)}

    np.savez(args.process_store_path, **processing_data)