from utilities import *

import math

def model_run(model, data_gen):
    if data_gen.args.method == 0:
        params = {
            "mu": {"0": 0.1 * np.random.rand(1, data_gen.args.n_xind[0], data_gen.args.rank[0]), "1": 0.1 * np.random.rand(data_gen.args.rank[0], data_gen.args.n_xind[-1], 1)},
        }
    elif data_gen.args.method == 1:
        params = {
            "mu": {"0": 0.1 * np.random.rand(data_gen.args.n_xind[0], data_gen.args.rank[0]),  "1": 0.1 * np.random.rand(data_gen.args.n_xind[1], data_gen.args.rank[1])},
            "core": 0.1 * np.random.rand(data_gen.args.rank[0], data_gen.args.rank[1]),
        }
    else:
        # params = {
        #     "mu": {"0": 0.1 * np.random.rand(data_gen.args.n_xind[0], data_gen.args.rank[0]),  "1": 0.1 * np.random.rand(data_gen.args.n_xind[1], data_gen.args.rank[0])},
        #     "core": 0.1 * np.random.rand(data_gen.args.rank[0]), 
        # }
        Params_init = np.load("./initial_params/Params_init.npz")
        params = {
            "mu": {"0": Params_init['0'],  "1" : Params_init['1']},
            "core": np.ones(data_gen.args.rank[0]), 
        }
    optimizer = optax.inject_hyperparams(optax.adam)(data_gen.args.lr)
    opt_state = optimizer.init(params)

    best_test_epoch_rmse, best_test_epoch_mae, best_test_epoch_relativeL2, best_test_epoch_relativeL1 = -1, -1, -1, -1
    best_test_rmse, best_test_mae, best_test_relativeL2, best_test_relativeL1 = math.inf, math.inf, math.inf, math.inf

    count = 0

    best_total_running_time_L2 = 0.0
    best_test_time_avg_L2 = 0.0
    total_running_time_training = 0.0
    total_running_time_testing = 0.0
    
    running_time_training_seq = []
    loss_collection = []
    test_relativeL2 = []
    train_time_col = []

    key = jax.random.key(data_gen.args.seed)

    rmse, mae, relativeL1, relativeL2, run_time_test = model.pred(params)
    total_running_time_testing += run_time_test
    test_relativeL2.append(relativeL2)
    train_time_col.append(total_running_time_training)

    for epo in range(model.args.epochs):
        key, subkey = jax.random.split(key)
        X_train_collocation = get_collocation_points(data_gen)
        X_train_Bound, Y_train_Bound = get_boundary_points(data_gen)
        start = time.perf_counter()
        params, opt_state, loss = model.step(optimizer, params, opt_state, subkey, X_train_collocation, X_train_Bound, Y_train_Bound)
        end = time.perf_counter()
        loss_collection.append(loss.item())

        total_running_time_training += (end - start)
        train_time_col.append(total_running_time_training)
        rmse, mae, relativeL1, relativeL2, run_time_test = model.pred(params)
        total_running_time_testing += run_time_test
        test_relativeL2.append(relativeL2)
        count += 1
        if rmse < best_test_rmse:
            best_test_rmse = rmse
            best_test_epoch_rmse = epo
            count = 0

        if mae < best_test_mae:
            best_test_mae = mae
            best_test_epoch_mae = epo
            count = 0
            
        if relativeL2 < best_test_relativeL2:
            best_test_relativeL2 = relativeL2
            best_test_epoch_relativeL2 = epo
            best_total_running_time_L2 = total_running_time_training
            best_test_time_avg_L2 = total_running_time_testing / (epo + 1)
            count = 0
            
        if relativeL1 < best_test_relativeL1:
            best_test_relativeL1 = relativeL1
            best_test_epoch_relativeL1 = epo
            count = 0
        store_intermediate_result(model.args, train_time_col, test_relativeL2, loss_collection)
        if count == model.args.early_stop or epo == (model.args.epochs - 1):
            write_records(data_gen.args, rmse, mae, relativeL2, relativeL1, best_test_rmse, best_test_mae, best_test_relativeL2, \
                best_test_relativeL1, best_test_epoch_rmse, best_test_epoch_mae, best_test_epoch_relativeL2, best_test_epoch_relativeL1, epo, \
                best_total_running_time_L2, best_test_time_avg_L2, total_running_time_training, total_running_time_testing)
            break
        
        if epo % model.args.log_interval == 0:
            write_records(data_gen.args, rmse, mae, relativeL2, relativeL1, best_test_rmse, best_test_mae, best_test_relativeL2, \
                best_test_relativeL1, best_test_epoch_rmse, best_test_epoch_mae, best_test_epoch_relativeL2, best_test_epoch_relativeL1, epo, \
                best_total_running_time_L2, best_test_time_avg_L2, total_running_time_training, total_running_time_testing)

        if (count % model.args.lr_gap == 0) and (count != 0):
            opt_state.hyperparams['learning_rate'] *= model.args.lr_decay

