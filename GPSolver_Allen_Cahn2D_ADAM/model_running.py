from utilities import *
import math


def model_run(model, data_gen):
    if data_gen.args.method == 0:
        muDict = {}
        pre_rank = 1
        for i in range(data_gen.args.n_order):
            if i == 0: n_xtrain = data_gen.args.n_x1train
            else: n_xtrain = data_gen.args.n_x2train
            muDict[str(i)] = 0.1 * np.random.rand(pre_rank, n_xtrain, data_gen.args.rank[i])
            pre_rank = data_gen.args.rank[i]
            
        params = {"mu": muDict,}
    elif data_gen.args.method == 1:
        muDict = {}
        for i in range(data_gen.args.n_order):
            if i == 0: n_xtrain = data_gen.args.n_x1train
            else: n_xtrain = data_gen.args.n_x2train
            muDict[str(i)] = 0.1 * np.random.rand(n_xtrain, data_gen.args.rank[i])

        params = {
            "mu": muDict,
            "core": 0.1 * np.random.rand(*data_gen.args.rank),
        } 
    else:
        muDict = {}
        Params_init = np.load("./initial_params/Params_init.npz")
        for i in range(data_gen.args.n_order):
            # if i == 0: n_xtrain = data_gen.args.n_x1train
            # else: n_xtrain = data_gen.args.n_x2train
            # muDict[str(i)] = 0.1 * np.random.rand(data_gen.args.n_xind[i], data_gen.args.rank[0])
            muDict[str(i)] = Params_init[str(i)]

        params = {
            "mu": muDict,
            # "core": 0.1 * np.random.rand(data_gen.args.rank[0]), 
            "core": np.ones(data_gen.args.rank[0]),
        }
            
    optimizer = optax.inject_hyperparams(optax.adam)(data_gen.args.lr)
    opt_state = optimizer.init(params)

    test_min_err, test_min_err2, test_min_err1 = math.inf, math.inf, math.inf
    min_epoch_test, min_epoch_test2, min_epoch_test1 = -1, -1, -1
    log_lsx = data_gen.args.log_lsx

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

    err, err1, err2, run_time_test = model.pred(params)
    total_running_time_testing += run_time_test
    test_relativeL2.append(err2)

    train_time_col.append(total_running_time_training)
    loss = model.loss(params, key)
    loss_collection.append(loss.item())
    

    for epo in range(model.args.epochs):
        # X_Collocation = data_gen.train_pde_gen()
        # X_train_bound, Y_train_bound = data_gen.train_boundary_gen()
        key, subkey = jax.random.split(key)
        start = time.perf_counter()
        params, opt_state, loss = model.step(optimizer, params, opt_state, subkey)
        end = time.perf_counter()
        loss_collection.append(loss.item())

        total_running_time_training += (end - start)
        train_time_col.append(total_running_time_training)

        err, err1, err2, run_time_test = model.pred(params)
        total_running_time_testing += run_time_test
        test_relativeL2.append(err2)

        count += 1
        if err < test_min_err:
            test_min_err = err
            min_epoch_test = epo
            count = 0

        if err1 < test_min_err1:
            test_min_err1 = err1
            min_epoch_test1 = epo
            count = 0

        if err2 < test_min_err2:
            test_min_err2 = err2
            min_epoch_test2 = epo
            best_total_running_time_L2 = total_running_time_training
            best_test_time_avg_L2 = total_running_time_testing / (epo + 1)
            count = 0

        store_intermediate_result(model.args, train_time_col, test_relativeL2, loss_collection)
        if count == model.args.early_stop or epo == (model.args.epochs - 1):
            write_records(data_gen.args, err, err1, err2, test_min_err, test_min_err1, test_min_err2, min_epoch_test, min_epoch_test1, min_epoch_test2, log_lsx, epo, \
            best_total_running_time_L2, best_test_time_avg_L2, total_running_time_training, total_running_time_testing)
            break
        
        if epo % model.args.log_interval == 0:
            write_records(data_gen.args, err, err1, err2, test_min_err, test_min_err1, test_min_err2, min_epoch_test, min_epoch_test1, min_epoch_test2, log_lsx, epo, \
            best_total_running_time_L2, best_test_time_avg_L2, total_running_time_training, total_running_time_testing)

        if (count % model.args.lr_gap == 0) and (count != 0):
            opt_state.hyperparams['learning_rate'] *= model.args.lr_decay

