from utilities import *

import sys
def test(model, data_gen):
    model.eval()
    idx_start = 0
    diff_accum_square = 0.0
    diff_accum_abs = 0.0
    n_total = data_gen.X_test.shape[0]
    # model.node_pickup_idx = get_idx_kernel(data_gen)
    runtime_total = 0.0
    
    with torch.no_grad():
        while idx_start < n_total:
            idx_end = min(n_total, idx_start + model.args.test_batch)
            start = time.perf_counter()
            y_pred = model.pred(data_gen.X_test[idx_start:idx_end]).to('cpu')
            end = time.perf_counter()
            runtime_total += end - start
            diff = y_pred - torch.tensor(data_gen.Y_test[idx_start:idx_end].reshape(-1), dtype=FLOAT)
            diff_accum_square += torch.sum(torch.square(diff)).item()
            diff_accum_abs += torch.sum(torch.abs(diff)).item()
            idx_start = idx_end
    mae = diff_accum_abs/n_total
    relativeL1 = diff_accum_abs/data_gen.YSUM_ABS
    relativeL2 = np.sqrt(diff_accum_square)/data_gen.YNorm2_test
    rmse = np.sqrt(diff_accum_square/n_total)
    return rmse, mae, relativeL2, relativeL1, runtime_total


def train(model, data_gen, optimizer):
    model.train()
    # model.node_pickup_idx = get_idx_kernel(data_gen, False)
    optimizer.zero_grad()
    model.X_train_Bound, model.Y_train_Bound = get_boundary_points(data_gen)

    loss_total = 0.0
    time_usage = 0.0

    start = time.perf_counter()
    loss = model.forward_boundary()
    loss.backward()
    end = time.perf_counter()
    loss_total += loss.item()
    time_usage += end - start

    idx_col_pickup = get_idx_col(data_gen)
    idx_start = 0
    n_total = idx_col_pickup.shape[0]
    while idx_start < n_total:
        idx_end = min(n_total, idx_start + data_gen.args.n_train_batch)
        model.X_train_collocation, model.Force_train_collocation = get_collocation_points(data_gen, idx_col_pickup[idx_start:idx_end])
        start = time.perf_counter()
        loss = model.forward_col()
        loss.backward()
        end = time.perf_counter()
        loss_total += loss.item()
        time_usage += end - start
        idx_start = idx_end
    if data_gen.args.regulator:
        start = time.perf_counter()
        loss = model.forward_reg()
        loss.backward()
        end = time.perf_counter()
        time_usage += end - start
        loss_total += loss.item()
    optimizer.step()
    return loss_total, time_usage
    

def model_run(model, data_gen):
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=model.args.lr)

    best_test_epoch_rmse, best_test_epoch_mae, best_test_epoch_relativeL2, best_test_epoch_relativeL1 = -1, -1, -1, -1
    best_test_rmse, best_test_mae, best_test_relativeL2, best_test_relativeL1 = math.inf, math.inf, math.inf, math.inf
    start_epoch = 0
    count = 0
    if data_gen.args.resume_training:
        model, optimizer, start_epoch, count, best_test_rmse, best_test_mae, best_test_relativeL2, best_test_relativeL1, \
            best_test_epoch_rmse, best_test_epoch_mae, best_test_epoch_relativeL2, best_test_epoch_relativeL1 = load_ckp(model, optimizer)

    """
    model.X_train_collocation = prepare_collocation_point(data_gen.x_train_collocation, model.args)
    model.Force_train_collocation = torch.tensor(data_gen.F_train_PDE, dtype=FLOAT, device=model.args.device)
    model.X_train_Bound = [torch.tensor(data_gen.x_train_boundary[:, i:(i+1)], dtype=FLOAT, device=model.args.device) for i in range(model.args.n_order)]
    model.Y_train_Bound = torch.tensor(data_gen.y_train_boundary, dtype=FLOAT, device=model.args.device)
    """

    best_total_running_time_L2 = 0.0
    best_test_time_avg_L2 = 0.0
    total_running_time_training = 0.0
    total_running_time_testing = 0.0
    
    running_time_training_seq = []
    loss_collection = []
    test_relativeL2 = []
    train_time_col = []

    rmse, mae, relativeL2, relativeL1, run_time_test = test(model, data_gen)
    total_running_time_testing += run_time_test
    test_relativeL2.append(relativeL2)
    train_time_col.append(total_running_time_training)

    while start_epoch < data_gen.args.epochs:
        loss, training_time = train(model, data_gen, optimizer)

        loss_collection.append(loss)

        total_running_time_training += training_time
        train_time_col.append(total_running_time_training)

        rmse, mae, relativeL2, relativeL1, run_time_test = test(model, data_gen)

        total_running_time_testing += run_time_test
        test_relativeL2.append(relativeL2)
        count += 1
        if rmse < best_test_rmse:
            best_test_rmse = rmse
            best_test_epoch_rmse = start_epoch
            count = 0

        if mae < best_test_mae:
            best_test_mae = mae
            best_test_epoch_mae = start_epoch
            # count = 0
            
        if relativeL2 < best_test_relativeL2:
            best_test_relativeL2 = relativeL2
            best_test_epoch_relativeL2 = start_epoch
            best_state = model.save_state()
            torch.save(best_state, model.args.best_model_dir)
            best_total_running_time_L2 = total_running_time_training
            best_test_time_avg_L2 = total_running_time_testing / (start_epoch + 2)
            count = 0

        if relativeL1 < best_test_relativeL1:
            best_test_relativeL1 = relativeL1
            best_test_epoch_relativeL1 = start_epoch
            # count = 0
        store_intermediate_result(model.args, train_time_col, test_relativeL2, loss_collection)
        if count == model.args.early_stop or start_epoch  == (model.args.epochs - 1):
            write_records(data_gen.args, rmse, mae, relativeL1, relativeL2, best_test_rmse, best_test_mae, best_test_relativeL1, best_test_relativeL2, best_test_epoch_rmse, best_test_epoch_mae, best_test_epoch_relativeL1, best_test_epoch_relativeL2, start_epoch, \
                best_total_running_time_L2, best_test_time_avg_L2, total_running_time_training, total_running_time_testing)
            break
        
        if start_epoch % model.args.log_interval == 0:
            checkpoint = {
                'epoch': start_epoch + 1,
                'state_dict': model.save_state(),
                'optimizer': optimizer.state_dict(),
                'count': count,
                'btr': best_test_rmse,
                'btm': best_test_mae,
                'btr2': best_test_relativeL2,
                'btr1': best_test_relativeL1,
                'bter': best_test_epoch_rmse,
                'btem': best_test_epoch_mae,
                'bter2': best_test_epoch_relativeL2,
                'bter1': best_test_epoch_relativeL1
            }
            torch.save(checkpoint, data_gen.args.checkpoint_dir)
            write_records(data_gen.args, rmse, mae, relativeL1, relativeL2, best_test_rmse, best_test_mae, best_test_relativeL1, best_test_relativeL2, best_test_epoch_rmse, best_test_epoch_mae, best_test_epoch_relativeL1, best_test_epoch_relativeL2, start_epoch, \
                best_total_running_time_L2, best_test_time_avg_L2, total_running_time_training, total_running_time_testing)

        if (count % model.args.lr_gap == 0) and (count != 0):
            for g in optimizer.param_groups:
                g['lr'] *= model.args.lr_decay

        start_epoch += 1

