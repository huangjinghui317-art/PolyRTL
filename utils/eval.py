from sklearn import metrics
import numpy as np
from scipy.stats import kendalltau
from scipy.stats import pearsonr


def MAE(pred, real):
    mae = metrics.mean_absolute_error(real, pred)
    mae = round(mae, 4)
    # print('MAE: ', mae)
    return mae


def NMAE(pred, real):
    mae = metrics.mean_absolute_error(real, pred) / np.mean(real)
    mae = round(mae, 3)
    print('NMAE: ', mae)
    return mae


def MAPE_one_val(pred, real):
    # print(real, pred)
    loss_ori = ((real - pred) / real) * 100
    # print('MAPE: ', round(loss_ori,1))
    return loss_ori


def MAPE(pred, real):
    loss_ori = abs(real - pred) / (abs(real)+1e-4)
    loss_new = []
    for los in loss_ori:
        if los > 1:
            los = 1
        loss_new.append(los)

    loss_new = np.array(loss_new)
    loss = loss_new.mean() * 100.0
    loss = round(loss, 4)
    # print('MAPE: ', loss)
    return loss


def MAPE_v2(pred, real):
    non_zero_mask = real != 0

    real_filtered = real[non_zero_mask]
    pred_filtered = pred[non_zero_mask]

    if len(real_filtered) == 0:
        print("MAPE: (Labels are all zero)")
        return 0

    loss_ori = abs(real_filtered - pred_filtered) / abs(real_filtered)

    loss_capped = np.clip(loss_ori, a_min=0, a_max=1)

    loss = loss_capped.mean() * 100.0
    loss = round(loss, 4)
    # print('MAPE wo zero-type data:', loss)

    return loss


def MSE(pred, real):
    mse = metrics.mean_squared_error(real, pred, squared=True)
    mse = round(mse, 3)
    print('MSE: ', mse)
    return mse


def NMSE(pred, real):
    mse = metrics.mean_squared_error(real, pred, squared=True) / np.mean(real)
    mse = round(mse, 3)
    print('NMSE: ', mse)
    return mse


def RMSE(pred, real):
    rmse = metrics.mean_squared_error(real, pred, squared=False)
    rmse = round(rmse, 3)
    print('RMSE: ', rmse)
    return rmse


def NRMSE(pred, real):
    rmse = metrics.mean_squared_error(real, pred, squared=False) / np.mean(real)
    rmse = round(rmse, 3)
    print('NRMSE: ', rmse)
    return rmse


def RRSE(pred, real):
    rrse = np.sqrt(np.sum(np.square(pred - real)) / np.sum(np.square(pred - np.mean(real))))
    rrse = round(rrse, 4)
    # print('RRSE: ', rrse)
    return rrse


def RAE(pred, real):
    rae = np.sum(pred - real) / np.sum(pred - np.mean(real))
    rae = round(rae, 3)
    print('RAE: ', rae)
    return rae


def kendall_tau(pred, real):
    corr, p_value = kendalltau(real, pred)
    k_tau = round(corr, 3)
    print('Kendall Tau: ', k_tau)
    return k_tau


def R_corr(pred, real):
    r, p = pearsonr(real, pred)
    r = round(r, 4)
    # print('R: ', r)
    return r


def R2_corr(pred, real):
    r2 = metrics.r2_score(real, pred)
    r2 = round(r2, 4)
    # print('R2: ', r2)
    return r2


def regression_metrics(pred, real):
    pred = np.array(pred)
    real = np.array(real)

    r = R_corr(pred, real)
    mape_val = MAPE(pred, real)
    rrse_val = RRSE(pred, real)
    mape2 = MAPE_v2(pred, real)

    ### average prediction error
    mae = MAE(pred, real)

    # print('\n')

    return r, mape_val, rrse_val, mape2, mae


def regression_metrics2(pred, real):
    pred = np.array(pred)
    real = np.array(real)

    # r = R_corr(pred, real)
    # r2 = R2_corr(pred, real)

    # ---- PCC ----
    if len(real) < 2 or np.std(pred) == 0 or np.std(real) == 0:
        r = np.nan
    else:
        r = R_corr(pred, real)

    # ---- R2 ----
    if len(real) < 2 or np.var(real) == 0:
        r2 = np.nan
    else:
        r2 = R2_corr(pred, real)

    mape_val = MAPE(pred, real)
    rrse_val = RRSE(pred, real)
    mape2 = MAPE_v2(pred, real)

    ### average prediction error
    mae = MAE(pred, real)

    # print('\n')

    return r, mape_val, rrse_val, mape2, mae, r2


def classify_metrics(pred, real):
    pred = np.array(pred)
    real = np.array(real)

    recall = metrics.recall_score(real, pred)
    recall = round(recall, 3)
    print('sensitivity: ', recall)

    specificity = metrics.recall_score(real, pred, pos_label=0)
    specificity = round(specificity, 3)
    print('Specificity: ', specificity)

    balanced_accuracy = (recall + specificity) / 2
    balanced_accuracy = round(balanced_accuracy, 3)
    print('Balanced Accuracy: ', balanced_accuracy)

    return recall, specificity, balanced_accuracy
