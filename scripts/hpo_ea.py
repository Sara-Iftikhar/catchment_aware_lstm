"""
==========================
10. HPO Entity Aware
==========================
"""

#%%
import site
site.addsitedir(r"E:\AA\AI4Water")
site.addsitedir(r"E:\AA\easy_mpl")

from ai4water.functional import Model

import os
SEP = os.sep
import numpy as np
import pandas as pd
import math
from skopt.plots import plot_objective
from typing import Union

from skopt.plots import plot_objective
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from SeqMetrics import RegressionMetrics

from ai4water.utils.utils import jsonize
from ai4water.utils.utils import dateandtime_now
from ai4water.preprocessing import Transformations
from ai4water.hyperopt import Categorical, Real, Integer, HyperOpt

from utils import prepare_data, eval_model
#%%

PREFIX = f"hpo_lstm_ea_{dateandtime_now()}"
ITER = 0
num_iterations = 200
lookback = 12
target = 'TP'
transformation = 'box-cox'
#%%

MONITOR = {"rmse": [], "nse": [], "r2": [], "mape": []}
#%%


def objective_fn(
        prefix: str = None,
        return_model: bool = False,
        epochs:int = 500,
        verbosity: int = 0,
        predict : bool = False,
        target = target,
        **suggestions
                )->Union[float, Model]:

    suggestions = jsonize(suggestions)

    print(suggestions)

    global ITER

    transformer = Transformations(transformation)

    tr_x, tr_y, val_x, val_y, test_x, test_y = prepare_data(
        target=target, lookback=lookback, batch_size=suggestions["batch_size"])

    x_mean = tr_x[0].mean()
    x_std = tr_x[0].std()
    x_mean_st = tr_x[1].mean()
    x_std_st = tr_x[1].std()

    y_mean = tr_y.mean()
    y_std = tr_y.std()

    tr_x[0] = (tr_x[0] - x_mean) / x_std
    tr_x[1] = (tr_x[1] - x_mean_st) / x_std_st
    #tr_y = (tr_y - y_mean) / y_std
    tr_y = transformer.fit_transform(tr_y)

    val_x[0] = (val_x[0] - x_mean) / x_std
    val_x[1] = (val_x[1] - x_mean_st) / x_std_st
    #val_y = (val_y - y_mean) / y_std
    val_y = transformer.transform(val_y)

    test_x[0] = (test_x[0] - x_mean) / x_std
    test_x[1] = (test_x[1] - x_mean_st) / x_std_st
    #test_y = (test_y - y_mean) / y_std
    test_y = transformer.transform(test_y)

    num_dyn_inputs = tr_x[0].shape[-1]
    num_static_inputs = tr_x[1].shape[-1]

    layers = {
        "Input_dyn": {"batch_shape": (suggestions["batch_size"], lookback, num_dyn_inputs)},
        "Input_cat": {"batch_shape": (suggestions["batch_size"], num_static_inputs)},
        "EALSTM": {"config": {"units": suggestions["units"], "num_static_inputs": num_static_inputs},
                   "inputs": ["Input_dyn", "Input_cat"]},
        "Dense": 1
    }

    # build model
    _model = Model(model = {"layers": layers},
                          batch_size=suggestions["batch_size"],
                          epochs=epochs,
                          lr=suggestions["lr"],
                            prefix=prefix or PREFIX,
                            verbosity=verbosity)

    # train model
    _model.fit(x=tr_x, y=tr_y, validation_data=(val_x, val_y))

    # evaluate model
    t, p = _model.predict(x=val_x, y=val_y, return_true=True,
                                             process_results=False)
    metrics = RegressionMetrics(t, p)
    val_score = metrics.rmse()

    for metric in MONITOR.keys():
        val = getattr(metrics, metric)()
        MONITOR[metric].append(val)

    # here we are evaluating model with respect to mse, therefore
    # we don't need to subtract it from 1.0
    if not math.isfinite(val_score):
        val_score = 9999

    print(f"{ITER} {val_score}")

    ITER += 1

    if return_model:
        return _model

    return val_score
#%%

param_space = [
    Integer(30, 100, name="units"),
    Real(0.00001, 0.01, name="lr"),
    Categorical(["relu", "elu", "tanh", "sigmoid"], name="activation"),
    Categorical([8, 12, 16, 24, 32, 48, 64, 128], name="batch_size")
                ]
#%%

x0 = [30, 0.001, 'relu', 8]
#%%

optimizer = HyperOpt(
    algorithm="bayes",
    objective_fn=objective_fn,
    param_space=param_space,
    x0=x0,
    num_iterations=num_iterations,
    process_results=False, # we can turn it False if we want post-processing of results
    opt_path=f"results{SEP}{PREFIX}"
)
#%%

#
# results = optimizer.fit()
#
# best_iteration = optimizer.best_iter()
#
# print(f"optimized parameters are \n{optimizer.best_paras()} at {best_iteration}")
#
# for key in ['rmse', 'mape']:
#     print(key, np.nanmin(MONITOR[key]), np.nanargmin(MONITOR[key]))
#
# for key in ['r2', 'nse']:
#     print(key, np.nanmax(MONITOR[key]), np.nanargmax(MONITOR[key]))
#
# model = objective_fn(prefix=f"{PREFIX}{SEP}best",
#                      return_model=True,
#                      epochs=500,
#                      verbosity=1,
#                      predict=True,
#                      **optimizer.best_paras())
#
# tr_x, tr_y, val_x, val_y, test_x, test_y = prepare_data(
#     target=target, lookback=lookback,
#     batch_size=optimizer.best_paras()['batch_size']
# )
#
# transformer = Transformations(transformation)
#
# x_mean = tr_x[0].mean()
# x_std = tr_x[0].std()
# x_mean_st = tr_x[1].mean()
# x_std_st = tr_x[1].std()
#
# y_mean = tr_y.mean()
# y_std = tr_y.std()
#
# tr_x[0] = (tr_x[0] - x_mean) / x_std
# tr_x[1] = (tr_x[1] - x_mean_st) / x_std_st
# #tr_y = (tr_y - y_mean) / y_std
# tr_y = transformer.fit_transform(tr_y)
#
# val_x[0] = (val_x[0] - x_mean) / x_std
# val_x[1] = (val_x[1] - x_mean_st) / x_std_st
# #val_y = (val_y - y_mean) / y_std
# val_y = transformer.transform(val_y)
#
# test_x[0] = (test_x[0] - x_mean) / x_std
# test_x[1] = (test_x[1] - x_mean_st) / x_std_st
# #test_y = (test_y - y_mean) / y_std
# test_y = transformer.transform(test_y)
#
# model.evaluate(x=tr_x, y=tr_y, metrics=['r2', 'nse', 'rmse'])
# model.evaluate(x=val_x, y=val_y, metrics=['r2', 'nse', 'rmse'])
# model.evaluate(x=test_x, y=test_y, metrics=['r2', 'nse', 'rmse'])
#
# # tr_t, tr_p = model.predict(x=tr_x, y=tr_y, return_true=True)
# #
# # val_t, val_p = model.predict(x=val_x, y=val_y, return_true=True)
#
# t, p = model.predict(x=test_x, y=test_y, return_true=True)
#
# print(f'r2 = {RegressionMetrics(t, p).r2()}')
# print(f'nse = {RegressionMetrics(t, p).nse()}')
# print(f'rmse = {RegressionMetrics(t, p).rmse()}')
#
# optimizer._plot_convergence(save=True)
#
# optimizer.plot_importance(save=True)
# plt.tight_layout()
# plt.show()
#
# _ = plot_objective(results)
# plt.savefig(f'{optimizer.opt_path}//objective.png')
# plt.show()
#
# optimizer.save_iterations_as_xy()
#
# optimizer._plot_parallel_coords(figsize=(14, 8), save=True)
#
# monitor = pd.DataFrame.from_dict(MONITOR)
#
# monitor.to_csv (f'{optimizer.opt_path}//monitor.csv', index = True, header=True)
