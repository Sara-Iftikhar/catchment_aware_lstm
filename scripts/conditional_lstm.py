"""
====================
4. Conditional
====================
"""

#%%
import site
site.addsitedir(r"E:\AA\AI4Water")
site.addsitedir(r"E:\AA\easy_mpl")

from ai4water import Model
from ai4water.preprocessing import Transformations
from utils import prepare_data, eval_model

# %%

lookback = 12
batch_size = 64
units = 64
lr = 0.0001

# %%

tr_x, tr_y, val_x, val_y, test_x, test_y = prepare_data(
    target="TP", lookback=lookback, batch_size=batch_size)

# %%

transformer = Transformations('box-cox')

# %%

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

# %%
num_dyn_inputs = tr_x[0].shape[-1]
num_static_inputs = tr_x[1].shape[-1]

# %%
layers = {
    "Input_dyn": {"batch_shape": (batch_size, lookback, num_dyn_inputs)},
    "Input_cat": {"batch_shape": (batch_size, num_static_inputs)} ,
    "Conditionalize": {"config": {"units": units},
                       "inputs": ["Input_cat"],
                       "outptus": "cond_out"},

    "LSTM": {"config": {"units": units}, "inputs": "Input_dyn",
             'call_args': {'initial_state': ['Conditionalize', 'Conditionalize']}},

    "Dense": 1
}
#%%

model = Model(model = {"layers": layers},
              batch_size=batch_size,
              epochs=100,
              lr=lr)

# %%
h = model.fit(x=tr_x, y=tr_y, validation_data=(val_x, val_y))
#%%

eval_model(model, tr_x, tr_y, batch_size=batch_size, prefix="Training")
#%%

eval_model(model, val_x, val_y, batch_size=batch_size, prefix="Validation")
#%%

eval_model(model, test_x, test_y, batch_size=batch_size, prefix="Test")
#%%

model.predict(test_x, test_y, plots=['residual', 'edf', 'regression', 'prediction'
                                     ])