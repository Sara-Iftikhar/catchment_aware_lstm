
import site
site.addsitedir("D:\\mytools\\AI4Water")


from ai4water import Model

from utils import prepare_data, eval_model

lookback = 12
batch_size = 32
lr = 0.0001

# %%
tr_x, tr_y, val_x, val_y, test_x, test_y = prepare_data(
    target="TOC", lookback=lookback, batch_size=batch_size)

# %%
num_dyn_inputs = tr_x[0].shape[-1]
num_static_inputs = tr_x[1].shape[-1]

# %%
layers = {
    "Input_dyn": {"batch_shape": (batch_size, lookback, num_dyn_inputs)},
    "Input_cat": {"batch_shape": (batch_size, num_static_inputs)} ,
    "LSTM": {"config": {"units": 32}, "inputs": "Input_dyn"},
    "Dense_cat": {"config": {"units": 32}, "inputs": "Input_cat"},
    "Multiply": {"config": {}, "inputs": ["LSTM", "Dense_cat"]},
    "Dense": 1
}

# %%
model = Model(model = {"layers": layers},
              batch_size=batch_size,
              epochs=100,
              lr=lr)

# %%
h = model.fit(x=tr_x, y=tr_y, #validation_data=(val_x, val_y)
              )

eval_model(model, tr_x, tr_y, batch_size=batch_size, prefix="Training")

eval_model(model, val_x, val_y, batch_size=batch_size, prefix="Validation")

eval_model(model, test_x, test_y, batch_size=batch_size, prefix="Test")

