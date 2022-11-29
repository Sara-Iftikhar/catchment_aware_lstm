"""
This file comparse the baselin approach
"""

import site
site.addsitedir("D:\\mytools\\AI4Water")

from ai4water import Model

import numpy as np

from utils import prepare_data, eval_model

lookback = 12
batch_size = 32
lr = 0.0001

tr_x, tr_y, val_x, val_y, test_x, test_y = prepare_data(
    target="TOC", lookback=lookback, batch_size=batch_size,
    treat_cat_as_ts=True
)

tr_x = np.concatenate(tr_x, axis=2)
val_x = np.concatenate(val_x, axis=2)
test_x = np.concatenate(test_x, axis=2)

num_inputs = tr_x.shape[-1]

layers = {
    "Input_dyn": {"batch_shape": (batch_size, lookback, num_inputs)},
    "LSTM": {"config": {"units": 32}},
    "Dense": 1
}

model = Model(model = {"layers": layers},
              batch_size=batch_size,
              epochs=100,
              lr=lr)

h = model.fit(x=tr_x, y=tr_y, validation_data=(val_x, val_y))

eval_model(model, tr_x, tr_y, batch_size=batch_size, prefix="Training")

eval_model(model, val_x, val_y, batch_size=batch_size, prefix="Validation")

eval_model(model, test_x, test_y, batch_size=batch_size, prefix="Test")