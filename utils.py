
import site
site.addsitedir("D:\\mytools\\AI4Water")


import numpy as np
import pandas as pd
from ai4water.datasets import Quadica
from ai4water.utils.utils import TrainTestSplit, print_something
from ai4water.utils import prepare_data as to_supervised

from SeqMetrics import RegressionMetrics


def prepare_data(
        target,
        batch_size:int,
        input_features = None,
        lookback = 12,
        drop_remainder:bool = True,
        treat_cat_as_ts:bool = False,
):
    """

    >>> tr_x, tr_y, val_x, val_y, test_x, test_y = prepare_data(
    ... target="TOC", batch_size=10
    ... )

    """

    ds = Quadica(path = r'F:\data\Quadica')
    dyn, cat = ds.fetch_monthly(features=target, max_nan_tol=0)

    # %%
    print(cat.shape)

    # %%
    cat = cat.dropna(axis=1)
    cat.drop(['Station', 'Q_StartDate', 'Q_EndDate'], axis=1, inplace=True)

    print(cat.shape)

    # %%
    if input_features is None:
        input_features = ['median_Q', 'OBJECTID', 'avg_temp', 'precip', 'pet']
    output_features = [f'median_C_{target}']

    tr_val_stns, test_stns, *_ = TrainTestSplit(seed=313).split_by_random(dyn['OBJECTID'].unique())
    tr_stns, val_stns, *_ = TrainTestSplit(seed=313).split_by_random(tr_val_stns)

    tr_df_x = []
    tr_df_y = []
    tr_dfs_cat = []
    val_df_x = []
    val_df_y = []
    val_dfs_cat = []
    test_df_x = []
    test_df_y = []
    test_dfs_cat = []

    for (idx_dyn, grp_dyn), (idx_cat, grp_cat) in zip(dyn.groupby("OBJECTID"), cat.groupby("OBJECTID")):

        assert idx_dyn == idx_cat

        df_dyn = pd.DataFrame()
        df_dyn[input_features+output_features] = grp_dyn[input_features+output_features]

        df_dyn_x, _, df_dyn_y = to_supervised(
            df_dyn, lookback=lookback,
            num_inputs=len(input_features),
            num_outputs=len(output_features)
        )

        if treat_cat_as_ts:
            df_cat, _, _ = to_supervised(
                grp_cat,
                lookback=lookback,
                num_inputs = grp_cat.shape[-1]
            )
        else:
            df_cat = grp_cat[lookback - 1:]

        if idx_dyn in tr_stns:
            tr_df_x.append(df_dyn_x)
            tr_df_y.append(df_dyn_y)
            tr_dfs_cat.append(df_cat)

        elif idx_dyn in val_stns:
            val_df_x.append(df_dyn_x)
            val_df_y.append(df_dyn_y)
            val_dfs_cat.append(df_cat)

        elif idx_dyn in test_stns:
            test_df_x.append(df_dyn_x)
            test_df_y.append(df_dyn_y)
            test_dfs_cat.append(df_cat)

    train_x = np.row_stack(tr_df_x)
    train_y = np.row_stack(tr_df_y).reshape(-1, 1)
    train_x_cat = np.row_stack(tr_dfs_cat)

    val_x = np.row_stack(val_df_x)
    val_y = np.row_stack(val_df_y).reshape(-1, 1)
    val_x_cat = np.row_stack(val_dfs_cat)

    test_x = np.row_stack(test_df_x)
    test_y = np.row_stack(test_df_y).reshape(-1, 1)
    test_x_cat = np.row_stack(test_dfs_cat)

    if drop_remainder:
        residue = train_y.shape[0] % batch_size
        train_x = train_x[0:-residue]
        train_x_cat = train_x_cat[0:-residue]
        train_y = train_y[0:-residue]

        residue = val_y.shape[0] % batch_size
        val_x = val_x[0:-residue]
        val_x_cat = val_x_cat[0:-residue]
        val_y = val_y[0:-residue]

        residue = test_y.shape[0] % batch_size
        test_x = test_x[0:-residue]
        test_x_cat = test_x_cat[0:-residue]
        test_y = test_y[0:-residue]

    print_something("\nTraining")
    print_something(train_x, "x")
    print_something(train_x_cat, "x")
    print_something(train_y, "y")
    print_something("\nValidation")
    print_something(val_x, "x")
    print_something(val_x_cat, "x")
    print_something(val_y, "y")
    print_something("\nTest")
    print_something(test_x, "x")
    print_something(test_x_cat, "x")
    print_something(test_y, "y")

    return [train_x, train_x_cat], train_y, \
        [val_x, val_x_cat], val_y, \
        [test_x, test_x_cat], test_y


def eval_model(model, inputs, outputs, batch_size, prefix=""):

    test_p = model.predict(x=inputs, batch_size=batch_size, verbose=0)

    metrics = RegressionMetrics(outputs, test_p)
    print(f"\n {prefix}")
    for metric in ["nse", "rmse", "r2", "mape"]:
        val = getattr(metrics, metric)()
        print(metric, round(val, 4))

    return
