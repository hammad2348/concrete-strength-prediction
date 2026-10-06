"""
Predicting concrete compressive strength from mix composition.

Dataset: Yeh, I. (1998). Concrete Compressive Strength. UCI Machine Learning
Repository. 1030 mixes, 8 inputs, target = 28-day-equivalent compressive
strength in MPa.

Run:  python concrete_strength.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # save figures to file instead of opening a window
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42

COLS = ["cement", "slag", "fly_ash", "water", "superplasticizer",
        "coarse_agg", "fine_agg", "age"]


# ---------------------------------------------------------------- 1. load data
def load_data():
    """Return X (features) and y (strength in MPa)."""
    try:
        from ucimlrepo import fetch_ucirepo
        repo = fetch_ucirepo(id=165)
        X = repo.data.features.copy()
        y = repo.data.targets.copy()
    except Exception as err:
        print(f"ucimlrepo failed ({err}); falling back to local Concrete_Data.xls")
        # Download the zip from the UCI page, unzip, put Concrete_Data.xls next
        # to this script, and pip install xlrd.
        raw = pd.read_excel("Concrete_Data.xls")
        X, y = raw.iloc[:, :8], raw.iloc[:, 8]

    # Rename by position. The column order is fixed by the dataset
    # documentation, so this is safer than matching the long original names.
    X.columns = COLS
    y = pd.Series(np.asarray(y).ravel(), name="strength")
    return X, y


# ------------------------------------------------- 2. engineered features
def add_engineering_features(X):
    """Add two features justified by concrete theory, not by trial and error."""
    X = X.copy()

    # Abrams' law: strength falls as the water-to-binder ratio rises.
    # Slag and fly ash are cementitious, so they belong in the binder, not
    # the aggregate.
    binder = X["cement"] + X["slag"] + X["fly_ash"]
    X["water_binder"] = X["water"] / binder

    # Strength gain against time is roughly logarithmic, which is why 28 days
    # is the standard test age.
    X["log_age"] = np.log(X["age"])

    return X


# ------------------------------------------------------------- 3. evaluation
def evaluate(name, model, Xtr, Xte, ytr, yte, results):
    """Fit, score on the held-out test set, and cross-validate on the training set."""
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    rmse = mean_squared_error(yte, pred) ** 0.5
    r2 = r2_score(yte, pred)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, Xtr, ytr, cv=cv,
                                scoring="neg_root_mean_squared_error")
    cv_rmse = -cv_scores.mean()

    results.append({"model": name, "test_rmse_mpa": round(rmse, 2),
                    "test_r2": round(r2, 3), "cv_rmse_mpa": round(cv_rmse, 2)})
    return model, pred


def main():
    X_raw, y = load_data()
    print(f"{len(X_raw)} mixes, {X_raw.shape[1]} raw features")
    print(f"strength range: {y.min():.1f} to {y.max():.1f} MPa, mean {y.mean():.1f}\n")

    X = add_engineering_features(X_raw)

    # ------------------------------------------------- exploratory plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].scatter(X["cement"], y, s=8, alpha=0.4)
    ax[0].set(xlabel="cement (kg/m3)", ylabel="strength (MPa)")
    ax[1].scatter(X["water_binder"], y, s=8, alpha=0.4, color="tab:red")
    ax[1].set(xlabel="water / binder ratio", ylabel="strength (MPa)")
    ax[2].scatter(X["age"], y, s=8, alpha=0.4, color="tab:green")
    ax[2].set(xlabel="age (days)", ylabel="strength (MPa)", xscale="log")
    fig.tight_layout()
    fig.savefig("fig1_exploration.png", dpi=150)

    # ------------------------------------------------- split once, reuse
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=RANDOM_STATE)
    raw_cols = COLS                       # before feature engineering
    all_cols = list(X.columns)            # after feature engineering

    results = []

    # Baseline: linear regression on the raw ingredient masses.
    evaluate("Linear (raw features)", LinearRegression(),
             Xtr[raw_cols], Xte[raw_cols], ytr, yte, results)

    # Same model, with the two physics-based features added.
    evaluate("Linear (+ w/b, log age)", LinearRegression(),
             Xtr[all_cols], Xte[all_cols], ytr, yte, results)

    # Nonlinear models.
    rf, rf_pred = evaluate(
        "Random forest",
        RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),
        Xtr[all_cols], Xte[all_cols], ytr, yte, results)

    evaluate("Gradient boosting",
             GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                       max_depth=3, random_state=RANDOM_STATE),
             Xtr[all_cols], Xte[all_cols], ytr, yte, results)

    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    table.to_csv("results.csv", index=False)

    # ------------------------------------------------- what the model used
    importance = pd.Series(rf.feature_importances_, index=all_cols).sort_values()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    importance.plot.barh(ax=ax[0])
    ax[0].set(xlabel="random forest feature importance")

    ax[1].scatter(yte, rf_pred, s=12, alpha=0.5)
    lims = [y.min(), y.max()]
    ax[1].plot(lims, lims, "k--", lw=1)
    ax[1].set(xlabel="measured strength (MPa)", ylabel="predicted strength (MPa)")
    fig.tight_layout()
    fig.savefig("fig2_results.png", dpi=150)

    print("\nTop features:")
    print(importance.sort_values(ascending=False).head(4).round(3).to_string())
    print("\nSaved: fig1_exploration.png, fig2_results.png, results.csv")


if __name__ == "__main__":
    main()
