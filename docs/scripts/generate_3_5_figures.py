"""Generate figures for diploma section 3.5 (ensemble models)."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "3_5"
DATA = ROOT / "data" / "features" / "BTC-USDT_1h.parquet"
ARTIFACT_LGB = ROOT / "artifacts" / "BTC-USDT_1h" / "20260516T133447Z_533db42b" / "artifacts" / "lgb.pkl"
ARTIFACT_XGB = ROOT / "artifacts" / "BTC-USDT_1h" / "20260516T133447Z_533db42b" / "artifacts" / "xgb.pkl"

from feature_engineering.config import DIRECTION_FEATURE_COLUMNS


def _direction_labels(close: pd.Series, horizon: int = 12) -> pd.Series:
    return (close.shift(-horizon) > close).astype(int)


def _load_xy(sample_n: int = 8000):
    df = pd.read_parquet(DATA)
    feats = DIRECTION_FEATURE_COLUMNS
    y = _direction_labels(df["close"], horizon=12)
    mask = df[feats].notna().all(axis=1) & y.notna()
    X = df.loc[mask, feats].astype(float)
    y = y.loc[mask].astype(int)
    if len(X) > sample_n:
        X = X.iloc[-sample_n:]
        y = y.iloc[-sample_n:]
    split = int(len(X) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], feats


def plot_lgb_xgb():
    import lightgbm as lgb
    import xgboost as xgb

    X_tr, X_va, y_tr, y_va, feats = _load_xy()
    lgbm = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        objective="binary",
        verbosity=-1,
        class_weight="balanced",
        subsample=0.8,
        colsample_bytree=0.8,
    )
    lgbm.fit(
        X_tr,
        y_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        eval_names=["train", "valid"],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )

    xgbm = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        objective="binary:logistic",
        eval_metric="logloss",
        verbosity=0,
    )
    xgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # Feature importance
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    imp_l = pd.Series(lgbm.feature_importances_, index=feats).sort_values()
    imp_x = pd.Series(xgbm.feature_importances_, index=feats).sort_values()
    imp_l.plot(kind="barh", ax=axes[0], color="#2ca02c")
    axes[0].set_title("LightGBM — feature importance (gain)")
    imp_x.plot(kind="barh", ax=axes[1], color="#1f77b4")
    axes[1].set_title("XGBoost — feature importance")
    fig.tight_layout()
    fig.savefig(OUT / "tabular_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # LGB curves from evals
    res = lgbm.evals_result_
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, color in [("train", "#2ca02c"), ("valid", "#d62728")]:
        if name in res and "binary_logloss" in res[name]:
            ax.plot(res[name]["binary_logloss"], label=name, color=color)
    ax.set_xlabel("Итерация (дерево)")
    ax.set_ylabel("binary_logloss")
    ax.set_title("LightGBM — кривые обучения")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "lgb_train_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Predictions sample
    probs = lgbm.predict_proba(X_va)[:, 1]
    sample = pd.DataFrame(
        {"timestamp": X_va.index[:8], "prediction": np.round(probs[:8], 4)}
    )
    sample.to_csv(OUT / "lgb_prediction_sample.csv", index=False)

    # Comparison bar
    from sklearn.metrics import roc_auc_score

    p_l = lgbm.predict_proba(X_va)[:, 1]
    p_x = xgbm.predict_proba(X_va)[:, 1]
    auc_l = roc_auc_score(y_va, p_l)
    auc_x = roc_auc_score(y_va, p_x)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["LightGBM", "XGBoost"], [auc_l, auc_x], color=["#2ca02c", "#1f77b4"])
    ax.set_ylabel("ROC-AUC (validation)")
    ax.set_ylim(0.45, max(auc_l, auc_x) + 0.05)
    ax.set_title("Сравнение табличных моделей")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate([auc_l, auc_x]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "lgb_xgb_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return lgbm, xgbm, X_va, y_va


def plot_dl():
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    from models.trend.gru_model import GRUTrendModel
    from models.volatility.cnn_model import CNNVolatilityModel

    df = pd.read_parquet(DATA).iloc[-6000:].copy()
    feats = DIRECTION_FEATURE_COLUMNS
    y = _direction_labels(df["close"], 12)
    sub = df[feats + ["close"]].dropna()
    y = y.loc[sub.index]
    scaler = StandardScaler()
    sub[feats] = scaler.fit_transform(sub[feats])

    gru = GRUTrendModel(window_size=24, n_features=len(feats), lr=0.0005, dropout=0.2)
    gru.feature_cols = feats
    gru.build_model()

    # Save summary text
    lines = []
    gru.model.summary(print_fn=lines.append)
    (OUT / "gru_model_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    X, y_seq = gru.prepare_sequences(sub, y)
    split = int(len(X) * 0.85)
    hist_gru = gru.model.fit(
        X[:split],
        y_seq[:split],
        validation_data=(X[split:], y_seq[split:]),
        epochs=12,
        batch_size=64,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=0,
    )

    cnn = CNNVolatilityModel(window_size=24, n_features=len(feats), lr=0.0005, dropout=0.4)
    cnn.feature_cols = feats
    cnn.build_model()
    lines_c = []
    cnn.model.summary(print_fn=lines_c.append)
    (OUT / "cnn_model_summary.txt").write_text("\n".join(lines_c), encoding="utf-8")

    Xc, yc = cnn.prepare_sequences(sub, y)
    hist_cnn = cnn.model.fit(
        Xc[:split],
        yc[:split],
        validation_data=(Xc[split:], yc[split:]),
        epochs=12,
        batch_size=64,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=0,
    )

    # Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, hist, title in [
        (axes[0], hist_gru.history, "GRU"),
        (axes[1], hist_cnn.history, "CNN"),
    ]:
        ax.plot(hist["loss"], label="train")
        ax.plot(hist["val_loss"], label="validation")
        ax.set_title(f"{title} — loss")
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("binary_crossentropy")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "dl_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Sequence heatmap (first validation window)
    batch = np.asarray(X, dtype=float)
    if batch.ndim != 3:
        raise ValueError(f"Expected sequences (N, L, F), got {batch.shape}")
    window = batch[min(split, len(batch) - 1)]
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(window.T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("Шаг в окне (t−23 … t)")
    ax.set_title("Пример входного окна GRU (нормализованные признаки)")
    plt.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "gru_sequence_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CNN kernel visualization (first conv filters on first channel aggregate)
    conv_layer = next(
        ly for ly in cnn.model.layers if ly.__class__.__name__ == "Conv1D"
    )
    w = conv_layer.get_weights()[0]
    n_show = min(8, w.shape[2])
    fig, axes = plt.subplots(2, 4, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            ax.plot(w[:, 0, i], marker="o", markersize=3)
            ax.set_title(f"Фильтр {i+1}")
            ax.grid(True, alpha=0.2)
    fig.suptitle("CNN Conv1D — веса ядер (первый входной канал)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "cnn_kernel_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # GRU predictions sample
    probs = gru.model.predict(X[split : split + 8], verbose=0).flatten()
    pd.DataFrame(
        {"step": range(1, 9), "prediction": np.round(probs, 4)}
    ).to_csv(OUT / "gru_prediction_sample.csv", index=False)


def plot_artifact_importance():
    if not ARTIFACT_LGB.exists():
        return
    with open(ARTIFACT_LGB, "rb") as f:
        lgb_art = pickle.load(f)
    model = getattr(lgb_art, "model", lgb_art)
    if not hasattr(model, "feature_importances_"):
        return
    cols = getattr(lgb_art, "feature_cols", DIRECTION_FEATURE_COLUMNS)
    imp = pd.Series(model.feature_importances_, index=cols[: len(model.feature_importances_)]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot(kind="barh", ax=ax, color="#9467bd")
    ax.set_title("LightGBM (артефакт пайплайна) — importance")
    fig.tight_layout()
    fig.savefig(OUT / "lgb_artifact_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tf = None
    try:
        plot_lgb_xgb()
    except Exception as e:
        print("tabular plots skipped:", e)
    try:
        import tensorflow  # noqa: F401

        plot_dl()
    except Exception as e:
        print("dl plots skipped:", e)
    try:
        plot_artifact_importance()
    except Exception as e:
        print("artifact importance skipped:", e)
    print("Saved to", OUT)


if __name__ == "__main__":
    main()
