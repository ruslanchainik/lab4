import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

from nn import SingleLayerNet, Perceptron, train, bce_loss

ISU = 408607
PLOTS = "plots"
os.makedirs(PLOTS, exist_ok=True)


def split_data(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=ISU, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=ISU, stratify=y_tmp)
    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr)
    X_val = sc.transform(X_val)
    X_te = sc.transform(X_te)
    y_tr = y_tr.reshape(-1, 1).astype(np.float64)
    y_val = y_val.reshape(-1, 1).astype(np.float64)
    y_te = y_te.reshape(-1, 1).astype(np.float64)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def pick_threshold(model, X_val, y_val):
    p = model.predict_proba(X_val).ravel()
    yv = y_val.ravel()
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(p.min(), p.max(), 101):
        pred = (p >= t).astype(int)
        f = f1_score(yv, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t)


def evaluate(model, X, y, threshold=0.5):
    p = model.predict_proba(X)
    pred = (p >= threshold).astype(int)
    return {
        "loss": float(bce_loss(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion": confusion_matrix(y, pred).tolist(),
        "threshold": threshold,
        "proba": p.ravel(),
        "true": y.ravel(),
    }


def grid_search(builder, X_tr, y_tr, X_val, y_val, grid, seed=ISU):
    best = None
    for lr in grid["lr"]:
        for bs in grid["batch_size"]:
            for hp in grid.get("hidden", [None]):
                rng = np.random.default_rng(seed)
                model = builder(rng, hp)
                hist = train(model, X_tr, y_tr, X_val, y_val,
                             lr=lr, epochs=400, batch_size=bs,
                             seed=seed, patience=60)
                val_loss = min(hist["val_loss"])
                cfg = {"lr": lr, "batch_size": bs, "hidden": hp,
                       "val_loss": val_loss}
                if best is None or val_loss < best["val_loss"]:
                    best = {**cfg, "model": model, "history": hist}
    return best


def build_decision_grid(model, X):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if X.shape[1] > 2:
        pad = np.zeros((grid.shape[0], X.shape[1] - 2))
        grid = np.c_[grid, pad]
    zz = model.predict_proba(grid).reshape(xx.shape)
    return xx, yy, zz


def plot_history(histories, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in histories.items():
        axes[0].plot(h["train_loss"], label=f"{name} train")
        axes[0].plot(h["val_loss"], "--", label=f"{name} val")
        axes[1].plot(h["train_acc"], label=f"{name} train")
        axes[1].plot(h["val_acc"], "--", label=f"{name} val")
    axes[0].set_title(f"{title}: cross-entropy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_title(f"{title}: accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close(fig)


def plot_decision(models, X, y, title, fname):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
    if len(models) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        xx, yy, zz = build_decision_grid(model, X)
        ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)
        ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.0)
        ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="RdBu",
                   edgecolors="k", s=20)
        ax.set_title(f"{title}: {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close(fig)


def plot_roc(results, title, fname):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["true"], r["proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC: {title}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close(fig)


def plot_metrics_bar(all_results, fname):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = []
    values = {m: [] for m in metrics}
    for ds_name, res in all_results.items():
        for mdl_name, r in res.items():
            labels.append(f"{ds_name}\n{mdl_name}")
            for m in metrics:
                values[m].append(r[m])
    x = np.arange(len(labels))
    w = 0.16
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 2) * w, values[m], w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Метрики на тестовой выборке")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close(fig)


def plot_batch_lr_study(builder, X_tr, y_tr, X_val, y_val, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for bs in [8, 32, 128]:
        rng = np.random.default_rng(ISU)
        model = builder(rng)
        h = train(model, X_tr, y_tr, X_val, y_val, lr=1e-2,
                  epochs=200, batch_size=bs, seed=ISU, patience=200)
        axes[0].plot(h["val_loss"], label=f"batch={bs}")
    axes[0].set_title(f"{title}: влияние batch_size (lr=0.01)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for lr in [1e-1, 1e-2, 1e-3]:
        rng = np.random.default_rng(ISU)
        model = builder(rng)
        h = train(model, X_tr, y_tr, X_val, y_val, lr=lr,
                  epochs=200, batch_size=32, seed=ISU, patience=200)
        axes[1].plot(h["val_loss"], label=f"lr={lr}")
    axes[1].set_title(f"{title}: влияние lr (batch=32)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname), dpi=120)
    plt.close(fig)


def run_dataset(name, X, y, hidden_sizes):
    print(f"\n=== {name} ===")
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)

    def single_builder(rng, _=None):
        return SingleLayerNet(X_tr.shape[1], rng)

    def perc_builder(rng, h):
        return Perceptron(X_tr.shape[1], h, rng)

    grid_single = {"lr": [1e-1, 5e-2, 1e-2], "batch_size": [8, 32, 64]}
    grid_perc = {"lr": [1e-1, 5e-2, 1e-2],
                 "batch_size": [8, 32, 64],
                 "hidden": hidden_sizes}

    best_single = grid_search(single_builder, X_tr, y_tr, X_val, y_val, grid_single)
    best_perc = grid_search(perc_builder, X_tr, y_tr, X_val, y_val, grid_perc)

    print(f"Single-layer best: lr={best_single['lr']} bs={best_single['batch_size']} "
          f"val_loss={best_single['val_loss']:.4f}")
    print(f"Perceptron best: lr={best_perc['lr']} bs={best_perc['batch_size']} "
          f"hidden={best_perc['hidden']} val_loss={best_perc['val_loss']:.4f}")

    t_single = pick_threshold(best_single["model"], X_val, y_val)
    t_perc = pick_threshold(best_perc["model"], X_val, y_val)
    res_single = evaluate(best_single["model"], X_te, y_te, t_single)
    res_perc = evaluate(best_perc["model"], X_te, y_te, t_perc)
    print(f"Thresholds: single={t_single:.3f} perceptron={t_perc:.3f}")

    tag = name.lower().replace(" ", "_")
    plot_history({"single-layer": best_single["history"],
                  "perceptron": best_perc["history"]},
                 name, f"history_{tag}.png")
    plot_roc({"single-layer": res_single, "perceptron": res_perc},
             name, f"roc_{tag}.png")

    if X.shape[1] == 2:
        plot_decision({"single-layer": best_single["model"],
                       "perceptron": best_perc["model"]},
                      X_te, y_te, name, f"decision_{tag}.png")

    plot_batch_lr_study(single_builder, X_tr, y_tr, X_val, y_val,
                        f"{name} (single-layer)", f"sensitivity_{tag}.png")

    return {
        "single-layer": {**{k: res_single[k] for k in
                            ["loss", "accuracy", "precision", "recall", "f1",
                             "roc_auc", "confusion", "threshold"]},
                         "config": {k: best_single[k] for k in
                                    ["lr", "batch_size", "hidden", "val_loss"]},
                         **{"true": res_single["true"],
                            "proba": res_single["proba"]}},
        "perceptron": {**{k: res_perc[k] for k in
                          ["loss", "accuracy", "precision", "recall", "f1",
                           "roc_auc", "confusion", "threshold"]},
                       "config": {k: best_perc[k] for k in
                                  ["lr", "batch_size", "hidden", "val_loss"]},
                       **{"true": res_perc["true"],
                          "proba": res_perc["proba"]}},
    }


def main():
    X1, y1 = make_moons(n_samples=400, noise=0.15, random_state=ISU)
    X2, y2 = make_classification(n_samples=200, n_features=5, n_redundant=2,
                                 random_state=ISU, n_informative=2,
                                 n_clusters_per_class=2, n_classes=2)
    results = {}
    results["moons"] = run_dataset("moons", X1, y1, hidden_sizes=[4, 8, 16, 32])
    results["classification"] = run_dataset("classification", X2, y2,
                                            hidden_sizes=[4, 8, 16, 32])

    plot_metrics_bar(results, "metrics_bar.png")

    report = {}
    for ds, res in results.items():
        report[ds] = {}
        for k, r in res.items():
            report[ds][k] = {kk: vv for kk, vv in r.items()
                            if kk not in ("true", "proba")}
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nГотово. Результаты в results.json, графики в plots/")


if __name__ == "__main__":
    main()
