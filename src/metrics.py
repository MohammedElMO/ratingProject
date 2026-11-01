
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, confusion_matrix


def evaluate(y_true, y_pred, name=""):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"model": name,
            # every class weighs the same
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "mae": round(mean_absolute_error(y_true, y_pred), 4)}             # avg stars off


def confusion(y_true, y_pred):
    labels = [1, 2, 3, 4, 5]
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                        index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels])


def save_preds(df, y_pred, name, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": df["row"].values, "y_true": df["rating"].values, "y_pred": y_pred}) \
      .to_csv(Path(out_dir) / f"{name}.csv", index=False)
    # saved per model -> Phase F compares models on the SAME reviews (McNemar test)
