from sklearn.model_selection import train_test_split
import re
import csv
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/mohamedadaly/LABR/master/data/"
FILES = ["reviews.tsv", "5class-unbalanced-train.txt",
         "5class-unbalanced-test.txt"]
COLS = ["rating", "review_id", "user_id", "book_id", "text"]
DIAC = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]')


def download_labr(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dest = data_dir / f
        if not dest.exists():
            print("downloading", f)
            urllib.request.urlretrieve(BASE_URL + f, dest)
        else:
            print("already here:", f)


def load_reviews(data_dir):
    df = pd.read_csv(Path(data_dir) / "reviews.tsv", sep="\t", header=None, names=COLS,
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_split_ids(data_dir):
    d = Path(data_dir)
    return (np.loadtxt(d / "5class-unbalanced-train.txt", dtype=int),
            np.loadtxt(d / "5class-unbalanced-test.txt", dtype=int))


def normalize(t):
    t = DIAC.sub('', str(t)).replace(
        '\u0640', '')        # diacritics + tatweel
    t = re.sub('[إأآا]', 'ا', t).replace('ى', 'ي').replace('ة', 'ه')
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)                   # رااااائع -> راائع
    return re.sub(r'\s+', ' ', t).strip()


def make_splits(df, tr_ids, te_ids, val_size=0.1, seed=42):
    # remember each review's original position (traceability)
    df = df.assign(row=np.arange(len(df)))
    train, test = df.iloc[tr_ids], df.iloc[te_ids]
    log = {"train_official": len(train)}

    # 1) leakage: drop train rows whose normalized text also exists in test
    leak = train.norm.isin(set(test.norm))
    train = train[~leak]
    log["removed_leak"] = int(leak.sum())

    # 2) duplicates: count how many times each (text, rating) pair appears
    cnt = train.groupby(["norm", "rating"]).size().rename("c").reset_index()
    cnt["is_max"] = cnt.c == cnt.groupby("norm").c.transform(
        "max")   # is this rating the most frequent for this text?
    # >1 means a tie between ratings
    n_max = cnt[cnt.is_max].groupby("norm").size()
    winners = cnt[cnt.is_max & cnt.norm.isin(n_max[n_max == 1].index)][[
        "norm", "rating"]]  # texts with a clear majority
    before = len(train)
    train = train.merge(winners, on=["norm", "rating"]).drop_duplicates("norm")
    # merge keeps only rows whose rating = the majority (tied texts vanish); drop_duplicates keeps 1 copy
    log["removed_dups_or_ties"] = before - len(train)

    # 3) validation carved from the cleaned train, same class proportions (stratify)
    train, val = train_test_split(
        train, test_size=val_size, stratify=train.rating, random_state=seed)
    log.update(train=len(train), val=len(val), test=len(test))
    return train, val, test, log


def save_splits(splits, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, d in splits.items():
        d.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")


def load_splits(data_dir):
    d = Path(data_dir) / "splits"
    return {n: pd.read_csv(d / f"{n}.csv", encoding="utf-8",
                           keep_default_na=False)            # stops pandas turning a review like "NA" into NaN
            for n in ["train", "val", "test"]}
