from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


def main():
    kfold = KFold(n_splits=10, shuffle=True)
    df = pd.read_csv("./ISIC_2020_Training_GroundTruth.csv")
    df = df[["image_name", "target"]]
    ben_df = df.loc[df["target"] == 0].sample(584)
    mal_df = df.loc[df["target"] == 1]
    df_final = pd.concat([ben_df, mal_df], ignore_index=True)

    for train, test in kfold.split(df_final):
        train_df = df_final.iloc[train]
        print(train_df["target"].value_counts())


if __name__ == "__main__":
    main()
