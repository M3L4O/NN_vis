from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import cv2


def get_clean_df(df, filepath):
    df = pd.read_csv(filepath)

    del df['patient_id']
    del df['sex']
    del df['age_approx']
    del df['anatom_site_general_challenge']
    del df['diagnosis']
    del df['target']

    columns = ["image", "label"]
    df.columns = columns
    df['image'] = df['image'].apply(lambda x: f"{filepath}/{x}.jpg")
    
    return df
