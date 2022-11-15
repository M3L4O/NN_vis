import imgaug.augmenters as iaa
import imgaug as ia
import pandas as pd
import numpy as np
import time
import cv2


def get_clean_df(filepath):
    """
    Função utilizada para remover as colunas não utilizadas e padronizar o DataFrame.

    Parametros
    ----------
        filepath : str
            Caminho até o arquivo contendo os metadados das instâncias.

    Returns
    -------
    DataFrame
        DataFrame contendo as informações necessárias para o treinamento.

    """
    df = pd.read_csv(filepath)
    df = df[["image_name", "benign_malignant"]]
    columns = ["image", "label"]
    df.columns = columns
    df["image"] = df["image"].apply(lambda x: f"{filepath}/{x}.jpg")

    return df


def get_augmenter():
    """
    Função utilizada para gerar um augmenter.

    Returns
    -------
    imgaug.augmenters.meta.SomeOf
        Função necessárias para ampliar a base de imagens.

    """
    ia.seed(int(time.time()))

    rotations = iaa.OneOf(
        [
            iaa.Rotate((-60, 60)),
            iaa.Rotate((-90, 90)),
            iaa.Rotate((-45, 45)),
            iaa.Rotate((-180, 180)),
            iaa.Rotate((-270, 270)),
        ]
    )

    zoom = iaa.OneOf(
        [
            iaa.Affine(scale=(1.5, 2.0)),
            iaa.Affine(scale=(1.0, 1.5)),
            iaa.Affine(scale=(2.0, 3.0)),
        ]
    )

    contrast = iaa.OneOf(
        [
            iaa.GammaContrast((0.5, 2.0)),
            iaa.LogContrast(gain=(0.6, 1.4)),
            iaa.LinearContrast((0.4, 1.6)),
        ]
    )

    geometric = iaa.SomeOf(
        2,
        [
            iaa.Affine(shear=(-16, 16)),
            iaa.ShearX((-20, 20)),
            iaa.ShearY((-20, 20)),
            iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        ],
    )

    noise = iaa.OneOf(
        [
            iaa.imgcorruptlike.GaussianNoise(severity=2),
            iaa.imgcorruptlike.ShotNoise(severity=2),
            iaa.imgcorruptlike.ImpulseNoise(severity=2),
            iaa.imgcorruptlike.SpeckleNoise(severity=2),
        ]
    )

    aug = iaa.SomeOf(
        (2, None),
        [
            rotations,
            zoom,
            contrast,
            geometric,
            noise,
        ],
    )

    return aug


def augment_generator(
    df, augmenter, dir_aug_images, img_size=(224, 224), new_images=10
):
    """
    Função utilizada para ampliar a base de imagens.

    Parametros
    ----------
        df : pd.DataFrame
            DataFrame contendo as instâncias a serem ampliadas.

        augmenter : imgaug.augmenters.meta.SomeOf
            Objeto responsável por aumentar a quantidade de versões de uma imagem.

        dir_aug_images : str
            Caminho até o diretório que irá armazenar as imagens geradas.

        img_size : tuple
            Tamanho da imagem

        new_images : int
            Número de novas imagens geradas a cada imagem.

    Returns
    -------
    DataFrame
        DataFrame contendo as informações das novas imagens geradas.

    """
    new_rows = []
    for i in range(len(df)):
        filename = df["image"].iloc[i].split(".")[2].split("/")[-1]
        label = df["label"].iloc[i]
        image = cv2.imread(df["image"].iloc[i])
        image = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size)
        aug_images = augmenter(images=[image] * new_images)

        for j in range(new_images):
            new_filepath = f"{dir_aug_images}/{filename}_{j}.jpg"
            cv2.imwrite(new_filepath, cv2.cvtColor(aug_images[j], cv2.COLOR_RGB2BGR))
            new_rows.append((new_filepath, label))

    aug_df = pd.DataFrame(new_rows, columns=["image", "label"])

    return aug_df


def main():

    filepath = "algumacoisa.csv"
    df = get_clean_df(filepath)
    benign_df, malignant_df = (
        df.loc[df["label"] == "benign"],
        df.loc[df["label"] == "malignant"],
    )
    dir_aug_images = "./aug_images/"
    aug_df = augment_generator(malignant_df, get_augmenter(), dir_aug_images,new_images=20)
    malignant_df= malignant_df.append(aug_df, ignore_index=True)
    samples = malignant_df.shape[0]
    benign_df = benign_df.samples(samples)

    final_df = malignant_df.append(benign_df, ignore_index=True)

    final_df.to_csv("metadados.csv")
    

if __name__ == "__main__":
    main()
