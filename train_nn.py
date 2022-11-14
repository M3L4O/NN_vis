import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def get_data_iterator(df, img_size=(224, 224), batch_size=32, mode="binary"):
    """
    Função utilizada para gerar um DataFrameIterator para treinar o modelo

    Parametros
    ----------
        df : pd.DataFrame
            DataFrame contendo as instâncias para treinarem/testarem o modelo.

        img_size : tuple
            Tamanho da imagem.

        batch_size : int
            Tamanho do batch.

        mode : str
            Class mode utilizado no ImageDataGenerator.

    Returns
    -------
    DataFrameIterator
        Objeto que contém as informações das imagens para treino.

    DataFrameIterator
        Objeto que contém as informações das imagens para validação.

    DataFrameIterator
        Objeto que contém as informações das imagens para teste.

    """

    train_df, dummy_df = train_test_split(
        df, train_size=0.8, shuffle=True, random_state=123
    )
    test_df, validation_df = train_test_split(
        dummy_df, train_size=0.5, shuffle=False, random_state=123
    )

    gen = ImageDataGenerator(rescale=1.0 / 255)

    train_ds = gen.flow_from_dataframe(
        train_df,
        xcol="image",
        ycol="label",
        target_size=img_size,
        class_mode=mode,
        shuffle=True,
        batch_size=batch_size,
    )

    validation_ds = gen.flow_from_dataframe(
        validation_df,
        xcol="image",
        ycol="label",
        target_size=img_size,
        class_mode=mode,
        shuffle=True,
        batch_size=batch_size,
    )

    test_ds = gen.flow_from_dataframe(
        test_df,
        xcol="image",
        ycol="label",
        target_size=img_size,
        class_mode=mode,
        shuffle=True,
        batch_size=batch_size,
    )

    return train_ds, validation_ds, test_ds


def get_model(model_class):
    """
    Função utilizada para instânciar um modelo pré-treinado.

    Parametros
    ----------
        model_class :
            Classe a ser instânciada.

    Returns
    -------
    keras.Model
        Modelo pronto para ser treinado.
    """

    METRICS = [
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    inputs = tf.keras.Input(shape=(224, 224, 3))

    base_model = model_class(include_top=False, weights="imagenet", input_tensor=inputs)
    base_model.trainable = False

    gap = GlobalAveragePooling2D()(base_model.output)

    outputs = Dense(1, activation="sigmoid")(gap)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)

    return model


def train(
    model,
    model_name,
    model_dir,
    train_ds,
    validation_ds,
    epochs=1000,
    batch_size=32,
):
    """
    Função utilizada para treinar os modelos.

    Parametros
    ----------
        model : keras.Model
            Modelo no ponto de ser treinado.

        model_name : str
            Nome do modelo.

        model_dir : str
            Caminho até o diretório que irá armazenar os modelos treinados.

        train_ds: DataFrameIterator
            Iterator contendo as instâncias para o treinamento.

        validation_ds : DataFrameIterator
            Iterator contendo as instâncias para a validação.

        epochs : int
            Quantidade de épocas máximas por treino.

        batch_size: int
            Tamanho do batch.

    """

    train_ds.reset()
    validation_ds.reset()

    checkpoint = ModelCheckpoint(
        filepath=f"{model_dir}/{model_name}.h5",
        monitor="loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )

    early = EarlyStopping(
        monitor="loss",
        min_delta=0,
        patience=20,
        verbose=1,
        mode="auto",
    )

    model.fit(
        train_ds,
        steps_per_epoch=train_ds.n // batch_size,
        epochs=epochs,
        validation_data=validation_ds,
        validation_steps=validation_ds.n // batch_size,
        callbacks=[checkpoint, early],
    )

    model = tf.keras.models.load_model(f"{model_dir}/{model_name}.h5")

    train_ds.reset()
    validation_ds.reset()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    model.trainable = True
    checkpoint = ModelCheckpoint(
        filepath=f"{model_dir}/{model_name}_fine.h5",
        monitor="loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )

    model.fit(
        train_ds,
        steps_per_epoch=train_ds.n // batch_size,
        epochs=epochs,
        validation_data=validation_ds,
        validation_steps=validation_ds.n // batch_size,
        callbacks=[checkpoint, early],
    )


def main():
    models = []
    model_names = []
    df = pd.read_csv("metadados.csv")
    train_ds, validation_ds, test_ds = get_data_iterator(df)
    for model, name in zip(models, model_names):
        train(model, name, "./models_trained", train_ds, validation_ds)


if __name__ == "__main__":
    main()
