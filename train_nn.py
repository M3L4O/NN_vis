import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback

def save_history(history, filename, output_dir):
    """
    Função utilizada para gerar um DataFrameIterator para treinar o modelo

    Parametros
    ----------
        history : pd.DataFrame
            Histórico do treinamento do modelo.

        filename: str
            Nome da imagem a ser salva.

        output_dir: str
            Local onde as imagens serão salvas.

        mode : str
            Class mode utilizado no ImageDataGenerator.
    """

    plt.plot(history.history["val_loss"])
    plt.title("Validation loss history")
    plt.ylabel("Loss value")
    plt.xlabel("No. epoch")
    plt.savefig(f"{output_dir}\\{filename}_loss.png")

    plt.plot(history.history["val_accuracy"])
    plt.title("Validation accuracy history")
    plt.ylabel("Accuracy value (%)")
    plt.xlabel("No. epoch")
    plt.savefig(f"{output_dir}\\{filename}_acc.png")


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
        x_col="image",
        y_col="label",
        target_size=img_size,
        class_mode=mode,
        shuffle=True,
        batch_size=batch_size,
    )

    validation_ds = gen.flow_from_dataframe(
        validation_df,
        x_col="image",
        y_col="label",
        target_size=img_size,
        class_mode=mode,
        shuffle=True,
        batch_size=batch_size,
    )

    test_ds = gen.flow_from_dataframe(
        test_df,
        x_col="image",
        y_col="label",
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

    gap = GlobalAveragePooling2D()(base_model.output)

    outputs = Dense(1, activation="sigmoid")(gap)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)

    return model


def train(
    model_class,
    model_name,
    model_dir,
    history_dir,
    train_ds,
    validation_ds,
    epochs=120,
    batch_size=32,
):
    """
    Função utilizada para treinar os modelos.

    Parametros
    ----------
        model_class : keras.Model
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
        filepath=f"{model_dir}\\{model_name}.h5",
        monitor="loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )

    early = EarlyStopping(
        monitor="loss",
        min_delta=0,
        patience=10,
        verbose=1,
        mode="auto",
    )
    model = get_model(model_class)

    history = model.fit(
        train_ds,
        steps_per_epoch=train_ds.n // batch_size,
        epochs=epochs,
        validation_data=validation_ds,
        validation_steps=validation_ds.n // batch_size,
        callbacks=[checkpoint, early],
    )

    save_history(history, filename=f"{model_name}_transfer", output_dir=history_dir)

    history = model.fit(
        train_ds,
        steps_per_epoch=train_ds.n // batch_size,
        epochs=epochs,
        validation_data=validation_ds,
        validation_steps=validation_ds.n // batch_size,
        verbose=0,
        callbacks=[checkpoint, early,  TqdmCallback(verbose=1)],
    )
    save_history(history, filename=f"{model_name}_fine", output_dir=history_dir)


def main():
    models = [InceptionV3, Xception, VGG19]
    model_names = ["InceptionV3", "Xception", "VGG19"]
    df = pd.read_csv("metadados.csv")
    results = []
    train_ds, validation_ds, test_ds = get_data_iterator(df)
    for model, name in zip(models, model_names):
        train(model, name, "models_trained", "history", train_ds, validation_ds)
        model = tf.keras.models.load_model(f"models_trained\\{name}_fine.h5")
        test_ds.reset()
        score = model.evaluate(test_ds)
        results.append((name, score[0], score[1] * 100))

    results_df = pd.DataFrame(results, columns=["name", "loss", "accuracy"])
    results_df.to_csv("results.csv")


if __name__ == "__main__":
    main()
