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
