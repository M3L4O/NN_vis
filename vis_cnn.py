from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, model_from_json
import tensorflow as tf


from skimage.segmentation import felzenszwalb, slic, mark_boundaries
from lime import lime_image
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

from matplotlib.pyplot import imshow
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def get_hit_miss(image_gen, preds):
    """
    Função utilizada para pegar os erros e acertos das predições.

    Parametros
    ----------
        image_gen : DirectoryIterator, DataFrameIterator
            Iterador das imagens usadas na predição
        preds : np.ndarray
            Predições geradas pelo modelo
    Returns
    -------
    dict
        Um dicionário contendo todos os erros
    dict
        Outro dicionário contendo todos os acertos
    """

    labels = image_gen.labels
    file_paths = image_gen.filepaths
    map_class = {v: k for k, v in image_gen.class_indices.items()}
    misses_list = []
    misses_true_class = []
    misses_pred_class = []
    hit_list = []
    hit_pred_class = []

    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = labels[i]
        if pred_index != true_index:
            misses_list.append(file_paths[i])
            misses_true_class.append(map_class[true_index])
            misses_pred_class.append(map_class[pred_index])
        else:
            hit_list.append(file_paths[i])
            hit_pred_class.append(map_class[pred_index])

    return (
        {
            "missies": misses_list,
            "misses_pred": misses_pred_class,
            "misses_true": misses_true_class,
        },
        {
            "hits": hit_list,
            "hits_pred": hit_pred_class,
        },
    )


def get_data_iterator(df, img_size=(224, 224), mode="binary"):
    """
    Função utilizada para pegar os erros e acertos das predições.

    Parametros
    ----------
        df : pd.DataFrame
            DataFrame contendo as instâncias.
        img_size : tuple
            Tupla contendo as dimensões das imagens
        mode: str
            String contendo o class_mode para gerar o DataFrameIterator
    Returns
    -------
        DataFrameIterator
            Um iterador contendo tanto as instâncias.
    """
    gen = ImageDataGenerator(rescale=1.0 / 255)
    ds = gen.flow_from_dataframe(
        df,
        x_col="image",
        y_col="label",
        target_size=img_size,
        class_mode=mode,
        color_mode="rgb",
        shuffle=False,
        batch_size=32,
    )

    return ds


def make_lime_vis(
    vis_ds,
    predict_fn,
    explainer=None,
    per_batch=None,
    segmentation_fn=None,
    out_dir=None,
):
    """
    Função utilizada para pegar os erros e acertos das predições.

    Parametros
    ----------
        vis_ds: DirectoryIterator, DataFrameIterator
            Iterador das instâncias a serem explicadas.

        predict_fn: function
            Função utilizada para classificar as instâncias.
        
        explainer: LimeImageExplainer
            Objeto para gerar explicações.

        per_batch: int
            Quantidade de explicações a serem geradas por batch.

        segmentation_fn: function
            Função de segmentação a ser utilizada na explicação.
        
        out_dir:
            Diretório onde será armazenado as explicações.
    """
    map_class = {v: k for k, v in vis_ds.class_indices.items()}

    if explainer is None:
        explainer = lime_image.LimeImageExplainer()

    if per_batch is None:
        per_batch = vis_ds["batch_size"]

    if segmentation_fn is None:
        segmentation_fn = lambda x: felzenszwalb(x, scale=50, sigma=0.5, min_size=50)

    for j in range(len(vis_ds)):
        images, labels = vis_ds.next()
        for i in range(per_batch):
            explanation = explainer.explain_instance(
                images[i],
                predict_fn,
                top_labels=5,
                hide_color=0,
                segmentation_fn=segmentation_fn,
                num_samples=1000,
            )

            ind = explanation.top_labels[0]
            dict_heatmap = dict(explanation.local_exp[ind])
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

            fig, ax = plt.subplots(1, 2, figsize=(8, 8))

            ax[0].imshow(mark_boundaries(images[i], explanation.segments))
            img = ax[1].imshow(
                heatmap, cmap="RdBu", vmin=-heatmap.max(), vmax=heatmap.max()
            )
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.15)
            fig.colorbar(img, cax=cax)
            if len(labels[i]) == 2:
                pass
                # fig.suptitle(f"Expected:{} Predicted:{}")
            else:
                fig.suptitle(f"Predicted:{map_class[labels[i]]}")

            plt.tight_layout()
            fig.savefig(f"./{out_dir}/LIME_{vis_ds.filenames[j*vis_ds['batch_size']+i]}")
            plt.close(fig)
