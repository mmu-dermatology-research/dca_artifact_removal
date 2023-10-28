"""
Module for plotting TSNE for the balanced dataset.

Much of the logic here was interpolated from:
https://nextjournal.com/ml4a/image-t-sne

with some minor tweaking for this use case.
"""
import re
from typing import Tuple, List
from glob import glob
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_all_data(
    data_filepath: str,
    kernel: Tuple[int, int],
    glob_pattern: str = "/**/clean/**"
) -> Tuple[Tuple[np.ndarray], Tuple[str], Tuple[str]]:
    """
    Retrieve all of the image data from the balanced DCA set.

    :param data_filepath: the root of the data subdir
    :type data_filepath: str
    :param kernel: the kernel size used to transform images
    :type kernel: Tuple[int, int]
    :param glob_pattern: the pattern to recursively load images
    :type glob_pattern: str

    :returns: (flattened image data, class, filepath)
    :rtype: Tuple[Tuple, Tuple, Tuple]
    """
    data = [
        (_img_transform(file, kernel), _get_class(file), file)
        for file in glob(data_filepath + glob_pattern, recursive=True)
        if file[-4:] in [".jpg", ".png"]
    ]
    x, y, z = zip(*data)
    return x, y, z


def _img_transform(filepath: str, kernel: Tuple[int, int]) -> np.ndarray:
    """
    Transform an image from a filepath into a flattened vector.

    :param filepath: the full filepath of the image
    :type filepath: str
    :param kernel: the size that the image should be transformed into.
    :type kernel: Tuple[int, int]

    :returns: the flattened image vector
    :rtype: np.ndarray
    """
    return cv2.resize(
        cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_RGB2GRAY),
        kernel
    ).flatten()


def _get_class(filepath: str) -> str:
    """
    Get the class associated to the image.

    This is extracted from the name of the images parent directory.

    :param filepath: the full filepath of the image
    :type filepath: str

    :returns: the classification associated to the image
    :rtype: str
    """
    return re.split(r'/', filepath)[-2]


def get_pca_features(data: Tuple[np.ndarray], n_components: int) -> np.ndarray:
    """
    Fit the PCA model and apply dimensionality reduction to the data.

    :param data: the data to fit the PCA to
    :type data: Tuple[np.ndarray]

    :returns: the transformed data
    :rtype: np.ndarray
    """
    return PCA(n_components=n_components).fit_transform(np.array(data))


def get_tsne_embeddings(
    data: np.array,
    n_components: int,
    learning_rate: int,
    angle: float,
    verbose: int
) -> np.ndarray:
    """
    Fit the TSNE model and get embeddings from transform.

    :param data: the data to embed
    :type data: np.array
    :param n_components: the dimensionality of the embedded space
    :type n_components: int
    :param learning_rate: the learning rate to apply to the model
    :type learning_rate: float
    :param angle:
    :type angle: float
    :param verbose:
    :type verbose: int
    """
    return TSNE(
        n_components=n_components,
        learning_rate=learning_rate,
        angle=angle,
        verbose=verbose
    ).fit_transform(data)


def get_class_encodings(classes: Tuple[str]) -> List[int]:
    """
    Get the encodings for classes
    * binary dca dataset only

    :param classes: the classes to encode
    :type classes: List[str]

    :returns: the corresponding encoding for classes
    :rtype: List[int]
    """
    return [get_encoding_from_class_name(x) for x in classes]


def get_encoding_from_class_name(class_name: str) -> int:
    """
    Get the encoding that corresponds to a class name
    * binary dca dataset only

    :param class_name: the class name to determine the encoding of
    :type class_name: str

    :returns: the corresponding encoding
    :rtype: int
    """
    return 1 if class_name == "mel" else 0


def get_class_name_from_encoding(encoding: int) -> str:
    """
    Get the class name that corresponds to an encoding
    * binary dca dataset only

    :param encoding: the encoding to determine the class of
    :type encoding: int

    :returns: the corresponding class name
    :rtype: str
    """
    return "mel" if encoding == 1 else "oth"


def normalize_2d_embeddings(
    embeddings: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize the 2d embeddings produced by the TSNE model between
    0 and 1.

    :param embeddings: the embeddings produced by the tsne model
    :type embeddings: np.ndarray

    :returns: the normalised embeddings as 2 separate arrays (x and y)
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    tsne_x, tsne_y = embeddings[:, 0], embeddings[:, 1]
    return (
        (tsne_x-np.min(tsne_x)) / (np.max(tsne_x) - np.min(tsne_x)),
        (tsne_y-np.min(tsne_y)) / (np.max(tsne_y) - np.min(tsne_y))
    )


def tsne_duo_plot(
    plot_width: int,
    plot_height: int,
    max_dim: int,
    filepaths: Tuple[str],
    classes: Tuple[str],
    tsne_encodings: Tuple[np.ndarray, np.ndarray],
    figsize: Tuple[int, int],
    savepath: str
):
    """
    """
    full_image = Image.new('RGBA', (plot_width, plot_height))
    for img, x, y in zip(filepaths, tsne_encodings[0], tsne_encodings[1]):
        n = Image.open(img)
        rs = max(1, n.width/max_dim, n.height/max_dim)
        n = n.resize((int(n.width/rs), int(n.height/rs)), Image.LANCZOS)
        full_image.paste(
            n,
            (int((plot_width-max_dim)*x), int((plot_height-max_dim)*y)),
            mask=n.convert('RGBA')
        )

    colours = ListedColormap(['green', 'red'])
    class_names = ["oth", "mel"]
    classes_enc = get_class_encodings(classes)

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes[0].imshow(full_image)
    scatter = axes[1].scatter(
        tsne_encodings[0],
        tsne_encodings[1],
        c=classes_enc,
        s=8,
        cmap=colours
    )

    for ax in axes:
        ax.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            bottom=False
        )

    axes[1].invert_yaxis()
    axes[1].legend(
        handles=scatter.legend_elements()[0],
        fontsize="20",
        labels=class_names
    )

    # fig.savefig(savepath)


def tsne_duo_plot_2(
    plot_width: int,
    plot_height: int,
    max_dim: int,
    filepaths: List[Tuple[str]],
    classes: List[Tuple[str]],
    class_names: List[str],
    tsne_encodings: List[Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int],
    savepath: str
):
    """
    """
    full_image = Image.new('RGBA', (plot_width, plot_height))
    for i in range(len(classes)):
        for img, x, y in zip(filepaths[i], tsne_encodings[i][0], tsne_encodings[i][1]):
            n = Image.open(img)
            rs = max(1, n.width/max_dim, n.height/max_dim)
            n = n.resize((int(n.width/rs), int(n.height/rs)), Image.LANCZOS)
            full_image.paste(
                n,
                (int((plot_width-max_dim)*x), int((plot_height-max_dim)*y)),
                mask=n.convert('RGBA')
            )

    colours = ['g', 'r', 'b']

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes[0].imshow(full_image)
    for i in range(len(classes)):
        scatter = axes[1].scatter(
            tsne_encodings[i][0],
            tsne_encodings[i][1],
            c=colours[i],
            s=8,
            label=class_names[i]
        )

    for ax in axes:
        ax.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            bottom=False
        )

    axes[1].invert_yaxis()
    axes[1].legend(fontsize="20")

    fig.savefig(savepath)
