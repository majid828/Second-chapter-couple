"""
Functional Principal Component Analysis.

Represents:

H_i(t)=mean(H)+sum(a_i psi_i)

Used for identifying dominant
transport memory variations.

"""

import numpy as np

from sklearn.decomposition import PCA





def perform_fpca(
        memory_functions,
        n_components=3
):
    """
    Perform FPCA using PCA on discretized
    functional data.


    Parameters
    ----------

    memory_functions :

        shape:

        (n_sites,n_time)



    n_components :

        number of modes



    Returns
    -------

    dictionary containing:

    scores

    eigenfunctions

    explained variance


    """

    X=np.asarray(
        memory_functions,
        dtype=float
    )


    if n_components>X.shape[0]:

        n_components=X.shape[0]



    pca=PCA(
        n_components=n_components
    )


    scores=pca.fit_transform(
        X
    )


    eigenfunctions=pca.components_


    return {

        "scores":
        scores,


        "eigenfunctions":
        eigenfunctions,


        "explained_variance":
        pca.explained_variance_ratio_,


        "model":
        pca

    }
