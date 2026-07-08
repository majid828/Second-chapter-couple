"""
Clustering of transport memory functions.

Uses unsupervised learning to identify
memory classes.

"""

import numpy as np

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler





def cluster_memory_functions(
        memory_functions,
        n_clusters=3,
        random_state=42
):
    """
    Cluster memory functions.

    Parameters
    ----------

    memory_functions :

        shape:

        (n_sites,n_time)


    n_clusters :

        number of transport classes



    Returns
    -------

    dictionary:

        labels

        cluster centers

        model


    """

    X=np.asarray(
        memory_functions,
        dtype=float
    )


    scaler=StandardScaler()


    X_scaled=scaler.fit_transform(
        X
    )



    model=KMeans(

        n_clusters=n_clusters,

        random_state=random_state,

        n_init=20

    )


    labels=model.fit_predict(
        X_scaled
    )



    centers=scaler.inverse_transform(
        model.cluster_centers_
    )


    return {

        "labels":
        labels,


        "cluster_centers":
        centers,


        "model":
        model

    }
