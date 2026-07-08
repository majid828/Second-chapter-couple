"""
Cross-validation for memory model evaluation.

The goal:

Evaluate how well a physical memory model
predicts unseen portions of H(t).


Uses blocked time-series validation.

Random shuffling is avoided because
transport memory is time dependent.

"""

import numpy as np

from sklearn.metrics import mean_squared_error





def cross_validate_memory_model(
        time,
        memory,
        fitting_function,
        parameters,
        folds=5
):
    """
    Perform blocked cross-validation.


    Parameters
    ----------

    time :
        time vector


    memory :
        recovered memory


    fitting_function :
        memory model function


    parameters :
        model parameters


    folds :
        number of sequential folds



    Returns
    -------

    dictionary:

        fold errors

        mean RMSE



    """

    time=np.asarray(
        time,
        dtype=float
    )


    memory=np.asarray(
        memory,
        dtype=float
    )


    if len(time)!=len(memory):

        raise ValueError(
            "Time and memory length mismatch."
        )


    n=len(time)


    fold_size=n//folds


    errors=[]



    for i in range(folds):


        start=i*fold_size


        if i==folds-1:

            end=n

        else:

            end=(i+1)*fold_size



        test_time=time[start:end]

        test_memory=memory[start:end]



        predicted=fitting_function(

            test_time,

            *parameters

        )


        rmse=np.sqrt(

            mean_squared_error(

                test_memory,

                predicted

            )

        )


        errors.append(
            rmse
        )



    return {

        "fold_errors":
        np.array(errors),


        "mean_RMSE":
        np.mean(errors)

    }
