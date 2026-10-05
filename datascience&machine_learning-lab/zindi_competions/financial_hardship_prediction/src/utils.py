def get_class_weights(y):
    """
    Compute balanced class weights.
    """

    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    # return dict(zip(classes, weights))
    return weights.tolist()