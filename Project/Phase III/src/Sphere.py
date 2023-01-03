class sphere(x):
    """Sphere objective function.
    Has a global minimum at :code:`0` and with a search domain of
        :code:`[-inf, inf]`
    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`
    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    j = (x ** 2.0).sum(axis=1)

    return j