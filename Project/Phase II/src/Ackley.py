def ackley(x):
    """Ackley's objective function.
    Has a global minimum of `0` at :code:`f(0,0,...,0)` with a search
    domain of [-32, 32]
    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`
    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not np.logical_and(x >= -32, x <= 32).all():
        raise ValueError("Input for Ackley function must be within [-32, 32].")

    d = x.shape[1]
    j = (
        -20.0 * np.exp(-0.2 * np.sqrt((1 / d) * (x ** 2).sum(axis=1)))
        - np.exp((1 / float(d)) * np.cos(2 * np.pi * x).sum(axis=1))
        + 20.0
        + np.exp(1)
    )

    return j
