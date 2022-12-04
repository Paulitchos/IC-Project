import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher, Designer

# Import SwearmPackagePy
import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf

# Import class FireflyAlgorithm
from FireflyAlgorithm import FireflyAlgorithm 


def sphere(x):
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
    j = (x ** 2.0).sum(axis=0)

    return j

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

    d = x.shape[0]
    j = (
        -20.0 * np.exp(-0.2 * np.sqrt((1 / d) * (x ** 2).sum(axis=0)))
        - np.exp((1 / float(d)) * np.cos(2 * np.pi * x).sum(axis=0))
        + 20.0
        + np.exp(1)
    )

    return j

           
FA = FireflyAlgorithm()

# Set-up hyperparameters
#c1 - cognitive parameter
#c2 - social parameter
#c3 - inertica parameter
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)#A dimensao não é definida por nós, é a dimensao do problema
best = FA.run(function=sphere, dim=2, lb=-5, ub=5, max_evals=1000) # lb e ub representa os limites inferiores e superiores
print("\nFireFly [Class-Dim=2]: ",best)
print("\n")

alh = SwarmPackagePy.fa(10,tf.sphere_function, -5, 5, 2, 1000, 1, 1, 1, 0.1, 0, 0.1)
print("\nFireFly [Library-Dim=2]: ",alh.get_Gbest()) #Retorna a melhor posição do algoritmo
#print("Agentes: ",alh.get_agents()) #Retorna um histórico de todos os agentes do algoritmo 
print("\n")


# Perform optimizations sphere
cost,pos = optimizer.optimize(fx.sphere, iters=1000)

# Plot the cost
plot_cost_history(optimizer.cost_history)
plt.show()

# Plot the sphere function's mesh for better plots
m = Mesher(func=fx.sphere,
           limits=[(-1,1), (-1,1)])
# Adjust figure limits
d = Designer(limits=[(-1,1), (-1,1), (-0.1,1)],
             label=['x-axis', 'y-axis', 'z-axis'])

pos_history_3d = m.compute_history_3d(optimizer.pos_history) # preprocessing
animation3d = plot_surface(pos_history=pos_history_3d,
                           mesher=m, designer=d,
                           mark=(0,0,0))    
plt.show()
 
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=3, options=options)#A dimensao não é definida por nós, é a dimensao do problema
best = FA.run(function=sphere, dim=3, lb=-5, ub=5, max_evals=1000) # lb e ub representa os limites inferiors e superiores
print("\nFireFly [Class-Dim=3]: ",best)
print("\n")

alh = SwarmPackagePy.fa(10,tf.sphere_function, -5, 5, 3, 1000, 1, 1, 1, 0.1, 0, 0.1)
print("\nFireFly [Library-Dim=3]: ",alh.get_Gbest())
print("\n")
# Perform optimizationsphere
cost,pos = optimizer.optimize(fx.sphere, iters=1000)


# ================================================================== ACKLEY ================================================================== #

# Set-up hyperparameters
#c1 - cognitive parameter
#c2 - social parameter
#c3 - inertica parameter
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)#A dimensao não é definida por nós, é a dimensao do problema
best = FA.run(function=ackley, dim=2, lb=-5, ub=5, max_evals=1000) # lb e ub representa os limites inferiors e superiores
print("\nAckley [Class-Dim=2]: ",best)
print("\n")

alh = SwarmPackagePy.fa(10,tf.ackley_function, -5, 5, 2, 1000, 1, 1, 1, 0.1, 0, 0.1)
print("\nAckley [Library-Dim=2]: ",alh.get_Gbest()) #Retorna as cordenadas x,y
print("\n")
# Perform optimizationsphere
cost,pos = optimizer.optimize(fx.ackley, iters=1000)

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=3, options=options)#A dimensao não é definida por nós, é a dimensao do problema
best = FA.run(function=ackley, dim=3, lb=-5, ub=5, max_evals=1000) # lb e ub representa os limites inferiors e superiores
print("\nAckley [Class-Dim=3]: ",best)
print("\n")

alh = SwarmPackagePy.fa(10,tf.ackley_function, -5, 5, 3, 1000, 1, 1, 1, 0.1, 0, 0.1)
print("\nAckley [Library-Dim=3]: ",alh.get_Gbest())
print("\n")
# Perform optimizationsphere
cost,pos = optimizer.optimize(fx.ackley, iters=1000)






