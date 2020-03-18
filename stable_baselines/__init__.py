from stable_baselines.ppo2 import PPO2

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    pass
del mpi4py

__version__ = "2.9.0a0"
