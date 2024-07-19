
from others.AnimateParticleFilter import AnimateParticleFilter
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str)
    parser.add_argument("--sensing", type=str)
    parser.add_argument("--num_particles", type=int)
    parser.add_argument("--estimates", type=str)
    args = parser.parse_args()

    np.random.seed(42)

    AnimateParticleFilter(
        args.map,
        args.sensing,
        args.num_particles,
        args.estimates
    )
