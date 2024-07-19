
from others.AnimateKalmanFilter import AnimateKalmanFilter
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str)
    parser.add_argument("--sensing", type=str)
    parser.add_argument("--estimates", type=str)
    args = parser.parse_args()

    np.random.seed(42)

    AnimateKalmanFilter(
        args.map,
        args.sensing,
        args.estimates
    )
