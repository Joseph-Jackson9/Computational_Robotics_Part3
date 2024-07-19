from others.DeadReckoning import DeadReckoning
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str)
    parser.add_argument("--execution", type=str)
    parser.add_argument("--sensing", type=str)
    args = parser.parse_args()

    DeadReckoning(
        args.map,
        args.execution,
        args.sensing
    )
