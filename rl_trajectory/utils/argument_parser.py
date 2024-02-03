
import argparse


def get_argument_parser():

    parser = argparse.ArgumentParser(description='GNFlow Mountain Car')
    parser.add_argument('--seed', metavar='s', type=int, default=42)
    parser.add_argument('--max_steps', metavar='n', type=int, default=2000)
    parser.add_argument('--penalty', metavar='p', type=float, default=0.0)
    parser.add_argument('--max_eps', type=float, default=9e-1)
    parser.add_argument('--min_eps', type=float, default=5e-3)
    parser.add_argument('--num_episodes', metavar='e', type=int, default=100)
    parser.add_argument('--alpha', metavar='a', type=float, default=1e-2)
    parser.add_argument('--gamma', metavar='g', type=float, default=1.0)
    parser.add_argument('--rho', metavar='r', type=float, default=0.9)
    parser.add_argument('--store_steps', action='store_true')
    parser.add_argument('--store_every', type=int, default=50)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--savelog', type=str, default='./experiments/')
    parser.add_argument('--grid_size', type=int, default=16)

    return parser
