import argparse

def get_dataset_args():
    parser = argparse.ArgumentParser(description='Get admin token, mission and variant')
    
    parser.add_argument('--env-name', type=str,
                        help='the name of the environment; must be registered with gymnasium')
    parser.add_argument('--num-episodes', type=int, nargs=1, default=10,
                        help='the number of episodes to collect for the environment')
    parser.add_argument('--include-timeout', type=bool, nargs=1, default=True,
                        help='whether to include episodes terminated by timeout / truncated')
    
    args = parser.parse_args()

    return vars(args)
