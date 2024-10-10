from argparse import ArgumentParser

def get_model_parser():
    parser = ArgumentParser(description="Demo of argparse")
    parser.add_argument('--dir', default='/home/xc425/project/models')
    parser.add_argument('--runid', required=True)
    parser.add_argument('--step', type=int, default=-1)
    
    return parser

def get_dataset_parser():
    parser = ArgumentParser(description="Dataset Parser")
    parser.add_argument('--n_dim', required=True, type=int)
    parser.add_argument('--train_size', type=int, default=1500000)
    
    return parser
