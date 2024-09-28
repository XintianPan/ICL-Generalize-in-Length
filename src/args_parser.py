from argparse import ArgumentParser

def get_model_parser():
    parser = ArgumentParser(description="Demo of argparse")
    parser.add_argument('--dir', default='/home/xc425/project/models')
    parser.add_argument('--runid', required=True)
    
    return parser
