from ray_model import *
from IPython import embed

def add_metadata(checkpoint_file, metadata):
    state = pickle.load(open(checkpoint_file, "rb"))

    f = open(metadata, 'r')
    metadata = f.read()
    f.close()

    # if not 'metadata' in state.keys():
    #     print('None')
    state['metadata'] = metadata

    f = open('test_result', 'wb')
    pickle.dump(state, f)
    f.close()
    

import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--checkpoint',help='checkpoint path')
    parser.add_argument('-m','--metadata',help='metadata path')
    args = parser.parse_args()

    add_metadata(args.checkpoint, args.metadata)

    print('Done')
