import os
import torch
import logging
import argparse

from eval.evaluate import eval_model
from cnn import *
from data_augmentations import resize_to_224x224

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DL WS22/23 Competition')

    parser.add_argument('-m', '--model',
                        default='SampleModel',
                        help='Name of the Model class present in cnn.py (Eg: SampleModel)',
                        type=str)

    parser.add_argument('-p', '--saved_model_file',
                        default='sample_model',
                        help='Name of file inside models directory which contains the saved weights of the trained '
                             'model',
                        type=str)

    parser.add_argument('-D', '--test_data_dir',
                        default=os.path.join(os.getcwd(), 'dataset', 'test'),
                        help='Path to folder with the test data to evaluate the model on.'
                        + 'The organizers will populate the test folder with the unseen dataset to evaluate your model.'
                        )

    parser.add_argument('-d', '--data_augmentations',
                        default='resize_to_224x224',
                        help='Data augmentation to apply to data before passing it to the model. '
                        + 'Must be available in data_augmentations.py')

    parser.add_argument('-v', '--verbose',
                        default='INFO',
                        choices=['INFO', 'DEBUG'],
                        help='verbosity')

    args, unknowns = parser.parse_known_args()

    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    model_class = eval(args.model)
    eval_model(
        model=model_class(),
        saved_model_file=args.saved_model_file,
        test_data_dir=args.test_data_dir,
        data_augmentations=eval(args.data_augmentations)
    )
