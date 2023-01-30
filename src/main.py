import os
import argparse
import logging
import time
import numpy as np
import wandb

import timm
import timm.optim
import timm.data
import torch
import torchvision

from torch.utils.data import DataLoader, ConcatDataset
# from torchsummary import summary
from torchvision.datasets import ImageFolder

from src.eff_net import EfficientNetv2SFC, EfficientNetv2SBase
# from src.cnn import SampleModel
from src.eval.evaluate import eval_fn
from src.training import train_fn
from src.data_augmentations import *


def main(data_dir,
         torch_model,
         num_epochs=15,
         batch_size=32,
         learning_rate=1e-3,
         weight_decay=0.9,
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer="RMSprop",
         data_augmentations=None,
         save_model_str=None,
         use_all_data_to_train=False,
         exp_name=''):
    """
    Training loop for configurableNet.
    :param torch_model: model that we are training
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param weight_decay: model optimizer weight_decay (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during training (torch.optim.Optimizer)
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :param save_model_str: path of saved models (str)
    :param use_all_data_to_train: indicator whether we use all the data for training (bool)
    :param exp_name: experiment name (str)
    :return:
    """

    # WandB initialization
    wandb.login()
    wandb.init(project=f'{exp_name}')

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data augmentation
    if data_augmentations is None:
        data_augmentations = transforms.ToTensor()
    elif isinstance(data_augmentations, list):
        #print(type(data_augmentations))
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    # TODO include augmentation in creating datasets
    # Create the dataset
    train_data = timm.data.create_dataset(name="", root=os.path.join(data_dir, "train"), split="train",
                                          is_training=True, transform=resize_to_224x224)
    val_data = timm.data.create_dataset(name="", root=os.path.join(data_dir, "val"), split="validation",
                                        transform=resize_to_224x224)
    test_data = timm.data.create_dataset(name="", root=os.path.join(data_dir, "test"), split="test")

    channels, img_height, img_width = train_data[0][0].shape

    # image size
    input_shape = (channels, img_height, img_width)

    # instantiate training criterion
    train_criterion = train_criterion().to(device)
    score = []

    # Create the dataset loader
    if device == torch.device("cpu"):
        if use_all_data_to_train:
            train_loader = DataLoader(dataset=ConcatDataset([train_data, val_data, test_data]),
                                      batch_size=batch_size,
                                      shuffle=True)
            logging.warning('Training with all the data (train, val and test).')
        else:
            train_loader = DataLoader(dataset=train_data,
                                      batch_size=batch_size,
                                      shuffle=True)
            val_loader = DataLoader(dataset=val_data,
                                    batch_size=batch_size,
                                    shuffle=False)
    else:
        if use_all_data_to_train:
            train_loader = timm.data.create_loader(dataset=ConcatDataset([train_data, val_data, test_data]),
                                                   input_size=input_shape, batch_size=batch_size, is_training=True)
            logging.warning('Training with all the data (train, val and test).')
        else:
            train_loader = timm.data.create_loader(dataset=train_data, input_size=input_shape, batch_size=batch_size,
                                                   is_training=True)
            val_loader = timm.data.create_loader(dataset=train_data, input_size=input_shape, batch_size=batch_size)

    # TODO define num_classes based on labels
    model = torch_model(num_classes=10).to(device)

    # instantiate optimizer
    optimizer = timm.optim.create_optimizer_v2(model_or_params=model, opt=model_optimizer, weight_decay=weight_decay,
                                               lr=learning_rate)

    # TODO add lr scheduler*

    # Info about the model being trained
    # You can find the number of learnable parameters in the model here
    logging.info('Model being trained:')
    summary(model, input_shape,
            device='cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        train_score, train_loss = train_fn(model, optimizer, train_criterion, train_loader, device)
        logging.info('Train accuracy: %f', train_score)

        if not use_all_data_to_train:
            test_score = eval_fn(model, val_loader, device)
            logging.info('Validation accuracy: %f', test_score)
            score.append(test_score)
            wandb.log({'train_accuracy': train_score, 'train_loss': train_loss, 'test_accuracy': test_score,
                       'epoch': epoch})
        else:
            wandb.log({'train_accuracy': train_score, 'train_loss': train_loss, 'epoch': epoch})

    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        model_save_dir = os.path.join(os.getcwd(), save_model_str)

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        save_model_str = os.path.join(model_save_dir, exp_name + '_model_' + str(int(time.time())))
        torch.save(model.state_dict(), save_model_str)

    if not use_all_data_to_train:
        logging.info('Accuracy at each epoch: ' + str(score))
        logging.info('Mean of accuracies across all epochs: ' + str(100 * np.mean(score)) + '%')
        logging.info('Accuracy of model at final epoch: ' + str(100 * score[-1]) + '%')

    return score[-1]


if __name__ == '__main__':
    """
    This is just an example of a training pipeline.

    Feel free to add or remove more arguments, change default values or hardcode parameters to use.
    """
    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss}  # Feel free to add more

    cmdline_parser = argparse.ArgumentParser('DL WS20/21 Competition')

    cmdline_parser.add_argument('-m', '--model',
                                default='SampleModel',
                                help='Class name of model to train',
                                type=str)
    cmdline_parser.add_argument('-e', '--epochs',
                                default=100,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=32,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'dataset'),
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-l', '--learning_rate',
                                default=1e-3,
                                help='Optimizer learning rate',
                                type=float)
    cmdline_parser.add_argument('-w', '--weight_decay',
                                default=9e-1,
                                help='Optimizer weight decay',
                                type=float)
    cmdline_parser.add_argument('-L', '--training_loss',
                                default='cross_entropy',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='RMSprop',
                                help='Which optimizer to use during training',
                                type=str)
    cmdline_parser.add_argument('-p', '--model_path',
                                default='models',
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-n', '--exp_name',
                                default='default',
                                help='Name of this experiment',
                                type=str)
    cmdline_parser.add_argument('-d', '--data-augmentation',
                                default='resize_to_224x224',
                                help='Data augmentation to apply to data before passing to the model.'
                                     + 'Must be available in data_augmentations.py')
    cmdline_parser.add_argument('-a', '--use-all-data-to-train',
                                action='store_true',
                                help='Uses the train, validation, and test data to train the model if enabled.')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    main(
        data_dir=args.data_dir,
        torch_model=eval(args.model),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=args.optimizer,
        data_augmentations=eval(args.data_augmentation),  # Check data_augmentations.py for sample augmentations
        save_model_str=args.model_path,
        exp_name=args.exp_name,
        use_all_data_to_train=args.use_all_data_to_train
    )
