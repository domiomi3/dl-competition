import argparse
import os
import logging
import time

import numpy as np
import wandb
import torch
import neps

from functools import partial

from torch.utils.data import DataLoader, ConcatDataset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from cnn import *
from eval.evaluate import eval_fn
from training import train_fn
from data_augmentations import *
from bo_pipeline import get_pipeline_space
from utilities import set_seed


def main(data_dir,
         torch_model,
         bo=False,
         bo_iter=15,
         num_epochs=40,
         batch_size=16,
         learning_rate=1e-3,
         weight_decay=0.01,
         dropout=1e-5,
         momentum=0.9,
         cutmix_prob=0.3,
         beta=0.1,
         model_optimizer=torch.optim.RMSprop,
         train_criterion=torch.nn.CrossEntropyLoss,
         data_augmentations=None,
         save_model_str=None,
         use_all_data_to_train=False,
         log_wandb=False,
         exp_name=''):
    """
    Main function running HPO or single training.
    :param torch_model: model that we are training
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param weight_decay: model optimizer weight_decay (float)
    :param momentum: model optimizer momentum (float)
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
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    run_pipeline = partial(train_and_evaluate,
                           data_dir=data_dir,
                           torch_model=torch_model,
                           num_epochs=num_epochs,
                           batch_size=batch_size,
                           train_criterion=train_criterion,
                           model_optimizer=model_optimizer,
                           data_augmentations=data_augmentations,
                           save_model_str=save_model_str,
                           exp_name=exp_name,
                           use_all_data_to_train=use_all_data_to_train,
                           log_wandb=log_wandb
                           )

    if bo:
        run_bo(run_pipeline=run_pipeline, bo_iter=bo_iter)
    else:
        run_pipeline(learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout, momentum=momentum,
                     cutmix_prob=cutmix_prob, beta=beta)


def train_and_evaluate(data_dir,
                       torch_model,
                       num_epochs=40,
                       batch_size=16,
                       learning_rate=1e-3,
                       weight_decay=0.01,
                       dropout=1e-5,
                       momentum=0.9,
                       cutmix_prob=0.3,
                       beta=0.1,
                       model_optimizer=torch.optim.RMSprop,
                       train_criterion=torch.nn.CrossEntropyLoss,
                       data_augmentations=None,
                       save_model_str=None,
                       use_all_data_to_train=False,
                       log_wandb=False,
                       exp_name=''):
    """
    Training loop.
    :param torch_model: model that we are training
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param weight_decay: model optimizer weight_decay (float)
    :param momentum: model optimizer momentum (float)
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

    # Run ID for model retrieval and WandB plotting
    run_id = f'lr={learning_rate:.5f}_wd={weight_decay:.5f}_m={momentum:.5f}_dr={dropout:.5f}_cmix_prob={cutmix_prob:.5f}_beta' \
             f'={beta:.5f}'

    # WandB initialization if -wb flag enabled
    if log_wandb:
        wandb.login()
        wandb.init(project=f'{exp_name}', id=run_id, reinit=True)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data augmentation
    if data_augmentations is None:
        data_augmentations = transforms.ToTensor()
    elif isinstance(data_augmentations, list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    # Load the dataset
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=data_augmentations)
    val_data = ImageFolder(os.path.join(data_dir, 'val'), transform=data_augmentations)
    test_data = ImageFolder(os.path.join(data_dir, 'test'), transform=data_augmentations)

    channels, img_height, img_width = train_data[0][0].shape

    # image size
    input_shape = (channels, img_height, img_width)

    # instantiate training criterion
    train_criterion = train_criterion().to(device)
    score = []

    # Create the dataset loader
    if use_all_data_to_train:
        train_loader = DataLoader(dataset=ConcatDataset([train_data, val_data, test_data]),
                                  batch_size=batch_size,
                                  shuffle=True)
        logging.warning('Training with all the data (train, val and test).')
    else:
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    # Model initialization
    model = torch_model(num_classes=len(train_data.classes), dropout=dropout).to(device)

    # Optimizer initialization
    optimizer = model_optimizer(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                momentum=momentum)

    # Info about the model being trained
    # You can find the number of learnable parameters in the model here
    logging.info('Model being trained:')
    # summary(model, input_shape,
    #         device='cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        train_score, train_loss = train_fn(model, optimizer, train_criterion, train_loader, device, cutmix_prob, beta)
        logging.info('Train accuracy: %f', train_score)

        if not use_all_data_to_train:
            test_score = eval_fn(model, val_loader, device)
            logging.info('Validation accuracy: %f', test_score)
            score.append(test_score)
            if log_wandb:
                wandb.log({'train_accuracy': train_score, 'train_loss': train_loss, 'test_accuracy': test_score,
                           'epoch': epoch})
        else:
            if log_wandb:
                wandb.log({'train_accuracy': train_score, 'train_loss': train_loss, 'epoch': epoch})

    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        model_save_dir = os.path.join(os.getcwd(), '..', save_model_str)

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        save_model_str = os.path.join(model_save_dir, exp_name + '_model_' + run_id)
        torch.save(model.state_dict(), save_model_str)

    if not use_all_data_to_train:
        logging.info('Accuracy at each epoch: ' + str(score))
        logging.info('Mean of accuracies across all epochs: ' + str(100 * np.mean(score)) + '%')
        logging.info('Accuracy of model at final epoch: ' + str(100 * score[-1]) + '%')

    return 1 - score[-1]


def run_bo(run_pipeline, bo_iter):
    """
    Training loop for configurableNet.
    :param torch_model: model that we are training
    :return:
    """
    set_seed(124)
    logging.basicConfig(level=logging.INFO)
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results/bayesian_optimization')

    pipeline_space = get_pipeline_space()
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        overwrite_working_directory=True,
        root_directory=root_dir,
        max_evaluations_total=bo_iter,
        searcher='bayesian_optimization',
    )
    previous_results, pending_configs = neps.status(root_dir)
    neps.plot(root_dir)


if __name__ == '__main__':
    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss}
    optim_dict = {'rms_prop': torch.optim.RMSprop}

    cmdline_parser = argparse.ArgumentParser('DL WS22/23 Competition')

    cmdline_parser.add_argument('-m', '--model',
                                default='SampleModel',
                                help='Class name of model to train',
                                type=str)
    cmdline_parser.add_argument('-bo',
                                action='store_true',
                                help='Run Bayesian Optimization')
    cmdline_parser.add_argument('-i', '--bo_iter',
                                default=20,
                                help='Number of BO iterations.',
                                type=int)
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
                                default=1e-2,
                                help='Optimizer weight decay',
                                type=float)
    cmdline_parser.add_argument('-dr', '--dropout',
                                default=1e-5,
                                help='Optimizer dropout',
                                type=float)
    cmdline_parser.add_argument('-mo', '--momentum',
                                default=1e-2,
                                help='Optimizer momentum',
                                type=float)
    cmdline_parser.add_argument('-cm', '--cutmix_prob',
                                default=5e-1,
                                help='Cutmix probability',
                                type=float)
    cmdline_parser.add_argument('-be', '--beta',
                                default=1e-1,
                                help='Cutmix beta',
                                type=float)
    cmdline_parser.add_argument('-L', '--training_loss',
                                default='cross_entropy',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='rms_prop',
                                help='Which optimizer to use during training',
                                choices=list(optim_dict.keys()),
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
    cmdline_parser.add_argument('-a', '--use_all_data_to_train',
                                action='store_true',
                                help='Uses the train, validation, and test data to train the model if enabled.')
    cmdline_parser.add_argument('-wb', '--wandb',
                                action='store_true',
                                help='Enables WandB plotting.')

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
        bo=args.bo,
        bo_iter=args.bo_iter,
        num_epochs=args.epochs,
        momentum=args.momentum,
        dropout=args.dropout,
        cutmix_prob=args.cutmix_prob,
        beta=args.beta,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=optim_dict[args.optimizer],
        data_augmentations=eval(args.data_augmentation),
        save_model_str=args.model_path,
        exp_name=args.exp_name,
        use_all_data_to_train=args.use_all_data_to_train,
        log_wandb=args.wandb
    )
