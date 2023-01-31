import argparse
import os
import logging

import wandb
import torch
import neps
import numpy as np

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
         momentum=0.9,
         dropout=1e-5,
         cutmix_prob=0.3,
         beta=0.1,
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer=torch.optim.RMSprop,
         data_augmentations=None,
         exp_name='',
         save_model_str=None,
         use_all_data_to_train=False,
         log_wandb=False,
         competition=False
         ):
    """
    Main function running HPO or single training of a given model.
    :param data_dir: dataset path (str)
    :param torch_model: model to train (torch.nn.Module)
    :param bo: indicator whether Bayesian Optimization is used for hyperparameter optimization (bool)
    :param bo_iter: number of evaluations for Bayesian Optimization (int)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param weight_decay: model optimizer weight_decay (float)
    :param dropout: model optimizer dropout (float)
    :param momentum: model optimizer momentum (float)
    :param cutmix_prob: probability of CutMix occurrence for regularization (float)
    :param beta: distribution from which image combination ratio is drawn in CutMix (float)
    :param train_criterion: loss used during training (torch.nn._Loss)
    :param model_optimizer: model optimizer used during training (torch.optim.Optimizer)
    :param data_augmentations: list of data augmentations to apply
    (list[transformations], transforms.Composition[list[transformations]], None)
    :param exp_name: experiment name (str)
    :param save_model_str: path of saved models (str)
    :param use_all_data_to_train: indicator whether to use all the data for training (bool)
    :param log_wandb: indicator whether to plot performance score in WandB (bool)
    :param competition: overwrites model filename to "fast_model" (bool)
    :return:
    """
    # Set root directory for importing functions directly from folders
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    # Create a callable function based on train_and_evaluate to enable hyperparameter modifications during training
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
                           log_wandb=log_wandb,
                           competition=competition
                           )
    # Iterative training with hyperparameter optimization
    if bo:
        run_bo(run_pipeline=run_pipeline, bo_iter=bo_iter)
    # Single training
    else:
        run_pipeline(learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout, momentum=momentum,
                     cutmix_prob=cutmix_prob, beta=beta)


def train_and_evaluate(data_dir,
                       torch_model,
                       bo=False,
                       num_epochs=40,
                       batch_size=16,
                       learning_rate=1e-3,
                       weight_decay=0.01,
                       momentum=0.9,
                       dropout=1e-5,
                       cutmix_prob=0.3,
                       beta=0.1,
                       train_criterion=torch.nn.CrossEntropyLoss,
                       model_optimizer=torch.optim.RMSprop,
                       data_augmentations=None,
                       exp_name='',
                       save_model_str=None,
                       use_all_data_to_train=False,
                       log_wandb=False,
                       competition=False
                       ):
    """
    Training loop. Parameters described in main() .
    :return: (1-accuracy) on the validation set or None if -a enabled (float/None)
    """

    # Run ID for model retrieval and WandB plotting
    run_id = f'lr={learning_rate:.5f}_wd={weight_decay:.5f}_m={momentum:.5f}_dr={dropout:.5f}_cmix_prob={cutmix_prob:.5f}_beta' \
             f'={beta:.5f}_epochs={num_epochs}'

    # Initialize WandB (optional)
    if log_wandb:
        # Get the desired location for wandb folder
        os.environ['WANDB_DIR'] = os.path.join(os.getcwd(), '..')
        wandb.login()
        wandb.init(project=f'{exp_name}', id=run_id, reinit=True)

    # Configure device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set data augmentation
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

    # Get input shape for summary
    channels, img_height, img_width = train_data[0][0].shape
    input_shape = (channels, img_height, img_width)

    # Instantiate training criterion
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

    # Initialize model
    model = torch_model(num_classes=len(train_data.classes), dropout=dropout).to(device)

    # Initialize optimizer as in https://arxiv.org/abs/2104.00298
    optimizer = model_optimizer(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                momentum=momentum)

    # Log info
    logging.info('Model being trained:')
    summary(model, input_shape,
            device='cuda' if torch.cuda.is_available() else 'cpu')

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
                score.append(train_score)
                wandb.log({'train_accuracy': train_score, 'train_loss': train_loss, 'epoch': epoch})

    if save_model_str:
        model_save_dir = os.path.join(os.getcwd(), '..', save_model_str)

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        # For competition purposes the model is written directly to the "fast_model" file
        if competition:
            save_model_str = os.path.join(model_save_dir, 'fast_model')
        else:
            save_model_str = os.path.join(model_save_dir, exp_name + '_model_' + run_id)
        torch.save(model.state_dict(), save_model_str)

    if not use_all_data_to_train:
        logging.info('Accuracy at each epoch: ' + str(score))
        logging.info('Mean of accuracies across all epochs: ' + str(100 * np.mean(score)) + '%')
        logging.info('Accuracy of model at final epoch: ' + str(100 * score[-1]) + '%')
    # Return 1 - accuracy score, so it can be minimized in Bayesian Optimization
    if bo:
        return 1 - score[-1]


def run_bo(run_pipeline, bo_iter):
    """
    Search for HP configurations using Bayesian Optimization and then run model parametrized by them
    :param run_pipeline: function to train and evaluate model that returns performance score to optimize
    :param bo_iter: number of BO evaluations, number of tried HP configurations
    :return:
    """
    set_seed(124)
    logging.basicConfig(level=logging.INFO)
    root_dir = '../results/bayesian_optimization'

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
    cmdline_parser.add_argument('-D', '--data_dir',
                                default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'),
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-bo',
                                action='store_true',
                                help='Enables Bayesian Optimization for Hyperparameter search')
    cmdline_parser.add_argument('-i', '--bo_iter',
                                default=20,
                                help='Number of Bayesian Optimization iterations',
                                type=int)
    cmdline_parser.add_argument('-e', '--epochs',
                                default=100,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=32,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-mo', '--momentum',
                                default=1e-2,
                                help='Optimizer momentum',
                                type=float)
    cmdline_parser.add_argument('-dr', '--dropout',
                                default=1e-5,
                                help='Optimizer dropout',
                                type=float)
    cmdline_parser.add_argument('-l', '--learning_rate',
                                default=1e-3,
                                help='Optimizer learning rate',
                                type=float)
    cmdline_parser.add_argument('-w', '--weight_decay',
                                default=1e-2,
                                help='Optimizer weight decay',
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
    cmdline_parser.add_argument('-d', '--data-augmentation',
                                default='resize_to_224x224',
                                help='Data augmentation to apply to data before passing to the model.')
    cmdline_parser.add_argument('-n', '--exp_name',
                                default='default',
                                help='Name of this experiment',
                                type=str)
    cmdline_parser.add_argument('-c', '--competition',
                                action='store_true',
                                help='Overwrites model filename.')
    cmdline_parser.add_argument('-p', '--model_path',
                                default='models',
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-a', '--use_all_data_to_train',
                                action='store_true',
                                help='Uses the train, validation, and test data to train the model if enabled.')
    cmdline_parser.add_argument('-wb', '--wandb',
                                action='store_true',
                                help='Enables WandB plotting.')
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')

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
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        momentum=args.momentum,
        cutmix_prob=args.cutmix_prob,
        beta=args.beta,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=optim_dict[args.optimizer],
        data_augmentations=eval(args.data_augmentation),
        exp_name=args.exp_name,
        competition=args.competition,
        save_model_str=args.model_path,
        use_all_data_to_train=args.use_all_data_to_train,
        log_wandb=args.wandb
    )
