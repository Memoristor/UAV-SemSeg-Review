# coding=utf-8

import numpy as np
import argparse
import warnings
import torch
import random
import os

from modules import *
import datasets
import models


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    parser.add_argument('--input_height', type=int, default=512, help='Resized image height')
    parser.add_argument('--input_width', type=int, default=512, help='Resized image width')
    parser.add_argument('--num_epoch', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--optimizer', type=str, default='SGD', help='The optimizer for training')
    parser.add_argument('--init_lr', type=float, default=0.002, help='Initial learning rate')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='The gamma for the learning rate adjustment')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--phase', type=str, default='train', help='Phase choice = {train, test}')

    parser.add_argument('--seed', type=int, default=132, help='The random seed for python and torch')
    parser.add_argument('--model', type=str, nargs='+', help='The ConvNet model will be used')
    parser.add_argument('--densecrf', action='store_true', help='Use denseCRF in test phase or not')
    parser.add_argument('--convcrf', action='store_true', help='Use convCRF in test phase or not')
    parser.add_argument('--convcrf_fsize', type=int, default=13, help='The filter size of convCRF in test phase')

    parser.add_argument('--dataset', type=str, default='UAVid', help='The dataset which will be used')
    parser.add_argument('--ignore_classes', type=str, nargs='+', help='The classes to be ignored')
    parser.add_argument('--train_set', type=str, help='The path of train dataset', default='./data')
    parser.add_argument('--valid_set', type=str, help='The path of valid dataset', default='./data')
    parser.add_argument('--test_set', type=str, help='The path of test dataset', default='./data')
    parser.add_argument('--output_root', type=str, help='The path output sources', default='./output')

    parser.add_argument('--resume', type=str, default='epoch_last.pth', help='The saved model for resume')
    parser.add_argument('--retrain', action='store_true', help='Retrain model from first epoch or not')
    
    parser.add_argument('--init_group', action='store_true', help='Init default process group or not')
    parser.add_argument('--init_method', type=str, default='tcp://localhost:23456', help='Init methods for default process group')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    os.environ['PYTHONASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.init_group:
        torch.distributed.init_process_group('nccl', init_method=args.init_method, rank=0, world_size=1)

    if isinstance(args.model, str):
        model_list = [getattr(models, args.model)]
    else:
        model_list = [getattr(models, m) for m in args.model]
        
    if args.phase == 'train':
        train_dataset = getattr(datasets, args.dataset)(
            root_path=args.train_set,
            image_size=(args.input_height, args.input_width),
            phase='train',
            chw_format=True,
        )

        valid_dataset = getattr(datasets, args.dataset)(
            root_path=args.valid_set,
            image_size=(args.input_height, args.input_width),
            phase='valid',
            chw_format=True,
        )

        assert len(train_dataset) > 0 and len(valid_dataset) > 0

        for model in model_list:
            trainer = TrainModule(
                model=model(num_class=train_dataset.num_class),
                args=args,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                weight=np.array(train_dataset.weight),
            )

            trainer.train_model()

    else:  # test
        
        test_dataset = getattr(datasets, args.dataset)(
            root_path=args.test_set,
            image_size=(args.input_height, args.input_width),
            phase='test',
            chw_format=True,
        )

        assert len(test_dataset) > 0

        for model in model_list:
            tester = TestModule(
                model=model(num_class=test_dataset.num_class),
                args=args,
                test_dataset=test_dataset,
            )

            tester.test_model()
