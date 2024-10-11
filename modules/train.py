# coding=utf-8

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from modules.basic import BasicModule
from tqdm import tqdm
import calc_score
import numpy as np
import loss as ls
import torch
import json
import os

__all__ = ['TrainModule']


class TrainModule(BasicModule):
    """
    The module which is used for the training on the training dataset and testing on the validation dataset

    Params:
        model: nn.Module. The `ConvNet` model for training.
        args: parser.parse_args. The other custom arguments.
        train_dataset: torch.utils.data.Dataset. The dataset for training
        valid_dataset: torch.utils.data.Dataset. The dataset for validation
        weight: None or numpy array (default None). The weight of each classes.
    """

    def __init__(self, model, args, train_dataset, valid_dataset, weight=None):
        super(TrainModule, self).__init__(model, args)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.weight = weight

        # Get score matrices
        self.train_score = calc_score.SegmentationScore(class_names=self.train_dataset.class_rgb.keys(), ignore_classes=args.ignore_classes)
        self.valid_score = calc_score.SegmentationScore(class_names=self.valid_dataset.class_rgb.keys(), ignore_classes=args.ignore_classes)

        # Get losses for training
        self.ce_loss = ls.CrossEntropy2D(custom_weight=self.weight)
        self.bce_loss = ls.BinaryCrossEntropy2D()
        self.br_loss = ls.BoundariesRelaxation2D(custom_weight=self.weight, window_size=5)

    def adjust_learning_rate(self, current_iter, total_iter):
        """
        The learning rate strategy:
        lr = base_lr * (1 - current_iter/total_iter) ^ gamma
        """
        self.learning_rate = self.args.init_lr * (1 - current_iter / total_iter) ** self.args.lr_gamma

    def losses(self, pred, truth):
        """
        Calculate the loss of predicted and tested values/maps

        Params:
            truth: dict. The truth data, e.g. {'body': body, 'edge': edge}
            pred: dict. The predict data, e.g. {'body': body, 'edge': edge}

        Returns:
            return the dict of losses, e.g. {'body': body_loss, 'edge': edge_loss}
        """
        loss = {}
        for k in pred.keys():
            if k == 'lbl':
                y, z = truth[k], pred[k]
                loss[k] = self.ce_loss(z, y)
            if k == 'aux1':  # for bisenetv1
                y, z = truth['lbl'], pred[k]
                loss[k] = self.ce_loss(z, y)
            if k == 'aux2':  # for bisenetv1
                y, z = truth['lbl'], pred[k]
                loss[k] = self.ce_loss(z, y)
                
        return loss

    def run_epoch(self, dataloader, epoch, for_train=True, summary_writer=None, summary_freq=10):
        """
        Train or test for one epoch and get the scores and losses

        Params:
            dataloader: DataLoader. A data iterator for training or testing
            epoch: int. Current epoch
            for_train: bool (default True). Indicates whether to use for training (gradient enabled) or test (gradient disabled)
        """
        if for_train is True:
            self.model.train()
            score_matrix = self.train_score
            epoch_tag = 'Train'
        else:
            self.model.eval()
            score_matrix = self.valid_score
            epoch_tag = 'Valid'

        score_matrix.reset()

        epoch_loss = dict()
        epoch_score = dict()

        with torch.enable_grad() if for_train is True else torch.no_grad():

            bar = tqdm(dataloader)
            for i, truth in enumerate(bar):

                # To cuda if GPU is available
                if torch.cuda.is_available():
                    for k in truth.keys():
                        truth[k] = truth[k].to(torch.device(self.device))

                # Get predict results and losses
                pred = self.model(truth['img'])
                loss = self.losses(pred, truth)

                # Update confusion matrix
                lbl_truth = truth['lbl'].cpu().detach().numpy()
                lbl_pred = np.argmax(pred['lbl'].cpu().detach().numpy(), axis=1)
                score_matrix.update(truth=lbl_truth, pred=lbl_pred)

                # Add to summary
                if summary_writer is not None and i % summary_freq == 0:
                    step = epoch * (len(dataloader) // summary_freq + 1) + i // summary_freq
                    ind = np.random.randint(0, truth['img'].shape[0])

                    for k in truth.keys():
                        if k == 'img':
                            x = truth[k][ind].cpu().detach().numpy().transpose((1, 2, 0))
                            if dataloader.dataset.div_std is True:
                                x = x * dataloader.dataset.std + dataloader.dataset.mean
                            else:
                                x = x + dataloader.dataset.mean
                                
                            sum_tag = f'{epoch_tag}/{k}/'
                            summary_writer.add_image(sum_tag + 'truth', x, step, dataformats='HWC')

                    for k in pred.keys():
                        if k == 'lbl':
                            y, z = truth['lbl'][ind], torch.argmax(pred[k][ind], dim=0)
                            y, z = y.cpu().detach().numpy(), z.cpu().detach().numpy()
                            y = dataloader.dataset.decode_lbl(y) / 255.0
                            z = dataloader.dataset.decode_lbl(z) / 255.0
                            sum_tag = f'{epoch_tag}/{k}/'
                            summary_writer.add_image(sum_tag + 'truth', y, step, dataformats='CHW')
                            summary_writer.add_image(sum_tag + 'pred', z, step, dataformats='CHW')

                    # Flush summary writer
                    summary_writer.flush()

                # Set description
                bar.set_description(f"[{epoch_tag}] Epoch: {epoch + 1}/{self.args.num_epoch}")

                # Update loss
                for k in loss.keys():
                    if k not in epoch_loss.keys():
                        epoch_loss[k] = loss[k].cpu().detach().numpy()
                    else:
                        epoch_loss[k] += loss[k].cpu().detach().numpy()

                # If for training
                if for_train:
                    loss_all = 0
                    for k in loss.keys():
                        loss_all += loss[k]

                    # Update learning rate
                    for param_group in self.optimizer.param_groups:
                        self.adjust_learning_rate(
                            current_iter=len(dataloader) * epoch + i,
                            total_iter=len(dataloader) * self.args.num_epoch
                        )
                        param_group['lr'] = self.learning_rate

                    # Gradient backward
                    self.optimizer.zero_grad()
                    loss_all.backward()
                    self.optimizer.step()

            # Get loss of this epoch
            for k in epoch_loss.keys():
                epoch_loss[k] = epoch_loss[k] / len(dataloader)

            # Get score of this epoch
            epoch_score['OA'] = score_matrix.get_OA()  # A number
            epoch_score['CA'] = score_matrix.get_CA().tolist()  # A list
            epoch_score['mCA'] = np.nanmean(score_matrix.get_CA())  # A number
            epoch_score['IoU'] = score_matrix.get_IoU().tolist()  # A list
            epoch_score['mIoU'] = np.nanmean(score_matrix.get_IoU())  # A number
            epoch_score['F1'] = score_matrix.get_F1().tolist()  # A list
            epoch_score['mF1'] = np.nanmean(score_matrix.get_F1())  # A number

        return epoch_loss, epoch_score

    def train_model(self):
        """
        Train model
        """
        # Create dataloader for training and validation
        train_dataloader = DataLoader(
            dataset=self.train_dataset, 
            num_workers=self.args.num_workers,
            batch_size=self.args.batch_size, 
            shuffle=True, 
            drop_last=True,
        )

        valid_dataloader = DataLoader(
            dataset=self.valid_dataset, 
            num_workers=self.args.num_workers,
            batch_size=self.args.batch_size, 
            shuffle=False,
        )

        # Create a `SummaryWriter` object
        writer = SummaryWriter(self.save_logs)

        # Create or load record as `dict()`
        json_path = os.path.join(self.save_records, f'{self.model_name}.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                records = json.load(f)
        else:
            records = dict()

        # Best scores
        best_valid = {
            'OA': {'value': 0, 'epoch': 0},
            'mCA': {'value': 0, 'epoch': 0},
            'mIoU': {'value': 0, 'epoch': 0},
            'mF1': {'value': 0, 'epoch': 0},
        }
        
        # Start training
        for epoch in range(self.start_epoch, self.args.num_epoch):

            # Print key information
            print(f'[Train] model: {self.model_name}, dataset: {self.args.dataset}, init lr: {self.args.init_lr:5.5f}, current lr: {self.learning_rate:5.5f}, seed: {self.args.seed}')

            # Add learning rate to scalars
            writer.add_scalar('lr/last_lr', self.learning_rate, epoch)

            # Reset score matrix
            self.train_score.reset()
            self.valid_score.reset()

            # Run epoch
            train_loss, train_score = self.run_epoch(train_dataloader, epoch, True, writer, summary_freq=100)
            valid_loss, valid_score = self.run_epoch(valid_dataloader, epoch, False, writer, summary_freq=50)

            # Add loss and scores to scalars
            for k in train_loss.keys():
                if isinstance(train_loss[k], np.float) or isinstance(train_loss[k], np.int):
                    writer.add_scalars('loss/%s' % k, {'train': train_loss[k], 'valid': valid_loss[k]}, epoch)

            for k in train_score.keys():
                if isinstance(train_score[k], np.float) or isinstance(train_score[k], np.int):
                    writer.add_scalars('score/%s' % k, {'train': train_score[k], 'valid': valid_score[k]}, epoch)

            # Flush summary writer
            writer.flush()

            # Add records and save it
            for k in train_loss.keys():
                if k not in records.keys():
                    records[k] = {'train': [train_loss[k]], 'valid': [valid_loss[k]]}
                else:
                    records[k]['train'].append(train_loss[k])
                    records[k]['valid'].append(valid_loss[k])

            for k in train_score.keys():
                if k not in records.keys():
                    records[k] = {'train': [train_score[k]], 'valid': [valid_score[k]]}
                else:
                    records[k]['train'].append(train_score[k])
                    records[k]['valid'].append(valid_score[k])

            with open(json_path, 'w') as f:
                json.dump(records, f, indent=3)

            # Save model
            self.save_model(os.path.join(self.save_weights, 'epoch_last.pth'), epoch)
            
            # Save best model
            for k in best_valid.keys():
                # Save model and performance
                if valid_score[k] > best_valid[k]['value']:
                    best_valid[k]['value'] = valid_score[k]
                    best_valid[k]['epoch'] = epoch
                    
                    self.save_model(
                        path=os.path.join(self.save_weights, f'best_{k}.pth'),
                        epoch=epoch,
                    )
                    
                    with open(os.path.join(self.save_weights, f'best_{k}.txt'), 'w') as f:
                        f.write(f'[Valid] Model: {self.model_name}, Optimizer: {self.args.optimizer}, Epoch: {epoch + 1}, Seed: {self.args.seed}\n')
                        f.write('\n')

                        # Output fusion matrix
                        f.write('[Valid] The fusion matrix is\n')
                        f.write(str(self.valid_score.pt_confusion_matrix(show=False)))
                        f.write('\n\n')

                        # Scores for all categories
                        f.write('[Valid] The scores for all categories\n')
                        f.write(f'[Valid] OA: {self.valid_score.get_OA():4.4f}, '
                                f'mCA: {np.nanmean(self.valid_score.get_CA()):4.4f}, '
                                f'mIoU: {np.nanmean(self.valid_score.get_IoU()):4.4f}, '
                                f'mF1: {np.nanmean(self.valid_score.get_F1()):4.4f}\n')
                        f.write(str(self.valid_score.pt_score(['CA', 'IoU', 'F1'], show=False)))
                        f.write('\n\n') 

                print(f"[Valid] {k}: {valid_score[k]:5.5f}, best {k}: {best_valid[k]['value']:5.5f}/{best_valid[k]['epoch']}")
                
            self.valid_score.pt_score(['IoU', 'CA', 'F1'], label='Valid')

        # Close summary
        writer.close()
