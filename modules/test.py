# coding=utf-8

from calc_score import SegmentationScore
from torch.utils.data import DataLoader
from modules.basic import BasicModule
from PIL import Image
from tqdm import tqdm
from time import time
import numpy as np
import torch
import os

from torch.autograd import Variable
from models.graph_based.fullcrf import fullcrf
from models.graph_based.convcrf import convcrf


__all__ = ['TestModule']


def arr_to_str(arr, max_len=128):
    """
    Convert numpy array (ASCII) to str
    """
    string = ''
    for i, a in enumerate(arr):
        if i < max_len and a != 0:
            string = string + chr(a)
        else:
            break
    return string


class TestModule(BasicModule):
    """
    The module which is used for the testing dataset

    Params:
        model: nn.Module. The `ConvNet` model for training.
        args: parser.parse_args. The other custom arguments.
        test_dataset: torch.utils.data.Dataset. The dataset for testing
    """

    def __init__(self, model, args, test_dataset):
        super(TestModule, self).__init__(model, args)

        self.test_dataset = test_dataset

        # Get score matrices
        self.test_score = SegmentationScore(class_names=self.test_dataset.class_rgb.keys(), ignore_classes=args.ignore_classes)

        # Create dirs for outputs
        if self.args.densecrf:
            self.output_dir += '_densecrf'
        if self.args.convcrf: 
            self.output_dir += f'_convcrf_{self.args.convcrf_fsize}'
        
        self.save_images =  os.path.join(self.output_root, 'images', self.output_dir)
        self.save_tests =  os.path.join(self.output_root, 'tests', self.output_dir)
        os.makedirs(self.save_images, exist_ok=True)
        os.makedirs(self.save_tests, exist_ok=True)

    def test_model(self):
        """
        Test model
        """
        # Prepare model and score matrix
        self.model.eval()
        self.test_score.reset()

        # Create dataloader for testing
        dataloader = DataLoader(
            dataset=self.test_dataset,
            num_workers=self.args.num_workers,
            batch_size=1,
            shuffle=False
        )

        # Eval without grad
        with torch.no_grad():

            bar = tqdm(dataloader)
            ptime = []
            for i, truth in enumerate(bar):
            
                # Get file names and image size
                fnm = truth['fnm']
                hw = truth['hw']

                # To cuda if GPU is available
                if torch.cuda.is_available():
                    for k in truth.keys():
                        truth[k] = truth[k].to(torch.device(self.device))

                # Get and process predict results
                start_t = time()
                pred = self.model(truth['img'])
                end_t = time()
                ptime.append(end_t - start_t)

                bar.set_description(f"[Test] Model: {self.model_name}")

                for k in pred.keys():
                    if k == 'lbl':
                        x = truth['img']
                        y = truth[k]
                        p = pred[k]
                        
                        # Conditional Random Field
                        if self.args.densecrf or self.args.convcrf:

                            # get basic hyperparameters
                            nclasses = p.shape[1]
                            shape = p.shape[2:4]
                            
                            if self.args.convcrf:
                                config = convcrf.default_conf
                                config['filter_size'] = self.args.convcrf_fsize
                                config['pyinn'] = False
                                config['col_feats']['schan'] = 0.1  # normalization is used
                                use_gpu = torch.cuda.is_available()

                                # Create CRF module
                                gausscrf = convcrf.GaussCRF(conf=config, shape=shape, nclasses=nclasses, use_gpu=use_gpu)

                                img_var = Variable(x)
                                unary_var = Variable(torch.softmax(p, dim=1))
                                
                                # move to GPU if requested
                                if use_gpu:
                                    gausscrf.cuda()

                                # Perform CRF inference
                                start_t = time()
                                p = gausscrf.forward(unary=unary_var, img=img_var, num_iter=5)
                                end_t = time()
                                ptime[-1] += end_t - start_t
                            
                            if self.args.densecrf:
                                config = fullcrf.default_conf
                                config['col_feats']['schan'] = 0.1  # normalization is used
                                
                                # Perform CRF inference
                                start_t = time()
                                densecrf = fullcrf.FullCRF(config, shape, nclasses)
                                p = densecrf.batched_compute(torch.softmax(p, dim=1), x, softmax=False)
                                end_t = time()
                                p = torch.from_numpy(np.stack(p, axis=0).transpose(0, 3, 1, 2))
                                ptime[-1] += end_t - start_t
                                                            
                        # Update score matrix
                        z = torch.argmax(p, dim=1)
                        z = z.cpu().detach().numpy()
                        y = y.cpu().detach().numpy()
                        self.test_score.update(z, y)

                        # Get save path for `lbl`
                        save_path = os.path.join(self.save_images, k)
                        os.makedirs(save_path, exist_ok=True)

                        # Save image
                        for i in range(z.shape[0]):
                            z_dec = dataloader.dataset.decode_lbl(z[i])
                            if dataloader.dataset.chw_format:
                                z_dec = z_dec.transpose((1, 2, 0)).astype(np.uint8)
                                
                            dec_img = Image.fromarray(z_dec)
                            dec_img.resize((hw[i][1], hw[i][0]))
                            dec_img.save(os.path.join(save_path, arr_to_str(fnm[i])))

        # Save score matrix to file
        with open(os.path.join(self.save_tests, 'test.txt'), 'w') as f:
            f.write(f'[Test] Model: {self.model_name}, Optimizer: {self.args.optimizer}, Epoch: {self.start_epoch}, Seed: {self.args.seed}\n')
            f.write('\n')

            # Output fusion matrix
            f.write('[Test] The fusion matrix is\n')
            f.write(str(self.test_score.pt_confusion_matrix(show=False)))
            f.write('\n\n')

            # Scores for all categories
            f.write('[Test] The scores for all categories\n')
            f.write(f'[Test] OA: {self.test_score.get_OA():4.4f}, '
                    f'mCA: {np.nanmean(self.test_score.get_CA()):4.4f}, '
                    f'mIoU: {np.nanmean(self.test_score.get_IoU()):4.4f}, '
                    f'mF1: {np.nanmean(self.test_score.get_F1()):4.4f}\n')
            f.write(str(self.test_score.pt_score(['CA', 'IoU', 'F1'], show=False)))
            f.write('\n\n') 

            # Average processing speed and FPS
            f.write(f'[Test] Average process time: {np.nanmean(ptime) * 1000:3.3f} ms, FPS: {1 / np.nanmean(ptime):3.3f}\n')
            f.write('\n\n') 
            
        # Print saved information
        with open(os.path.join(self.save_tests, 'test.txt'), 'r') as f:
            print(f.read())
