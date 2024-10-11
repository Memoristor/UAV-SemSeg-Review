# coding=utf-8

import prettytable as pt
import numpy as np

__all__ = [
    'SegmentationScore', 
    'ClassificationScore',
]


class SegmentationScore(object):
    """
    Ternary classification confusion matrix:
        ------------------------------
                True1   True2   True3
        Pred1   TP1,1   FP2,1   FP3,1
        Pred2   FN1,2   TP2,2   FP3,2
        Pred3   FN1,3   FN2,3   TP3,3
        ------------------------------
        All     100%    100%    100%

    Params:
        class_names: list of str. The name of each class, i.e. ['IS', 'BD', ...]
    """

    def __init__(self, class_names, ignore_classes=None, float_format='5.6f'):
        self.class_names = list(class_names)
        self.ignore_classes = list() if ignore_classes is None else ignore_classes
        self.float_format = '{:' + float_format + '}'
        
        self.num_class = len(self.class_names)
        self.confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int32)

    def get_target_matrix(self):
        """
        Get target confusion matrix without ignored classes
        """
        target_idx = list()
        for i in range(self.num_class):
            if self.class_names[i] not in self.ignore_classes:
                target_idx.append(i)

        target_matrix = self.confusion_matrix[target_idx][:, target_idx]
        return target_matrix

    def get_target_classes(self):
        """
        Get target classes without ignored classes
        """
        target_label = list()
        for lbl in self.class_names:
            if lbl not in self.ignore_classes:
                target_label.append(lbl)

        return target_label

    def get_F1(self):
        """
        Calculate F1 score by the confusion matrix:

        F1 = 2 * (PR) / (P + R)
        """
        target_matrix = self.get_target_matrix()

        score = []
        for i in range(target_matrix.shape[0]):
            p = target_matrix[i, i] / np.sum(target_matrix[:, i])
            r = target_matrix[i, i] / np.sum(target_matrix[i, :])
            f1 = (2.0 * p * r) / (p + r)
            score.append(f1)

        return np.array(score)

    def get_OA(self):
        """
        Get OA (Overall Accuracy) by the confusion matrix, it is calculated as: the sum
        of diagonal elements divided by the sum of all elements of the matrix.

        OA = (TP + TN) / (TP + TN + FP + TN)
        """
        target_matrix = self.get_target_matrix()
        return np.diag(target_matrix).sum() / target_matrix.sum()

    def get_CA(self):
        """
        The CA (Class Accuracy) is to calculate the proportion of the correct pixel number
        of each category in all predicted pixels of the category, that is, the accuracy
        rate, and then accumulate the average. The formula for calculating the CA using
        the confusion matrix is: MA is equal to TP on the diagonal divided by the total
        number of pixels in the corresponding column

        MA = (TP) / (TP + FP)
        """
        target_matrix = self.get_target_matrix()
        return np.diag(target_matrix) / target_matrix.sum(axis=0)

    def get_IoU(self):
        """
        The IoU is the result of summing and averaging the ratio of the intersection
        and union between the predicted result and the true value of each category.

        IoU = (TP) / (TP + FP + FN)
        """
        target_matrix = self.get_target_matrix()
        result = []
        for i in range(target_matrix.shape[0]):
            result.append(target_matrix[i, i] / (target_matrix[i, :].sum() + target_matrix[:, i].sum() - target_matrix[i, i]))
        return np.array(result)
        # return np.diag(target_matrix) / (target_matrix.sum(axis=1) + target_matrix.sum(axis=0) - np.diag(target_matrix))

    def update(self, pred, truth):
        """
        Update the confusion matrix by ground truth seg map and predict seg map.

        Params:
            pred: 2-D/3-D np.array. The predict seg map, which shape is (H, W) or (N, H, W)
            truth: 2-D/3-D np.array. The ground truth seg map, which shape is (H, W) or (N, H, W)
        """
        mask = (truth >= 0) & (truth < self.num_class)
        hist = np.bincount(
            self.num_class * truth[mask].astype(int) + pred[mask],
            minlength=self.num_class ** 2,
        ).reshape(self.num_class, self.num_class).T

        self.confusion_matrix += hist

    def reset(self):
        """
        Reset confusion matrix
        """
        self.confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int32)

    def pt_confusion_matrix(self, show=True):
        """
        Print confusion matrix by `prettytable`
        """
        tb = pt.PrettyTable()

        field = ["P|T"]
        for cls in self.class_names:
            field.append(cls)

        tb.field_names = field

        for i, cls in enumerate(self.class_names):
            tb.add_row([cls, *self.confusion_matrix[i].tolist()])

        if show:
            print(tb)

        return tb

    def pt_score(self, key, label=None, show=True):
        """
        Print keys by `prettytable`

        Params
            key: str or list of str. The keys for pretty table, support ['CA', 'IoU', 'F1']
        """
        if isinstance(key, str):
            key = [key]
        elif isinstance(key, list or tuple):
            key = list(key)
        else:
            raise AttributeError('Expected keys: None, str, list or tuple, but given {}'.format(type(key)))

        tb = pt.PrettyTable()

        field = ["" if label is None else label]
        field.extend(self.get_target_classes())

        tb.field_names = field
        for k in key:
            if k == 'CA':
                row_str = [self.float_format.format(x) for x in self.get_CA().tolist()]
            elif k == 'IoU':
                row_str = [self.float_format.format(x) for x in self.get_IoU().tolist()]
            elif k == 'F1':
                row_str = [self.float_format.format(x) for x in self.get_F1().tolist()]
            else:
                raise AttributeError('Unknown key: {}'.format(k))
            tb.add_row([k, *row_str])

        if show:
            print(tb)

        return tb


class ClassificationScore(object):
    """
    The top-1 and top-5 evaluation criteria.
    """

    def __init__(self, num_class):
        self.num_class = num_class

        self.top1_state = {'sum_acc': 0, 'count': 0}
        self.top5_state = {'sum_acc': 0, 'count': 0}

    @staticmethod
    def topk_acc(pred, truth, topk=(1,)):
        """
        Get top-k accuracy by predict score for each class and the truth classes

        Note: There are still some bugs, which get different result from torch.topk()
        """
        pred_maxk = np.argsort(pred, axis=1)[:, -max(topk):][:, ::-1]
        # print(pred_maxk[0])

        acc = list()
        for k in topk:
            pred_match = np.logical_or.reduce(pred_maxk[:, :k] == truth.reshape((-1, 1)), axis=1)
            acc.append(pred_match.astype(np.int32).sum() / pred.shape[0])
        return acc

    def update(self, pred, truth):
        """
        Update state with predict score for each class and the truth classes

        Params:
            pred: 2-D np.array. The predict classification prob which shape is (N, C)
            truth: 1-D np.array. The ground truth classes, which shape is (N, )
        """
        # Get size of predict
        n, c = pred.shape

        # Get top-1 and top-k acc
        topk_acc_list = self.topk_acc(pred, truth, topk=(1, 5))

        # Get top-1 state
        self.top1_state['sum_acc'] += topk_acc_list[0] * n
        self.top1_state['count'] += n

        # Get top-5 state
        self.top5_state['sum_acc'] += topk_acc_list[1] * n
        self.top5_state['count'] += n

    def reset(self):
        """
        Reset state
        """
        self.top1_state = {'sum_acc': 0, 'count': 0}
        self.top5_state = {'sum_acc': 0, 'count': 0}

    def get_top1_acc(self):
        """
        Get top-1 accuracy
        """
        return self.top1_state['sum_acc'] / self.top1_state['count']

    def get_top5_acc(self):
        """
        Get top-5 accuracy
        """
        return self.top5_state['sum_acc'] / self.top5_state['count']

    def pt_score(self, key, label=None, show=True):
        """
        Print keys by `prettytable`

        Params
            key: str or list of str. The keys for pretty table, support ['top1_acc', 'top5_acc']
        """
        if isinstance(key, str):
            key = [key]
        elif isinstance(key, list or tuple):
            key = list(key)
        else:
            raise AttributeError('Expected keys: None, str, list or tuple, but given {}'.format(type(key)))

        tb = pt.PrettyTable()

        field = ["" if label is None else label]
        field.extend(key)
        tb.field_names = field

        row = ['Value']
        for k in key:
            if k == 'top1_acc':
                row.append(self.get_top1_acc())
            elif k == 'top5_acc':
                row.append(self.get_top5_acc())
            else:
                raise AttributeError('Unknown key: {}'.format(k))
        tb.add_row(row)

        if show:
            print(tb)

        return tb
