import numpy as np
import matplotlib.pyplot as plt

class SegMetrics:
    def __init__(self, nCls):
        self.nCls = nCls
        self.reset()

    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nCls, dtype=np.float32)
        self.per_class_iou = np.zeros(self.nCls, dtype=np.float32)
        self.mIou = 0
        self.dice_score = 0 
        self.batch_count = 0

    def fast_hist(self, a, b):
        assert a.size == 307200, f"Expected a.size to be 307200, but got {a.size}"
        assert b.size == 307200, f"Expected b.size to be 307200, but got {b.size}"
        k = (a >= 0) & (a < self.nCls)
        return np.bincount(self.nCls * a[k].astype(int) + b[k], minlength=self.nCls ** 2).reshape(self.nCls, self.nCls)

    def compute_hist(self, predict, gt):
        hist = self.fast_hist(gt, predict)
        return hist

    def update(self, predict, gt):
        predict = predict.cpu().detach().numpy().flatten()
        gt = gt.cpu().detach().numpy().flatten()

        epsilon = 0.000000001
        hist = self.compute_hist(predict, gt)
        overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
        mIou = np.nanmean(per_class_iou)

        # Compute Dice score
        dice_score = 2 * np.sum(predict[gt==1]) / (np.sum(predict) + np.sum(gt) + epsilon)

        # MAE

        self.overall_acc += overall_acc
        self.per_class_acc += per_class_acc
        self.per_class_iou += per_class_iou
        self.mIou += mIou
        self.dice_score += dice_score  # Update Dice score
        self.batch_count += 1

    def get_results(self, metric=None):
        overall_acc = self.overall_acc / self.batch_count
        per_class_acc = self.per_class_acc / self.batch_count
        per_class_iou = self.per_class_iou / self.batch_count
        mIou = self.mIou / self.batch_count
        dice_score = self.dice_score / self.batch_count  # Get average Dice score


        if metric is None:
            return overall_acc, per_class_acc, per_class_iou, mIou, dice_score  # Return all metrics
        elif metric == 'overall_acc':
            return overall_acc
        elif metric == 'per_class_acc':
            return per_class_acc
        elif metric == 'per_class_iou':
            return per_class_iou
        elif metric == 'mIou':
            return mIou
        elif metric == 'dice_score':
            return dice_score

        else:
            raise ValueError(f'Unknown metric: {metric}')
        
    def plot_histogram(self, predict, gt):
        predict = predict.cpu().numpy().flatten()
        gt = gt.cpu().numpy().flatten()
        hist = self.compute_hist(predict, gt)

        plt.figure(figsize=(20,20))
        plt.imshow(hist, cmap='hot', interpolation='nearest')
        plt.title('Confusion matrix/Histogram')
        plt.xlabel('Predicted')
        plt.ylabel('Ground truth')

        for i in range(hist.shape[0]):
            for j in range(hist.shape[1]):
                if hist[i, j]>50000:
                    color = 'k'
                else:
                    color = 'w'
                text = plt.text(j, i, hist[i, j], ha="center", va="center", color=color)

        plt.show()

class FlowMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mae_sum = 0 
        self.batch_count = 0

    def update(self, predict, gt):
        predict = predict.cpu().detach().numpy().flatten()
        gt = gt.cpu().detach().numpy().flatten()

        # MAE
        mae = np.mean(np.abs(predict - gt))

        self.mae_sum += mae
        self.batch_count += 1

    def get_results(self):
        mae = self.mae_sum / self.batch_count
        return mae
    

def iou(y_true, y_pred):
    # Flatten the input arrays and convert to boolean
    y_true = np.asarray(y_true).astype(np.bool_)
    y_pred = np.asarray(y_pred).astype(np.bool_)

    # Calculate the intersection and union
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)

    # Calculate IoU
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

def dice_coefficient(y_true, y_pred):
    # Flatten the input arrays and convert to boolean
    y_true = np.asarray(y_true).astype(np.bool_)
    y_pred = np.asarray(y_pred).astype(np.bool_)

    # Calculate the intersection
    intersection = np.logical_and(y_true, y_pred)

    # Calculate Dice coefficient
    dice_score = 2. * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred))

    return dice_score