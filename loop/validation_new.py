import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from utils.utils_training.utils_losses.of_loss import one_scale
from utils.utils_training.utils_metric.utils import SegMetrics, FlowMetrics
from utils.utils_vis.vis import visualize_sample_validation
import wandb
import itertools
from tqdm import tqdm

def compute_metrics(predictions, labels, num_classes, flow_result, flow_mask):
    """
    Compute precision, recall, f1-score, IoU, Dice coefficient, and MAE.
    """
    flat_predictions = predictions.cpu().numpy().flatten()
    flat_labels = labels.cpu().numpy().flatten()
    precision = precision_score(flat_labels, flat_predictions, average='weighted', labels=np.arange(num_classes), zero_division=0)
    recall = recall_score(flat_labels, flat_predictions, average='weighted', labels=np.arange(num_classes), zero_division=0)
    f1 = f1_score(flat_labels, flat_predictions, average='weighted', labels=np.arange(num_classes), zero_division=0)
    metrics = SegMetrics(num_classes)
    metrics.update(predictions, labels)
    iou_score = metrics.get_results('mIou')
    dice_score = metrics.get_results('dice_score')

    # Calculate the MAE
    metrics_flow = FlowMetrics()
    metrics_flow.update(flow_result, flow_mask)
    mae_score = metrics_flow.get_results()

    return precision, recall, f1, iou_score, dice_score, mae_score

def log_metrics_to_wandb(metrics_dict):
    """
    Log given metrics to wandb.
    """
    for key, value in metrics_dict.items():
        wandb.log({key: value})

def plot_confusion_matrix(cm, classes, epoch, save_dir):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{epoch}.png'))

def validate_model(model, dataloader_validation, device, segmentation_criterion, save_dir_validation, epoch, config):
    """
    Validate the model and log metrics.
    """
    model.eval()
    # all_preds, all_labels = [], []
    # metrics_summary = {
    #     "segmentation_accuracy_values": [],
    #     "mae_of_values": [],
    #     "precision_values": [],
    #     "recall_values": [],
    #     "f1_values": [],
    #     "iou_values": [],
    #     "dice_coefficient_values": [],
    #     "mae_values": []
    # }
    num_classes=config['num_classes']
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader_validation), total=len(dataloader_validation)):
            segmentation_image, segmentation_mask,stacked_flow,flow_mask = batch['segmentation_image'].to(device), batch['segmentation_mask'].to(device), batch['stacked_flow'].to(device),batch['flow_mask'].to(device)
            seg_result, flow_result = model(segmentation_image, stacked_flow)
            seg_predictions = torch.softmax(seg_result, dim=1)
            seg_predictions = torch.argmax(seg_predictions, dim=1)

            # all_preds.append(seg_predictions.cpu().numpy())
            # all_labels.append(segmentation_mask.cpu().numpy())

            # # Compute losses and metrics
            # segmentation_loss = segmentation_criterion(seg_result, segmentation_mask.long().to(device))
            # flow_loss = one_scale(flow_result, flow_mask)
            #precision, recall, f1, iou_score, dice_score, mae_score = compute_metrics(seg_predictions, segmentation_mask, num_classes, flow_result, flow_mask)

            # # Update metrics summary
            # metrics_summary["segmentation_accuracy_values"].append((seg_predictions == segmentation_mask).float().mean().item())
            # metrics_summary["mae_of_values"].append(flow_loss.item())
            # metrics_summary["precision_values"].append(precision)
            # metrics_summary["recall_values"].append(recall)
            # metrics_summary["f1_values"].append(f1)
            # metrics_summary["iou_values"].append(iou_score)
            # metrics_summary["dice_coefficient_values"].append(dice_score)
            # metrics_summary["mae_of_values"].append(mae_score)

    # Log metrics to wandb
    #log_metrics_to_wandb({f"Validation {key}": np.mean(value) if value else "N/A" for key, value in metrics_summary.items()})
    # Plot confusion matrix
    # cm = confusion_matrix(np.concatenate(all_labels).flatten(), np.concatenate(all_preds).flatten())
    # plot_confusion_matrix(cm, list(class_mapping_combined.keys()), epoch, save_dir_validation)

    # Additional visualizations and logging
    visualize_sample_validation(dataloader_validation, model, batch_idx, device, batch_idx, os.path.join(save_dir_validation, f'validation_sample_{epoch}'), config)

    print(f"Validation is done, Training is starting")
    #return metrics_summary