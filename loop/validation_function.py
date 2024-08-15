import torch
from utils.of_loss import one_scale
from utils.vis import visualize_sample_validation
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.utils_metric.utils import iou, dice_coefficient
from utils.utils import visualize_random_sample
import os
from sklearn.metrics import confusion_matrix
import itertools
from utils.vis import class_mapping_combined
#######################################
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.utils_metric.utils import Metrics, FlowMetrics


def validate_model(model,dataloader_validation, device, segmentation_criterion, save_dir_validation, epoch):
    model.eval()  # Set the model to evaluation mode



    segmentation_loss_values = []
    average_flow_loss_values = []
    total_loss_values = []
    segmentation_accuracy_values = []
    mae_of_values = []
    pixel_difference_values = []
    iou_values = []
    dice_coefficient_values = []
    all_preds = []
    all_labels = []
    precision_values = []
    recall_values = []
    f1_values = []
    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, batch in enumerate(dataloader_validation):
            # Extract the inputs from the batch
            segmentation_image = batch['segmentation_image'].to(device)
            segmentation_mask = batch['segmentation_mask'].to(device)
            stacked_flow = batch['stacked_flow'].to(device)
            flow_mask = batch['flow_mask'].to(device)

            # Forward pass through the model
            seg_result, flow_result_add = model(segmentation_image, stacked_flow)

            # Compute the segmentation predictions
            seg_predictions = torch.argmax(seg_result, dim=1)
            all_preds.append(seg_predictions.cpu().numpy())
            all_labels.append(segmentation_mask.cpu().numpy())

            
            metrics_flow = FlowMetrics()
            metrics_flow.update(flow_result_add, flow_mask)
            mae = metrics_flow.get_results()
            mae_of_values.append(mae)


            segmentation_mask = segmentation_mask.squeeze(1).long().to(device)
            metrics_seg = Metrics(6)
            metrics_seg.update(seg_predictions, segmentation_mask)
            seg_acc = metrics_seg.get_results('overall_acc')
            segmentation_accuracy_values.append(seg_acc)
            ##################################################
            segmentation_loss = segmentation_criterion(seg_result, segmentation_mask)

            flow_loss = one_scale(flow_result_add, flow_mask)

            # Compute the number of elements in the flow tensor
            num_elements = flow_result_add.numel()

            # Compute the average flow loss per element
            average_flow_loss = flow_loss.item() / num_elements

            
            # Compute the precision, recall, and F1-score
            precision = precision_score(segmentation_mask.cpu().numpy().flatten(), seg_predictions.cpu().numpy().flatten(), average='weighted')
            recall = recall_score(segmentation_mask.cpu().numpy().flatten(), seg_predictions.cpu().numpy().flatten(), average='weighted')
            f1 = f1_score(segmentation_mask.cpu().numpy().flatten(), seg_predictions.cpu().numpy().flatten(), average='weighted')

            # Store the precision, recall, and F1-score values
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

            # Compute the IoU and Dice coefficient
            iou_score = metrics_seg.get_results('mIou')
            dice_score = metrics_seg.get_results('dice_score')


            wandb.log({ "Validation Segmentation Accuracy": seg_acc})
            wandb.log({ "Validation Mean Absolute Error": mae})
            wandb.log({ "Validation IoU": iou_score})
            wandb.log({ "Validation Dice Coefficient": dice_score})
            wandb.log({ "Validation Segmentation Loss": segmentation_loss.item()})
            wandb.log({ "Validation Average Flow Loss": average_flow_loss})
            wandb.log({ "Validation Precision": precision})
            wandb.log({ "Validation Recall": recall})
            


    # Compute the average precision, recall, and F1-score
    avg_precision = np.mean(precision_values)
    avg_recall = np.mean(recall_values)
    avg_f1 = np.mean(f1_values)
    #wandb.log({"Validation Precision": avg_precision, "Validation Recall": avg_recall, "Validation F1": avg_f1})


    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    all_preds = all_preds.flatten()
    all_labels = all_labels.flatten()
    cm = confusion_matrix(all_labels, all_preds)
    classes = list(class_mapping.keys())
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))  # classes is the list of your classes
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
    plt.savefig(os.path.join(save_dir_validation, f'confusion_matrix_{epoch}.png'))

    print(f"Confusion Matrix: {cm}")
    avg_iou = np.mean(iou_values)
    avg_dice_coefficient = np.mean(dice_coefficient_values)
    print(f'Validation, Average IoU: {avg_iou}, Average Dice Coefficient: {avg_dice_coefficient}')


    visualize_sample_validation(dataloader_validation, model, batch_idx, device, batch_idx, os.path.join(save_dir_validation, f'validation_sample_{epoch}')) #check here later
    # Create a bar plot for the precision, recall, and F1-score
    metrics = [avg_precision, avg_recall, avg_f1]
    labels = ['Precision', 'Recall', 'F1']
    plt.figure(figsize=(10,10))
    plt.bar(labels, metrics)
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score')
    plt.savefig(os.path.join(save_dir_validation, f'precision_recall_f1_{epoch}.png'))
    
    print(f'Validation, Average segmentation loss: {np.mean(segmentation_loss_values)}')
    print(f"Validation is done, Training is starting")
    return segmentation_accuracy_values, mae_of_values, segmentation_loss_values, average_flow_loss_values, total_loss_values, pixel_difference_values, iou_values, dice_coefficient_values