import torch
import numpy as np
from utils.utils import visualize_random_sample
import os
from loop.validation_new import validate_model
import wandb
from utils.utils_training.utils_metric.utils import SegMetrics, FlowMetrics
from tqdm import tqdm

def calculate_losses(seg_result, segmentation_mask, flow_result_add, flow_mask, segmentation_criterion, flow_criterion):
    segmentation_loss = segmentation_criterion(seg_result, segmentation_mask)
    flow_loss = flow_criterion(flow_result_add, flow_mask)
    total_loss = segmentation_loss + flow_loss
    return segmentation_loss, flow_loss, total_loss

def calculate_metrics(seg_predictions, segmentation_mask, flow_result_add, flow_mask,config):
    seg_metrics = SegMetrics(config['num_classes'])
    seg_metrics.update(seg_predictions, segmentation_mask)
    segmentation_accuracy = seg_metrics.get_results('overall_acc')
    iou_score = seg_metrics.get_results('mIou')
    #dice_score = seg_metrics.get_results('dice_score')

    flow_metrics = FlowMetrics()
    flow_metrics.update(flow_result_add, flow_mask)
    mae = flow_metrics.get_results()

    return segmentation_accuracy, iou_score,  mae

def save_model_state(model, epoch, save_dir):
    model_path = save_dir + f'/model_at_{epoch}_epoch.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch
    }, model_path)
    
    # Save the model in wandb
    wandb.save(model_path)

def process_batch(batch, model, device, segmentation_criterion, flow_criterion, optimizer,config):
    # Extract the inputs from the batch
    segmentation_image = batch['segmentation_image'].to(device)
    segmentation_mask = batch['segmentation_mask'].to(device)
    stacked_flow = batch['stacked_flow'].to(device)
    flow_mask = batch['flow_mask'].to(device)

    # Forward pass through the model
    seg_result, flow_result_add = model(segmentation_image, stacked_flow)

    # Compute the segmentation predictions
    seg_predictions = torch.softmax(seg_result, dim=1)
    seg_predictions = torch.argmax(seg_predictions, dim=1)
    segmentation_mask = segmentation_mask.view(1, segmentation_image.shape[2], segmentation_image.shape[3] )

    # Calculate losses
    segmentation_loss, flow_loss, total_loss = calculate_losses(seg_result, segmentation_mask, flow_result_add, flow_mask, segmentation_criterion, flow_criterion)

    # Calculate metrics
    segmentation_accuracy, iou_score,  mae = calculate_metrics(seg_predictions, segmentation_mask, flow_result_add, flow_mask,config)

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return segmentation_loss, flow_loss, total_loss, segmentation_accuracy, iou_score,  mae


def continue_training_from_checkpoint(model, dataloader_train, dataloader_validation, device, segmentation_criterion, flow_criterion, optimizer, scheduler, save_dir_training, save_dir_validation, model_metrics_values, config):

    # Continue training from the loaded epoch
    train_model(model, dataloader_train, dataloader_validation, device, segmentation_criterion, flow_criterion, optimizer, scheduler, save_dir_training, save_dir_validation, model_metrics_values, config, start_epoch)


def train_model(model, dataloader_train,dataloader_validation, device,  segmentation_criterion, flow_criterion, optimizer, scheduler, save_dir_training, save_dir_validation, model_metrics_values,config, start_epoch):
    segmentation_loss_values = []
    flow_loss_values = []
    total_loss_values = []
    segmentation_accuracy_values = []
    mae_of_values = []
    pixel_difference_values = []
    last_sample = None
    iou_values = []
    #dice_coefficient_values = []


    #checkpoint_interval = 10
    for epoch in range(start_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        seg_loss_accumulator = 0
        flow_loss_accumulator = 0
        seg_acc_accumulator = 0
        mae_accumulator = 0
        iou_accumulator = 0
        #dice_score_accumulator = 0  
        for batch_idx, batch in tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Epoch {epoch}/{config['num_epochs']}"):
            torch.cuda.empty_cache()

            # Process the batch and get the losses and metrics
            segmentation_loss, flow_loss, total_loss, segmentation_accuracy, iou_score,  mae = process_batch(batch, model, device, segmentation_criterion, flow_criterion, optimizer,config)

            ############ Store the values ##########################
            seg_loss_accumulator += segmentation_loss.sum().item()
            flow_loss_accumulator += flow_loss.item()
            seg_acc_accumulator += segmentation_accuracy
            mae_accumulator += mae
            iou_accumulator += iou_score
            #dice_score_accumulator += dice_score

            last_sample = batch

            ###Print the loss for every N iterations###
            if batch_idx % 50 == 0:
                print(f'''\nEpoch [{epoch}/{config['num_epochs']}], Iteration [{batch_idx}/{len(dataloader_train)}],
                \nSegmentation Loss: {segmentation_loss.item()},
                \nFlow Loss: {flow_loss.item()},
                \nSegmentation Accuracy: {segmentation_accuracy},            
                \nMAe: {mae},
                \nIoU: {iou_score},
                \nTotal Loss: {total_loss.item()},
                \n''')
            #############################################

            wandb.log({"Training Segmentation Loss": segmentation_loss.item(),
                        "Training Mean Flow Loss": flow_loss,
                        "Training Segmentation Accuracy": segmentation_accuracy})

        # Define a dictionary to accumulate metrics
        metrics_accumulator = {
            "Average Training Segmentation Loss": seg_loss_accumulator,
            "Average Training Flow Loss": flow_loss_accumulator,
            "Average Training Segmentation Accuracy": seg_acc_accumulator,
            "Average Training Mean Absolute Error": mae_accumulator,
            "Average Training IoU": iou_accumulator,
            #"Average Training Dice Coefficient": dice_score_accumulator,
        }

        # Calculate averages and prepare for logging
        metrics_average = {metric: value / len(dataloader_train) for metric, value in metrics_accumulator.items()}

        # Log the averages using wandb
        wandb.log(metrics_average)
        
        save_model_state(model, epoch, save_dir_training)

        # Visualize samples after each epoch
        if last_sample is not None:
            visualize_random_sample(dataloader_train, model, epoch, device,save_dir_training,config)

        print(f"Training reached at epoch:{epoch}, validating the model")
        validate_model(model, dataloader_validation,device, segmentation_criterion, save_dir_validation,epoch, config)

        torch.cuda.empty_cache()

    return segmentation_accuracy_values, mae_of_values, segmentation_loss_values, flow_loss_values, total_loss_values, pixel_difference_values, iou_values#, dice_coefficient_values

'''
def train_model(model, dataloader_train,dataloader_validation, device,  segmentation_criterion, flow_criterion, optimizer, scheduler, save_dir_training, save_dir_validation, model_metrics_values,config, start_epoch):
    # Set up the optimizers
    seg_optimizer = torch.optim.Adam(model.segmentation_branch.parameters(), lr=1e-8)
    flow_optimizer = torch.optim.Adam(model.flow_branch.parameters(), lr=1e-9)

    # Rest of your code...

    for epoch in range(start_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()

        if epoch % 2 == 0:  # Segmentation epoch
            # Freeze the flow branch
            for param in model.flow_branch.parameters():
                param.requires_grad = False

            # Use the segmentation optimizer
            optimizer = seg_optimizer

        else:  # Optical flow epoch
            # Freeze the segmentation branch
            for param in model.segmentation_branch.parameters():
                param.requires_grad = False

            # Use the flow optimizer
            optimizer = flow_optimizer

        # Rest of your code...
'''