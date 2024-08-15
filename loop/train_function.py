import torch
import numpy as np
from utils.utils_metric.utils import iou, dice_coefficient
from utils.utils import visualize_random_sample
import os
from loop.validation_function import validate_model
import wandb
from utils.utils_metric.utils import Metrics, FlowMetrics

def train_model(model, dataloader_train,dataloader_validation, num_epochs, device,  segmentation_criterion, flow_criterion, optimizer, scheduler, save_dir_training, save_dir_validation, model_metrics_values,config):
    segmentation_loss_values = []
    flow_loss_values = []
    total_loss_values = []
    segmentation_accuracy_values = []
    mae_of_values = []
    pixel_difference_values = []
    last_sample = None
    iou_values = []
    dice_coefficient_values = []
    checkpoint_interval = 10

    for epoch in range(num_epochs):
        
        torch.cuda.empty_cache()
        model.train()
        seg_loss_accumulator = 0
        flow_loss_accumulator = 0
        seg_acc_accumulator = 0
        mae_accumulator = 0
        iou_accumulator = 0
        dice_score_accumulator = 0

        for batch_idx, batch in enumerate(dataloader_train):
            torch.cuda.empty_cache()
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
            # Compute the difference between the predicted and true segmentation


            seg_metrics = Metrics(config['num_classes'])
            seg_metrics.update(seg_predictions, segmentation_mask)
            segmentation_accuracy = seg_metrics.get_results('overall_acc')
            iou_score = seg_metrics.get_results('mIou')
            dice_score = seg_metrics.get_results('dice_score')

            accuracy_threshold = 0.85
            best_accuracy = 0.0
            # Log all accuracy values above the threshold
            if segmentation_accuracy > accuracy_threshold:
                with open(os.path.join(os.getcwd(), save_dir_training, 'accuracy.txt'), 'a') as f:
                    f.write(f'Accuracy: {segmentation_accuracy} at epoch {epoch} batch {batch_idx}\n')

            # Save the model's state only if the current accuracy is the best
            if segmentation_accuracy > best_accuracy:
                best_accuracy = segmentation_accuracy  # Update best accuracy

                # Save the model's state
                torch.save(model.state_dict(), save_dir_training + f'/model_best_at_{epoch}_epoch.pth')

            # Compute the absolute difference between the predicted and true flows
            flow_metrics = FlowMetrics()
            flow_metrics.update(flow_result_add, flow_mask)
            mae = flow_metrics.get_results()

            #############################################################
            segmentation_mask = segmentation_mask.squeeze(1).long().to(device)
            # Add the assertion check here
            n_classes = 6  # Replace with the actual number of classes
            assert segmentation_mask.min() >= 0 and segmentation_mask.max() < n_classes, \
                f"Target values out of range: {segmentation_mask.min()} - {segmentation_mask.max()}"
            ######### Loss Calculation ##################################

            segmentation_loss = segmentation_criterion(seg_result, segmentation_mask)
            flow_loss = flow_criterion(flow_result_add, flow_mask)
            total_loss = segmentation_loss + flow_loss

            ############ Store the values ##########################
            seg_loss_accumulator += segmentation_loss.sum().item()
            flow_loss_accumulator += flow_loss.item()
            seg_acc_accumulator += segmentation_accuracy
            mae_accumulator += mae
            iou_accumulator += iou_score
            dice_score_accumulator += dice_score
            ############ Backpropagation ##########################################
            
            start_mem = torch.cuda.memory_allocated()

            optimizer.zero_grad()

            total_loss.backward() # 7gb, 5gb
            optimizer.step()
            end_mem = torch.cuda.memory_allocated()
            print(f'Memory used: {(end_mem - start_mem) / 1024**2} MB')

            last_sample = batch

            ###Print the loss for every N iterations###
            if batch_idx % 50 == 0:
                print(f'''\nEpoch [{epoch}/{num_epochs}], Iteration [{batch_idx}/{len(dataloader_train)}],
                \nSegmentation Loss: {segmentation_loss.item()},
                \nFlow Loss: {flow_loss.item()},
                \nSegmentation Accuracy: {segmentation_accuracy},            
                \nMAe: {mae},
                \nIoU: {iou_score},
                \nDice Coefficient: {dice_score},
                \nTotal Loss: {total_loss.item()},
                \n''')
            #############################################

            wandb.log({"Training Segmentation Loss": segmentation_loss.item(),
                        "Training Mean Flow Loss": flow_loss,
                        "Training Segmentation Accuracy": segmentation_accuracy})


        avg_seg_loss = seg_loss_accumulator / len(dataloader_train)
        avg_flow_loss = flow_loss_accumulator / len(dataloader_train)
        avg_seg_acc = seg_acc_accumulator / len(dataloader_train)
        avg_mae = mae_accumulator / len(dataloader_train)
        avg_iou = iou_accumulator / len(dataloader_train)
        avg_dice_score = dice_score_accumulator / len(dataloader_train)

        # Log the averages using wandb
        wandb.log({
            "Average Training Segmentation Loss": avg_seg_loss,
            "Average Training Flow Loss": avg_flow_loss,
            "Average Training Segmentation Accuracy": avg_seg_acc,
            "Average Training Mean Absolute Error": avg_mae,
            "Average Training IoU": avg_iou,
            "Average Training Dice Coefficient": avg_dice_score,
                })
        
        if epoch % checkpoint_interval == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, save_dir_training + f'/model_at_{epoch}_epoch.pth')



        # Visualize samples after each epoch

        if last_sample is not None:
            visualize_random_sample(dataloader_train, model, epoch, device,save_dir_training)

        #if epoch % 10 == 0:
        print(f"Training reached at epoch:{epoch}, validating the model")

        validate_model(model, dataloader_validation,device, segmentation_criterion, save_dir_validation,epoch,config)





        #scheduler.step()

        torch.cuda.empty_cache()

    return segmentation_accuracy_values, mae_of_values, segmentation_loss_values, flow_loss_values, total_loss_values, pixel_difference_values, iou_values, dice_coefficient_values