import torch
import torch.nn as nn
from utils.utils_model.utils import conv, predict_flow, deconv, crop_like
from models.resnet101 import ResNet, Bottleneck
from models.flownets import FlowNetS
import torch.utils.checkpoint as cp
'''
Model Architecture:

1. Segmentation Branch:
   - Upsample seg_cat1_deconv by a factor of 16.
   - Downsample flow_outputs[1] (flow3) by a factor of 2.
   - Concatenate downsampled flow3 with layers[1] (layer3).
   - Deconvolve the concatenated output.
   - Upsample the deconvolved output by a factor of 8.
   - Downsample flow_outputs[0] (flow2) by a factor of 2.
   - Concatenate downsampled flow2 with layers[0] (layer2).
   - Deconvolve the concatenated output.
   - Upsample the deconvolved output by a factor of 4.
   - Concatenate the upsampled outputs from the above steps.
   - Deconvolve the concatenated output.
   - Apply final segmentation layers.

2. Flow Branch:
   - Upsample layers[2] (layer4) by a factor of 2.
   - Concatenate flow_outputs[2] (flow4) with upsampled layer4.
   - Upsample the concatenated output by a factor of 16.
   - Predict flow from the upsampled output.
   - Upsample layers[1] (layer3) by a factor of 2.
   - Concatenate flow_outputs[1] (flow3) with upsampled layer3.
   - Upsample the concatenated output by a factor of 8.
   - Predict flow from the upsampled output.
   - Upsample layers[0] (layer2) by a factor of 2.
   - Concatenate flow_outputs[0] (flow2) with upsampled layer2.
   - Upsample the concatenated output by a factor of 4.

'''

def ResNet101(channels=3):
    return ResNet(Bottleneck, [3,4,23,3], channels)


class SilkiMaus(nn.Module):
    def __init__(self, num_classes):
        super(SilkiMaus, self).__init__()
        self.flownet = FlowNetS()
        self.resnet = ResNet101(channels=3)


        self.predict_flow4_cat1 = predict_flow(514)
        self.predict_flow3_cat2 = predict_flow(258)
        self.predict_flow2_cat3 = predict_flow(130) 
        self.predict_flow_result = predict_flow(6)

        self.Downsample1 = nn.MaxPool2d(kernel_size=2) # This is for flow outputs
        self.Downsample2 = nn.MaxPool2d(kernel_size=4) # 
        self.Downsample3 = nn.MaxPool2d(kernel_size=8)

        self.Upsample5 = nn.Upsample(scale_factor=32) # This is for seg_cat1
        self.Upsample4 = nn.Upsample(scale_factor=16) # This is for seg_cat2
        self.Upsample3 = nn.Upsample(scale_factor=8) # This is for seg_cat3
        self.Upsample2 = nn.Upsample(scale_factor=4)
        self.Upsample1 = nn.Upsample(scale_factor=2)


        self.seg_result_conv = nn.Sequential(
            nn.Conv2d(224, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.LeakyReLU(0.01,inplace=True)
        )
        self.cat1_deconv = deconv(514, 256)

        self.cat2_deconv = deconv(258, 128)

        self.cat3_deconv = deconv(130, 64)

        self.seg_result_deconv = nn.Sequential(
                                nn.ConvTranspose2d(448, 224, kernel_size=3, stride=1, padding= 1, bias=False),
                                nn.LeakyReLU(0.1,inplace=True)
                                            )
    def checkpointed_operations_cat1(self, x):
        # Define a sequence of operations to checkpoint for seg_cat1
        x_deconv = self.cat1_deconv(x)
        x_up = self.Upsample4(x_deconv)
        return x_up

    def checkpointed_operations_cat2(self, x):
        # Define a sequence of operations to checkpoint for seg_cat2
        x_deconv = self.cat2_deconv(x)
        x_up = self.Upsample3(x_deconv)
        return x_up

    def checkpointed_operations_cat3(self, x):
        # Define a sequence of operations to checkpoint for seg_cat3
        x_deconv = self.cat3_deconv(x)
        x_up = self.Upsample2(x_deconv)
        return x_up
    

    def forward(self, input_resnet, input_flownets):


        flow_outputs = self.flownet(input_flownets)

        layers= self.resnet(input_resnet)

        #print(f'\n--------------MODEL RECEIVED INPUTS-------------------\n')

    ########################################################################################
        # First downsample the flow4 3rd index output by 1/2, which was 1/16 of the original input
        # Then concatenate with the layer4 3rd index output, which was 1/32 of the original input
        # Then deconvolve the concatenated output and upsample the result by *16
        flow4_down = self.Downsample1(flow_outputs[2]) # Downsample by 1/2
        seg_cat1 = torch.cat((flow4_down, layers[2]), dim=1)
        seg_cat1_deconv = self.cat1_deconv(seg_cat1)
        seg_cat1_up = self.Upsample4(seg_cat1_deconv) #1 gb increased
        #print(f'\n--------------SEGMENTATION BRANCH-------------------\n')

        # First downsample the flow3 2nd index output by 1/2, which was 1/8 of the original input
        # Then concatenate with the layer3 2nd index output, which was 1/16 of the original input
        # Then deconvolve the concatenated output and upsample the result by *8
        flow3_down = self.Downsample1(flow_outputs[1]) # LAYER3
        seg_cat2 = torch.cat((flow3_down, layers[1]), dim=1)
        seg_cat2_deconv = self.cat2_deconv(seg_cat2)
        seg_cat2_up = self.Upsample3(seg_cat2_deconv)

        # First downsample the flow2 1st index output by 1/2, which was 1/4 of the original input
        # Then concatenate with the layer2 1st index output, which was 1/8 of the original input
        # Then deconvolve the concatenated output and upsample the result by *4
        flow2_down = self.Downsample1(flow_outputs[0]) #LAYER2
        seg_cat3 = torch.cat((flow2_down, layers[0]), dim=1)
        seg_cat3_deconv = self.cat3_deconv(seg_cat3)
        seg_cat3_up = self.Upsample2(seg_cat3_deconv)

        # # Gradient Check Point
        # seg_cat1_up = cp.checkpoint(self.checkpointed_operations_cat1, seg_cat1, use_reentrant=False)
        # seg_cat2_up = cp.checkpoint(self.checkpointed_operations_cat2, seg_cat2, use_reentrant=False)
        # seg_cat3_up = cp.checkpoint(self.checkpointed_operations_cat3, seg_cat3, use_reentrant=False)

        seg_result_cat = torch.cat((seg_cat1_up, seg_cat2_up,seg_cat3_up), dim=1) # 2gb incresed
        seg_result_deconv = self.seg_result_deconv(seg_result_cat)
        seg_result = self.seg_result_conv(seg_result_deconv)


############################################################################################        
        #print(f'\n--------------FLOW BRANCH-------------------\n')
        
        # Flow Branch 
        # 1/16 all of them
        layer4_up = self.Upsample1(layers[2])
        flow4_cat1 = torch.cat((flow_outputs[2], layer4_up), dim=1)
        flow4_cat1_up = self.Upsample4(flow4_cat1)

        flow4_cat1_up_predict = self.predict_flow4_cat1(flow4_cat1_up)


        layer3_up = self.Upsample1(layers[1])
        flow3_cat2 = torch.cat((flow_outputs[1], layer3_up), dim=1)
        flow3_cat2_up = self.Upsample3(flow3_cat2) # 2gb increased

        flow3_cat2_up_predict = self.predict_flow3_cat2(flow3_cat2_up)
        
        layer2_up = self.Upsample1(layers[0])
        flow2_cat3 = torch.cat((flow_outputs[0], layer2_up), dim=1)
        flow2_cat3_up = self.Upsample2(flow2_cat3)

        flow2_cat3_up_predict = self.predict_flow2_cat3(flow2_cat3_up)


        flow_result_add = flow2_cat3_up_predict + flow3_cat2_up_predict + flow4_cat1_up_predict
        #print(f'\n--------------MODEL FINISHED-------------------\n')

        return seg_result, flow_result_add
    










class SilkiMaus_VAL(nn.Module):
    def __init__(self, num_classes):
        super(SilkiMaus_VAL, self).__init__()
        self.flownet = FlowNetS()
        self.resnet = ResNet101(channels=3)


        self.predict_flow4_cat1 = predict_flow(514)
        self.predict_flow3_cat2 = predict_flow(258)
        self.predict_flow2_cat3 = predict_flow(130) 
        self.predict_flow_result = predict_flow(6)

        self.Downsample1 = nn.MaxPool2d(kernel_size=2) # This is for flow outputs
        self.Downsample2 = nn.MaxPool2d(kernel_size=4) # 
        self.Downsample3 = nn.MaxPool2d(kernel_size=8)

        self.Upsample5 = nn.Upsample(scale_factor=32) # This is for seg_cat1
        self.Upsample4 = nn.Upsample(scale_factor=16) # This is for seg_cat2
        self.Upsample3 = nn.Upsample(scale_factor=8) # This is for seg_cat3
        self.Upsample2 = nn.Upsample(scale_factor=4)
        self.Upsample1 = nn.Upsample(scale_factor=2)


        self.seg_result_conv = nn.Sequential(
            nn.Conv2d(224, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.LeakyReLU(0.01,inplace=True)
        )
        self.cat1_deconv = deconv(514, 256)

        self.cat2_deconv = deconv(258, 128)

        self.cat3_deconv = deconv(130, 64)

        self.seg_result_deconv = nn.Sequential(
                                nn.ConvTranspose2d(448, 224, kernel_size=3, stride=1, padding= 1, bias=False),
                                nn.LeakyReLU(0.1,inplace=True)
                                            )
    def checkpointed_operations_cat1(self, x):
        # Define a sequence of operations to checkpoint for seg_cat1
        x_deconv = self.cat1_deconv(x)
        x_up = self.Upsample4(x_deconv)
        return x_up

    def checkpointed_operations_cat2(self, x):
        # Define a sequence of operations to checkpoint for seg_cat2
        x_deconv = self.cat2_deconv(x)
        x_up = self.Upsample3(x_deconv)
        return x_up

    def checkpointed_operations_cat3(self, x):
        # Define a sequence of operations to checkpoint for seg_cat3
        x_deconv = self.cat3_deconv(x)
        x_up = self.Upsample2(x_deconv)
        return x_up
    

    def forward(self, input_resnet, input_flownets):


        flow_outputs = self.flownet(input_flownets)

        layers= self.resnet(input_resnet)

        print(f'\n--------------MODEL RECEIVED INPUTS-------------------\n')

    ########################################################################################
        # First downsample the flow4 3rd index output by 1/2, which was 1/16 of the original input
        # Then concatenate with the layer4 3rd index output, which was 1/32 of the original input
        # Then deconvolve the concatenated output and upsample the result by *16
        flow4_down = self.Downsample1(flow_outputs[2]) # Downsample by 1/2
        seg_cat1 = torch.cat((flow4_down, layers[2]), dim=1)
        seg_cat1_deconv = self.cat1_deconv(seg_cat1)
        seg_cat1_up = self.Upsample4(seg_cat1_deconv) #1 gb increased
        print(f'\n--------------SEGMENTATION BRANCH-------------------\n')

        # First downsample the flow3 2nd index output by 1/2, which was 1/8 of the original input
        # Then concatenate with the layer3 2nd index output, which was 1/16 of the original input
        # Then deconvolve the concatenated output and upsample the result by *8
        flow3_down = self.Downsample1(flow_outputs[1]) # LAYER3
        seg_cat2 = torch.cat((flow3_down, layers[1]), dim=1)
        seg_cat2_deconv = self.cat2_deconv(seg_cat2)
        seg_cat2_up = self.Upsample3(seg_cat2_deconv)

        # First downsample the flow2 1st index output by 1/2, which was 1/4 of the original input
        # Then concatenate with the layer2 1st index output, which was 1/8 of the original input
        # Then deconvolve the concatenated output and upsample the result by *4
        flow2_down = self.Downsample1(flow_outputs[0]) #LAYER2
        seg_cat3 = torch.cat((flow2_down, layers[0]), dim=1)
        seg_cat3_deconv = self.cat3_deconv(seg_cat3)
        seg_cat3_up = self.Upsample2(seg_cat3_deconv)

        # # Gradient Check Point
        # seg_cat1_up = cp.checkpoint(self.checkpointed_operations_cat1, seg_cat1, use_reentrant=False)
        # seg_cat2_up = cp.checkpoint(self.checkpointed_operations_cat2, seg_cat2, use_reentrant=False)
        # seg_cat3_up = cp.checkpoint(self.checkpointed_operations_cat3, seg_cat3, use_reentrant=False)

        seg_result_cat = torch.cat((seg_cat1_up, seg_cat2_up,seg_cat3_up), dim=1) # 2gb incresed
        seg_result_deconv = self.seg_result_deconv(seg_result_cat)
        seg_result = self.seg_result_conv(seg_result_deconv)


############################################################################################        
        print(f'\n--------------FLOW BRANCH-------------------\n')
        
        # Flow Branch 
        # 1/16 all of them
        layer4_up = self.Upsample1(layers[2])
        flow4_cat1 = torch.cat((flow_outputs[2], layer4_up), dim=1)
        flow4_cat1_up = self.Upsample4(flow4_cat1)

        flow4_cat1_up_predict = self.predict_flow4_cat1(flow4_cat1_up)


        layer3_up = self.Upsample1(layers[1])
        flow3_cat2 = torch.cat((flow_outputs[1], layer3_up), dim=1)
        flow3_cat2_up = self.Upsample3(flow3_cat2) # 2gb increased

        flow3_cat2_up_predict = self.predict_flow3_cat2(flow3_cat2_up)
        
        layer2_up = self.Upsample1(layers[0])
        flow2_cat3 = torch.cat((flow_outputs[0], layer2_up), dim=1)
        flow2_cat3_up = self.Upsample2(flow2_cat3)

        flow2_cat3_up_predict = self.predict_flow2_cat3(flow2_cat3_up)


        flow_result_add = flow2_cat3_up_predict + flow3_cat2_up_predict + flow4_cat1_up_predict
        print(f'\n--------------MODEL FINISHED-------------------\n')

        return seg_result, flow_result_add

