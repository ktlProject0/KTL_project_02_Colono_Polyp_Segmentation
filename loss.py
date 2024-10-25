import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceChannelLoss(nn.Module):
    def __init__(self):
        super(DiceChannelLoss, self).__init__()

    def forward(self, pred, target, smooth=1e-9 ,weights_apply=False):
        
        pred = F.softmax(pred,dim=1) # batch,channel,h,w
        
        num_channels = pred.shape[1]
        dice = torch.zeros(num_channels, device=pred.device)
        
        for i in range(num_channels):
            pred_channel = pred[:, i]
            target_channel = target[:, i]
            if len(pred_channel.shape)==3:
                intersection = (pred_channel * target_channel).sum(dim=(0, 1, 2))
                dice_coeff = (2. * intersection + smooth) / (pred_channel.sum(dim=(0, 1, 2)) + target_channel.sum(dim=(0, 1, 2)) + smooth)
            else:
                intersection = (pred_channel * target_channel).sum(dim=(0, 1, 2,3))
                dice_coeff = (2. * intersection + smooth) / (pred_channel.sum(dim=(0, 1, 2,3)) + target_channel.sum(dim=(0, 1, 2,3)) + smooth)
                
            dice[i] = 1 - dice_coeff

        # Apply weight to the Dice Loss based on epoch_dice value
        if weights_apply:
            weights = (dice/torch.sum(dice))
            dice = dice * weights.to(pred.device)

        dice_loss = torch.exp(dice).sum() # weight automatically
        
        del pred,pred_channel,target_channel,intersection,dice_coeff
        torch.cuda.empty_cache()
        
        return dice, dice_loss