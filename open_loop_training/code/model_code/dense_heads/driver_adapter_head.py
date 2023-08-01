# Written by Xiasong Jia
# All Rights Reserved
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from matplotlib import pyplot as plt
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models import HEADS
from mmcv.runner import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models import builder
import numpy as np
import cv2 as cv



def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

def init_weights(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif classname.find('LayerNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('GroupNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('Linear') != -1:
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.xavier_normal_(m.weight)
        elif classname.find('Embedding') != -1:
            nn.init.trunc_normal_(m.weight, mean=0, std=0.02)

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)

class SEModule(nn.Module):
    def __init__(self, channels, act):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = act()
        self.fc2 = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AdapterCNN(nn.Module):
    def __init__(self, inplanes, outplanes, act=nn.ReLU, norm=nn.BatchNorm2d):
        super(AdapterCNN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes//4, kernel_size=3, padding=1,)
        self.act1 = act()
        self.bn1 = norm(inplanes//4)
        self.conv2 = nn.Conv2d(inplanes//4, inplanes, kernel_size=3, padding=1,)
        self.act2 = act()
        self.bn2 = norm(inplanes)
        self.se = SEModule(inplanes, act)
        self.out_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0,)
        self.act3 = act()
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        x += shortcut
        x = self.act2(x)
        x = self.out_conv(x)
        x = self.act3(x)
        return x

class MLP(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 act_layer=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim*2)
        self.act1 = act_layer()
        self.fc2 = nn.Linear(in_dim*2, out_dim)
        self.act2 = act_layer()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x

def tint(color, factor):
    r, g, b = color
    r = int(r + (255-r) * factor)
    g = int(g + (255-g) * factor)
    b = int(b + (255-b) * factor)
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return (r, g, b)



@HEADS.register_module()
class DriveAdapterHead(BaseModule):
    def __init__(self,
                 config=None,
                 bev_seg_head_cfg=None
                 ):

        self.config = config
        self.bev_seg_head_cfg = bev_seg_head_cfg
        self.act = nn.ReLU
        self.view_encoder_channels = 256
        self.norm = nn.BatchNorm2d
            
        super(DriveAdapterHead, self).__init__()

        self.build_layers()
        self.wp_loss_weight = 10.0
        self.action_loss_weight = 2.0
        self.fp16_enabled = False
        self.cnn_feature_loss_weight_list = [50.0, 20.0, 10.0, 5, 2.0, 1.0]
    
    def visualize_roach_input(self, road_mask, route_mask, lane_mask, lane_mask_broken, tl_green_masks, tl_yellow_masks, tl_red_masks, vehicle_masks, walker_masks, name="vis_bev_seg.png"):
        image = np.zeros([192, 192, 3], dtype=np.uint8)
        image[road_mask] = COLOR_ALUMINIUM_5
        image[route_mask] = COLOR_ALUMINIUM_3
        image[lane_mask] = COLOR_MAGENTA
        image[lane_mask_broken] = COLOR_MAGENTA_2
        h_len = 3#len(self._history_idx)-1
        for i, mask in enumerate(tl_green_masks):
            image[mask] = tint(COLOR_GREEN, (h_len-i)*0.2)
        for i, mask in enumerate(tl_yellow_masks):
            image[mask] = tint(COLOR_YELLOW, (h_len-i)*0.2)
        for i, mask in enumerate(tl_red_masks):
            image[mask] = tint(COLOR_RED, (h_len-i)*0.2)
        for i, mask in enumerate(vehicle_masks):
            image[mask] = tint(COLOR_BLUE, (h_len-i)*0.2)
        for i, mask in enumerate(walker_masks):
            image[mask] = tint(COLOR_CYAN, (h_len-i)*0.2)
        ev_mask = np.load("ev_mask.npy")
        image[ev_mask] = COLOR_WHITE
        plt.imshow(image)
        plt.savefig(name)

    ## Transform the BEV Segmentation into Roach's input format
    @force_fp32()
    def to_roach_bev_input(self, bev_seg_logit, route_mask):
        bev_seg_sigmoid = bev_seg_logit.sigmoid()## Road Mask - 0, Lane_Mask -1, broken lane - 2, 4 * vehicle - 3456, 4 * walker 7 8 9 10, green 11 12 13 14, yellow 15 16 17 18, red 19 20 21 22
        bev_seg_threshold = (bev_seg_logit>self.config["bev_seg_threshold"]).float()
        bev_seg_hard = (bev_seg_threshold - bev_seg_sigmoid).detach() + bev_seg_sigmoid ## With gradient
        bev_seg_hard[:, 1, :, :][bev_seg_hard[:, 2, :, :]==1] = 0.5
        bev_seg_hard[:, 11:15, :, :][bev_seg_hard[:, 11:15, :, :]==1] = 0.3137 ## Green Light
        bev_seg_hard[:, 11:15, :, :][bev_seg_hard[:, 15:19, :, :]==1] = 0.6667 ## Yellow Light
        bev_seg_hard[:, 11:15, :, :][bev_seg_hard[:, 19:23, :, :]==1] = 1.0 ## Red Light
        #Road Mask - 0, Route Mask - 1, Lane_Mask (broken lane - 0.5) -2, 4 * vehicle - 3456, 4 * walker 789 10, 4 * traffic_light 11 12 13 14 [-16, -11, -6, -1] 10hz!!!
        return torch.cat([bev_seg_hard[:, :1], route_mask.unsqueeze(1),  bev_seg_hard[:, 1, :, :].unsqueeze(1), bev_seg_hard[:, 3:15]], dim=1)

    def init_roach_head(self, roach_head_dict, freezed_module_name):
        roach_head_dict["features_extractor-cnn"] = nn.ModuleList([
            nn.Conv2d(15, 8, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        ])

        roach_head_dict["features_extractor-state_linear"] = nn.Sequential(
            nn.Linear(6, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 256), 
            nn.ReLU(inplace=True),
        )

        roach_head_dict["features_extractor-linear"] = nn.Sequential(
            nn.Linear(1024+256, 512), 
            nn.ReLU(inplace=True),
            nn.Linear(512, 256), 
            nn.ReLU(inplace=True),
            )
        
        roach_head_dict["policy_head"] = nn.Sequential(
            nn.Linear(256, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 256), 
            nn.ReLU(inplace=True),
        )

        roach_head_dict["value_head"] = nn.Sequential(
            nn.Linear(256, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 1), 
        )

        roach_head_dict["dist_mu"] = nn.Sequential(
            nn.Linear(256, 2),
            nn.Softplus(),
        )
        roach_head_dict["dist_sigma"] = nn.Sequential(
            nn.Linear(256, 2),
            nn.Softplus(),
        )

        self.load_roach_layers(roach_head_dict)
        for name in freezed_module_name:
            for param in roach_head_dict[name].parameters():
                param.requires_grad = False
            
    
    def load_roach_layers(self, module_dict):
        rl_state_dict = torch.load(self.config['rl_ckpt'], map_location='cpu')['policy_state_dict']
        for key, value in rl_state_dict.items():
            tmp = key.split(".")
            name = "-".join(tmp[:-2])
            index = int(tmp[-2])
            attri = tmp[-1]
            if attri == "weight":
                module_dict[name][index].weight = torch.nn.Parameter(value).to(module_dict[name][index].weight.device)
            else:
                module_dict[name][index].bias = torch.nn.Parameter(value).to(module_dict[name][index].bias.device)

    def build_layers(self):
        self.bev_seg_head = builder.build_head(self.bev_seg_head_cfg)
        self.roach_head = nn.ModuleDict()
        self.init_roach_head(self.roach_head, self.config["freezed_module_name"])
        self.adapter_dict = nn.ModuleDict()
        if "adapter_module" in self.config and len(self.config["adapter_module"]) != 0:
            self.adapter_downsample_raw_feat = nn.ModuleList([
                        nn.Conv2d(self.view_encoder_channels + 1, self.view_encoder_channels, kernel_size=5, stride=2),
                        self.norm(self.view_encoder_channels),
                        self.act(),
                        nn.Conv2d(self.view_encoder_channels, self.view_encoder_channels, kernel_size=5, stride=2),
                        self.norm(self.view_encoder_channels),
                        self.act(),
                        nn.Conv2d(self.view_encoder_channels, self.view_encoder_channels * 2, kernel_size=5, stride=2),
                        self.norm(self.view_encoder_channels * 2),
                        self.act(),
                        nn.Conv2d(self.view_encoder_channels * 2, self.view_encoder_channels * 2, kernel_size=3, stride=2),
                        self.norm(self.view_encoder_channels * 2),
                        self.act(),
                        nn.Conv2d(self.view_encoder_channels * 2, self.view_encoder_channels * 4, kernel_size=3, stride=2),
                        self.norm(self.view_encoder_channels * 4),
                        self.act(),
                        nn.Conv2d(self.view_encoder_channels * 4, self.view_encoder_channels * 4, kernel_size=3, stride=1),
                        self.norm(self.view_encoder_channels * 4),
                        self.act(),
                    ])
            self.adapter_flatten_mlp = nn.Linear(self.view_encoder_channels*4*4, self.view_encoder_channels*4)
            if "features_extractor-cnn" in self.config["adapter_module"]:
                self.adapter_dict["features_extractor-cnn"] = nn.ModuleList([
                    AdapterCNN(inplanes=8+self.view_encoder_channels, outplanes=8, act=self.act, norm=self.norm),
                    AdapterCNN(inplanes=16+self.view_encoder_channels, outplanes=16, act=self.act, norm=self.norm),
                    AdapterCNN(inplanes=32+self.view_encoder_channels*2, outplanes=32, act=self.act, norm=self.norm),
                    AdapterCNN(inplanes=64+self.view_encoder_channels*2, outplanes=64, act=self.act, norm=self.norm),
                    AdapterCNN(inplanes=128+self.view_encoder_channels*4, outplanes=128, act=self.act, norm=self.norm),
                    AdapterCNN(inplanes=256+self.view_encoder_channels*4, outplanes=256, act=self.act, norm=self.norm),
                ])
            if "features_extractor-linear" in self.config["adapter_module"]:
                self.adapter_dict["features_extractor-linear"] = MLP(256+self.view_encoder_channels*4, 256)
            if "policy_head" in self.config["adapter_module"]:
                self.adapter_dict["policy_head"] = MLP(256+self.view_encoder_channels*4, 256)
        
        if "use_traj_head" in self.config and self.config["use_traj_head"]:
            self.traj_head = nn.Sequential(
                nn.Linear(self.view_encoder_channels*4+256, 512),
                self.act(),
                nn.Linear(512, 256),
                self.act(),
                nn.Linear(256, 2*self.config.pred_len)
            )
            
    @auto_fp16()
    def roach_forward(self, roach_module_dict, bev_feat, state_list):
        bev_mid_feature = bev_feat
        bev_mid_feature_lis = []
        for i in range(0, len(roach_module_dict["features_extractor-cnn"]), 2):
            bev_mid_feature = roach_module_dict["features_extractor-cnn"][i+1](roach_module_dict["features_extractor-cnn"][i](bev_mid_feature))
            bev_mid_feature_lis.append(bev_mid_feature)
        flattened_bev_feature = bev_mid_feature.flatten(start_dim=1)

        latent_state = roach_module_dict["features_extractor-state_linear"](state_list)
        latent_state = roach_module_dict["features_extractor-linear"](torch.cat([flattened_bev_feature, latent_state], dim=-1))

        values = roach_module_dict["value_head"](latent_state)
        latent_pi = roach_module_dict["policy_head"](latent_state)
        mu = roach_module_dict["dist_mu"](latent_pi)
        sigma = roach_module_dict["dist_sigma"](latent_pi)
        return {
            "bev_mid_feature_lis":bev_mid_feature_lis,
            "flattened_bev_feature":flattened_bev_feature,
            "latent_state":latent_state,
            "values":values,
            "latent_pi":latent_pi,
            "mu":mu,
            "sigma":sigma,
        }

    @auto_fp16()
    def roach_with_adapter_forward(self, roach_module_dict, bev_seg, state_list, bev_raw_feat):
        bs = bev_raw_feat.shape[0]
        bev_raw_feat_lis = []
        for i in range(0, len(self.adapter_downsample_raw_feat), 3):
            bev_raw_feat = self.adapter_downsample_raw_feat[i+2](self.adapter_downsample_raw_feat[i+1](self.adapter_downsample_raw_feat[i](bev_raw_feat)))
            bev_raw_feat_lis.append(bev_raw_feat)
        
        bev_mid_feature = bev_seg
        bev_mid_feature_lis = []
        for i in range(0, len(roach_module_dict["features_extractor-cnn"]), 2):
            bev_mid_feature = roach_module_dict["features_extractor-cnn"][i+1](roach_module_dict["features_extractor-cnn"][i](bev_mid_feature))
            if "features_extractor-cnn" in self.config["adapter_module"]:
                bev_mid_feature = self.adapter_dict["features_extractor-cnn"][i//2](torch.cat([bev_mid_feature, bev_raw_feat_lis[i//2]], dim=1))
            bev_mid_feature_lis.append(bev_mid_feature)
        flattened_bev_feature = bev_mid_feature.flatten(start_dim=1)

        latent_state_raw = roach_module_dict["features_extractor-state_linear"](state_list)
        latent_state = roach_module_dict["features_extractor-linear"](torch.cat([flattened_bev_feature, latent_state_raw], dim=-1))
        if "features_extractor-linear" in self.config["adapter_module"] or "policy_head" in self.config["adapter_module"]:
            flattened_bev_raw_feat = self.adapter_flatten_mlp(bev_raw_feat.flatten(start_dim=1))
        if "features_extractor-linear" in self.config["adapter_module"]:
            latent_state = self.adapter_dict["features_extractor-linear"](torch.cat([latent_state, flattened_bev_raw_feat], dim=-1))
        values = roach_module_dict["value_head"](latent_state)
        latent_pi = roach_module_dict["policy_head"](latent_state)
        if "policy_head" in self.config["adapter_module"]:
            latent_pi =  self.adapter_dict["policy_head"](torch.cat([latent_pi, flattened_bev_raw_feat], dim=-1))

        mu = self.fp32_forward(roach_module_dict["dist_mu"], latent_pi)
        sigma = self.fp32_forward(roach_module_dict["dist_sigma"], latent_pi)
        res_dict = {
            "bev_mid_feature_lis":bev_mid_feature_lis,
            "flattened_bev_feature":flattened_bev_feature,
            "latent_state":latent_state,
            "values":values,
            "latent_pi":latent_pi,
            "mu":mu,
            "sigma":sigma,
        }
        if "use_traj_head" in self.config and self.config["use_traj_head"]:
            res_dict["pred_wp"] = self.fp32_forward(self.traj_head, torch.cat([latent_state_raw.detach(), flattened_bev_raw_feat], dim=-1)).view(bs, self.config.pred_len, 2)
        return res_dict
    
    @force_fp32()
    def fp32_forward(self, module, input):
        return module(input)
    
    @force_fp32()
    def _get_action_beta(self, alpha, beta):
        x = torch.zeros_like(alpha)
        x[:, 1] += 0.5
        mask1 = (alpha > 1) & (beta > 1)
        x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)
        mask2 = (alpha <= 1) & (beta > 1)
        x[mask2] = 0.0
        mask3 = (alpha > 1) & (beta <= 1)
        x[mask3] = 1.0
        # mean
        mask4 = (alpha <= 1) & (beta <= 1)
        x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)
        x = x * 2 - 1
        return x
    
    def forward(self, bev_feat, route_mask, state_list):
        bev_seg_logit = self.bev_seg_head(bev_feat) ## Road Mask - 0, Lane_Mask -1, broken lane - 2, 4 * vehicle - 3456, 4 * walker 7 8 9 10, green 11 12 13 14, yellow 15 16 17 18, red 19 20 21 22
        roach_bev_input = self.to_roach_bev_input(bev_seg_logit, route_mask)

        if "adapter_module" in self.config and len(self.config["adapter_module"]) != 0:
            bev_feat = torch.cat([bev_feat, route_mask.unsqueeze(1)], dim=1)
            res_dict = self.roach_with_adapter_forward(self.roach_head, roach_bev_input, state_list, bev_feat)
        else:
            res_dict = self.roach_forward(self.roach_head, roach_bev_input, state_list) ## No adapter
        res_dict["bev_seg_logit"] = bev_seg_logit
        res_dict["roach_bev_input"] = roach_bev_input
        return res_dict
    
    @force_fp32()
    def loss(self,
             batch,
             pred,):
        loss_dict = dict()
        gt_waypoints = batch['waypoints']

        ## Calculate Offset
        with torch.no_grad():
            pred_action = self._get_action_beta(pred['mu'], pred['sigma'])
            gt_action = self._get_action_beta(batch['action_mu'], batch['action_sigma'])
            l1_action = F.l1_loss(pred_action, gt_action, reduction="none").detach().mean(dim=0)
            loss_dict["current_throttle_brake_offset"] = l1_action[0]
            loss_dict['current_steer_offset'] = l1_action[1]
            ## Turning Scene - Larger Weight
            focal_weight = (1+gt_action[:, 1].abs())**3/2.0
            if "mask_ap_brake" in self.config and self.config["mask_ap_brake"]:
                ap_mask = (~batch["only_ap_brake"]).float()
            else:
                ap_mask = torch.ones_like(focal_weight)
        
        if "action" in self.config["head_loss"]:
            dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
            dist_pred = Beta(pred['mu'], pred['sigma'])
            kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
            action_loss = torch.mean(kl_div[..., 0] * focal_weight) *0.5 + torch.mean(kl_div[..., 1] * focal_weight.unsqueeze(1)) *0.5
            loss_dict["action_loss"] = action_loss * self.action_loss_weight
            
        if "bev_seg" in self.config["head_loss"]:
            ## No prediction for route
            bev_seg_label = torch.cat([batch["bev_seg_label"][:, :1, :, :], batch["bev_seg_label"][:, 2:, :, :]], dim=1)

            ###Visualization
            if False:
                bev_seg_label_numpy_bool = (batch["bev_seg_label"].cpu().numpy()==1)
                b = bev_seg_label_numpy_bool[0]
                ## Road Mask - 0, Route Mask - 1, Lane_Mask -2, broken lane - 3, 4 * vehicle - 4567, 4 * walker 8 9 10 11, green 12 13 14 15, yellow 16 17 18 19, red 20 21 22 23
                self.visualize_roach_input(road_mask=b[0], route_mask=b[1], lane_mask=b[2], lane_mask_broken=b[3], tl_green_masks=b[12:16], tl_yellow_masks=b[16:20], tl_red_masks=b[20:24], vehicle_masks=b[4:8], walker_masks=b[8:12])
                import ipdb
                ipdb.set_trace()
            ## Segmentation Loss Per Sample (No reduction for batch!!!)
            bev_seg_loss = self.bev_seg_head.loss(pred["bev_seg_logit"], bev_seg_label)
            loss_dict["bev_seg_dice_loss"] = bev_seg_loss['bev_seg_dice_loss']
            loss_dict["bev_seg_mask_loss"] = bev_seg_loss['bev_seg_mask_loss']
            with torch.no_grad():
                #with torch.no_grad():
                bev_seg_pred_bool = pred["bev_seg_logit"].detach().sigmoid()>0.5
                bev_seg_labe_bool = bev_seg_label.bool()
                iou = calculate_IoU(bev_seg_pred_bool, bev_seg_labe_bool)
                whehther_have_class = bev_seg_labe_bool.any(-1).any(-1).float()
                #(iou*whehther_have_class).sum()/whehther_have_class.sum()
                loss_dict["mIoU"] = ((iou*whehther_have_class).sum(dim=1)+1e-6)/(whehther_have_class.sum(dim=1)+1e-6).mean()

        for each_loss_type in self.config["head_loss"]:
            if "cnn_feature" in each_loss_type:
                cnn_feature_index = int(each_loss_type[-1])
                loss_dict["cnn_feature_loss"+str(cnn_feature_index)] = (F.smooth_l1_loss(pred["bev_mid_feature_lis"][cnn_feature_index], batch["grid_feature"][cnn_feature_index], reduction="none").mean(dim=(1, 2, 3)) * ap_mask).mean() * self.cnn_feature_loss_weight_list[cnn_feature_index]
        if "latent_state" in self.config["head_loss"]:
            loss_dict["latent_state_loss"] = (F.smooth_l1_loss(pred["latent_state"], batch["feature"], reduction="none").mean(-1)*ap_mask).mean() * 2.5
        if "future_action" in self.config["head_loss"]:
            future_action_loss = 0
            gt_future_action_mu = torch.stack(batch['future_action_mu'][:-1], axis=1)
            gt_future_action_sigma = torch.stack(batch['future_action_sigma'][:-1], axis=1)
            dist_sup = Beta(gt_future_action_mu, gt_future_action_sigma)
            dist_pred = Beta(pred['future_mu'], pred['future_sigma'])
            kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
            future_action_loss += torch.mean(kl_div[..., 0] * focal_weight.unsqueeze(1)) * 0.5 + torch.mean(kl_div[..., 1] * focal_weight.unsqueeze(1)) * 0.5
            loss_dict["future_action_loss"] = future_action_loss * 5.0
        if "traj" in self.config["head_loss"]:
            with torch.no_grad():
                wp_offset = F.l1_loss(pred['pred_wp'], gt_waypoints[:, :, :],reduction="none").detach().mean(dim=0)
                mean_wp_offset = wp_offset.mean(dim=0) 
                loss_dict["current_longitudinal_offset"] = mean_wp_offset[0] 
                loss_dict["current_lateral_offset"] = mean_wp_offset[1]
            loss_dict["wp_loss"] = (F.smooth_l1_loss(pred['pred_wp'], gt_waypoints, reduction="none") * focal_weight.unsqueeze(1).unsqueeze(1)).mean() * 10
        return loss_dict
    

def calculate_IoU(pred, gt):
    intersection = (pred & gt).float().sum((2, 3))  # Will be zero if Truth=0 or Prediction=0
    union = (pred | gt).float().sum((2, 3))         # Will be zzero if both are 0
    smooth = 1e-6
    return (intersection + smooth) / (union + smooth)
