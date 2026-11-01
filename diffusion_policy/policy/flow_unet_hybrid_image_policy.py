from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        self.num_rep = 1
        self.separate = False
        self.global_eps = None
        tasks = {10: 'square', 20: 'transport', 10: 'tool_hang', 10: 'lift', 10: 'can'}
        self.task = tasks[action_dim]
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )
        self.slices = {
            'square': {
                'visual': [(0, 64), (71, 135)], 
                'non_visual': [(64, 67), (67, 71), (135, 137)]
            },
            'transport': {
                'visual': [(7, 71), (80, 144), (146, 210), (210, 274)], 
                'non_visual': [(0, 3), (3, 7), (71, 73), (73, 76), (76, 80), (144, 146)]
            }
        }
        self.slices['tool_hang'] = self.slices['square']
        self.slices['lift'] = self.slices['square']
        self.slices['can'] = self.slices['square']
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']  # 视觉编码器（来自 robomimic，处理图像+低维观测） obs_encoder 里有多个子模块，分别处理不同模态的输入，最终把特征拼在一起 
                                                                         # obs_encoder.obs_nets['agentview_image'] 是处理图像输入的编码器，obs_encoder.obs_nets['low_dim'] 是处理低维观测的编码器
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D( # 1D UNet 模型，输入是动作序列（和可能的条件），输出是动作序列的噪声预测
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler  # 噪声调度器，定义了扩散过程中的噪声添加和去除方式 [控制扩散/流的时间步]
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.global_eps = None # for log_prob
        self.sigma = 0.001 # for log_prob
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference (含密度估计) ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        # 从噪声出发，沿ODE方向积分到动作
        trajectory = torch.randn(   # 从高斯噪声出发
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        start, end, sign = 1, 0, -1
        timesteps = torch.linspace(start, end, self.num_inference_steps+1, device=condition_data.device)
        timesteps = (timesteps * scheduler.config.num_train_timesteps).long()
        
        # Compute log density of x0 with normal distribution
        log_p_x0 = -0.5 * math.log(2 * math.pi) - 0.5 * trajectory**2
        log_p_x0 = log_p_x0.reshape(trajectory.shape[0], -1).sum(dim=1)
        
        trace = 0; d = trajectory.reshape(trajectory.shape[0], -1).shape[1]
        # Hutchinson's trace estimator with finite difference approximation, 估计流的雅可比矩阵的迹（trace），用于计算 log p(x_T) 时的变化率（log p(x_T) = log p(x_0) - integral of trace over time）
        sigma = 0.001/math.sqrt(d)
        if self.global_eps is None or self.global_eps.shape != trajectory.shape:
            self.global_eps = torch.randn_like(trajectory).to(condition_data.device)

        for t in timesteps[:-1]:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            t = t*torch.ones(len(trajectory)).to(trajectory.device)
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            # Hutchinson's trace estimator with finite difference approximation
            model_output_perturb = model(trajectory + sigma*self.global_eps, t, 
                local_cond=local_cond, global_cond=global_cond)
            e_dzdx = (model_output_perturb - model_output) / sigma
            e_dzdx_e = e_dzdx * self.global_eps
            approx_tr_dzdx = e_dzdx_e.reshape(trajectory.shape[0], -1).sum(dim=1)
            trace = trace + sign * approx_tr_dzdx/self.num_inference_steps

            # 3. compute previous image: x_t -> x_t-1
            trajectory = trajectory + sign * model_output/self.num_inference_steps
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]
        log_p_xT = log_p_x0 - trace
        return trajectory, log_p_xT       

    def get_continuous_columns(self, tensor, ranges):
        selected_columns = []
        for start, end in ranges:
            selected_columns.append(tensor[:, start:end])  # end+1 to include the end index
        return torch.cat(selected_columns, dim=1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: # 预测动作的接口，输入是一个包含观测的字典，输出是一个包含动作的字典
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)  # global_cond 是把 n_obs_steps 个时间步的观测特征拼在一起，作为全局条件输入到模型中，模型在每个时间步都能看到整个观测序列的信息
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
            
        if self.num_rep > 1:
            cond_data = cond_data.repeat_interleave(self.num_rep, dim=0)
            cond_mask = cond_mask.repeat_interleave(self.num_rep, dim=0)
            global_cond = global_cond.repeat_interleave(self.num_rep, dim=0)
        # run sampling
        nsample, log_p_xT = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        result = {
            'action': action,
            'action_pred': action_pred,
            'global_cond': global_cond,
            'log_p_xT': log_p_xT
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch): # 训练时的损失计算
        """
        1. 从训练集采一批 (obs, action) 对
        2. 视觉编码 obs → global_cond
        3. 采样纯噪声 x1 和时间点 t
        4. 直线插值得到中间点 noisy_trajectory
        5. 网络前向预测速度
        6. 算 MSE loss 让预测接近真实方向 (x1 - x0)
        7. backprop 更新网络参数
        """
        # normalize input  batch 里已经包含了 obs 和 action 的序列，obs 是原始观测（图像+低维），action 是原始动作
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:])) # 截取前 n_obs_steps 帧
            nobs_features = self.obs_encoder(this_nobs) # 视觉编码器提取的特征，输入是 B*T 个观测，输出是 B*T 个特征向量
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timestamps = torch.rand((bsz,), device=trajectory.device)
        cont_t = timestamps.view(-1, *([1] * (noise.dim() - 1)))  # 随机时间步 t ∈ [0,1]
        timesteps = (timestamps * self.noise_scheduler.config.num_train_timesteps).long()
        
        # Flow step: x0 -> x1 flow matching 数据准备
        x0, x1 = trajectory, noise  # x0 是原始动作序列，x1 是噪声，模型的目标是从 x1 预测出 x0-x1 的方向（或者说噪声），然后在采样时通过迭代去除噪声得到动作序列
        direction = x1 - x0  # 预测噪声的方向（或者说 x0 到 x1 的流的方向），模型的输出是这个方向的估计值
        noisy_trajectory = x0 + cont_t*direction  # 添加噪声，得到扩散过程中的当前状态 noisy_trajectory 是在 x0 和 x1 之间的一个点，具体位置由 cont_t 决定（相当于扩散/流的时间步）
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the direction  让模型预测这个"方向"（而不是像 diffusion 那样预测噪声本身） 网络预测方向 
        pred = self.model(noisy_trajectory, timesteps,  # 模型输入 noisy_trajectory 和 timesteps，以及可能的条件 local_cond 和 global_cond，输出是对 direction 的预测
            local_cond=local_cond, global_cond=global_cond)
        target = direction # 训练目标是让模型预测出 direction，即从 noisy_trajectory 到 x1 的方向（或者说从 noisy_trajectory 到 x0 的反方向） 目标是方向，不是噪声

        def loss_process(loss):
            loss = loss * loss_mask.type(loss.dtype)
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
            return loss
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss_process(loss)
        return loss
