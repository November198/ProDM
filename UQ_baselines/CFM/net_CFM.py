import sys
sys.path.append('../..')
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
# 无条件 1D UNet：输入 (B, L, 10) + time t → 输出同形状的速度场。模型本身无条件 → 只学边际分布 p(O)
def get_unet(input_dim):
    return ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=False
    )


    