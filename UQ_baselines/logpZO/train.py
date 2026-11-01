import tqdm
import torch
import os
import matplotlib.pyplot as plt
import sys
master_dir = '../../UQ_baselines'
sys.path.append(master_dir)
import CFM.net_CFM as Net
import data_loader
from argparse import ArgumentParser
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    
parser = ArgumentParser()
parser.add_argument("--type", default='square', type=str)
parser.add_argument("--policy_type", default='flow', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    type = args.type  # L20-26数据准备
    X, Y = data_loader.get_data(type=type, adjust_shape=True, diffusion=args.policy_type == 'diffusion')
    global_cond_dim = X.shape[1]; input_dim = Y.reshape(Y.shape[0], 16, -1).shape[-1]
    print(f'Current feature shape: {global_cond_dim}')
    train_data = torch.utils.data.TensorDataset(X)  # 注意 只用观测O_t作为输入，不用动作A_t，RND用（A_t,O_t）
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    ckpt_file = f'{type}_{args.policy_type}.ckpt'
    
    # choice of model/method
    net = Net.get_unet(input_dim).to(device)  # 网络（公用CFM的UNet）
    EPOCHS = 200
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
    if os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        starting_epochs = ckpt['epoch']
        losses = ckpt['losses']
    else:
        starting_epochs = 0
        losses = []

    t = tqdm.trange(starting_epochs, EPOCHS)
    for i in t:
        # Save checkpoint before to avoid nan loss
        ckpt = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': i+1,
            'losses': losses
        }
        torch.save(ckpt, ckpt_file)

        net.train()
        loss_i = []
        for (x_batch, ) in tqdm.tqdm(train_loader, desc='Training Batches'): # 训练循环
            observation = x_batch.to(device)
            optimizer.zero_grad()
            x0, x1 = observation, torch.randn_like(observation).to(device) # x0是数据分布，x1是噪声分布
            vtrue = x1 - x0 # 速度场（噪声-数据）
            cont_t = torch.rand(len(x1),).to(device)
            cont_t = cont_t.view(-1, *[1 for _ in range(len(observation.shape)-1)])
            xnow = x0 + cont_t * vtrue # 当前时刻的样本（线性插值）
            time_scale = 100 # In UNet, which takes discrete time steps
            vhat = net(xnow, (cont_t.view(-1)*time_scale).long()) # 预测的速度场
            loss = (vhat - vtrue).pow(2).mean()
            loss.backward()
            # Terminate if loss is NaN
            if torch.isnan(loss):
                print("Loss is NaN")
                raise ValueError(f"NaN loss at epoch {i}")
            loss_i += [loss.item()]
            optimizer.step()
        losses += [sum(loss_i)/len(loss_i)]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_title("Training loss")
        ax.plot(losses)
        suffix = type
        os.makedirs('images', exist_ok=True)
        plt.savefig(f"images/training_loss_{suffix}.png")
        plt.close('all')