# coding=utf-8
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from datetime import datetime
from .utils import shuffle_tensor, InfoNCELoss
import os
import csv
import json
from icecream import ic


def save_opt_settings(opt, log_dir):
    settings_path = os.path.join(log_dir, 'settings.json')

    settings_dict = {}
    for key, value in vars(opt).items():
        try:
            json.dumps(value)
            settings_dict[key] = value
        except TypeError:
            settings_dict[key] = str(value)

    with open(settings_path, 'w') as f:
        json.dump(settings_dict, f, indent=4)

    print(f"Saved settings to {settings_path}")


def train(model, train_dataset, gpu, epoches_num, lr, opt, use_wandb=False):
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = f'log/{opt.split_ratio}/{timestamp}'
    save_dir = f'save/{opt.split_ratio}/{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    save_opt_settings(opt, log_dir)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    beta = 1
    device = opt.device
    tripletLoss = nn.TripletMarginLoss(margin=beta, reduction='mean').to(opt.device)
    infonce_loss = InfoNCELoss(temperature=0.2)
    tbar = tqdm(range(epoches_num))
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    csv_file = open(f'{log_dir}/training_log.csv', 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch',
        'train_loss'
    ])

    model.set_graph_features(train_dataset.graph_features.to(device))
    model.set_summary_features(train_dataset.summary_features.to(device))
    for epoch in tbar:
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            index, graph_edges, edge_types, bit_width, density, max_path, label_dict = batch
            index = index.to(gpu)
            graph_edges = graph_edges.to(gpu)
            edge_types = edge_types.to(gpu)
            bit_width = bit_width.to(gpu)
            density = density.to(gpu)
            max_path = max_path.to(gpu)
            # 0[pad] 1[cls]
            node_mask = (index[:, :] > 1)
            graph, graph_emb, summary, summary_emb, y, z = model(index, graph_edges, edge_types, bit_width, max_path, density)

            # train_size % batch_size should >= 1
            yn = shuffle_tensor(y)
            zn = shuffle_tensor(z)
            loss = tripletLoss(y, z, zn) + tripletLoss(z, y, yn)
            loss.backward()
            optimizer.step()

            if graph.size(0) == opt.batch_size:
                tbar.set_postfix({"batch": i, "loss": loss.item()}, refresh=True)
            train_loss += loss.item()

        # 计算平均训练损失
        scheduler.step()
        train_loss /= len(train_loader)

        tbar.set_postfix({
            "Epoch": epoch,
            'Total': train_loss
        }, refresh=True)

        csv_writer.writerow([
            epoch,
            train_loss
        ])
        csv_file.flush()

        torch.save(model.state_dict(), f'{save_dir}/epoch_{epoch + 1}.pth')

    csv_file.close()

