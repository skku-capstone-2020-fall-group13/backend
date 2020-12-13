import os
import glob
import json
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

from unet.transform import Unfold, RotateAll, AdjustColor, CloneCat
from unet.util import init_logger, set_seed, rgb_to_cat, color2idx

import multiprocessing
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)
max_cpu_count = multiprocessing.cpu_count()


def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        
        outputs = model(batch[0])
        loss = loss_fn(outputs, batch[1])
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss


def eval(model, eval_loader, loss_fn, device):
    val_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            batch = tuple(t.to(device) for t in batch)

            outputs = model(batch[0])
            loss = loss_fn(outputs, batch[1])
            val_loss += loss.item()

    return val_loss


def main(args):
    set_seed(args.seed)
    init_logger()

    device = "cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    cpu_count = min(args.cpu_count, max_cpu_count)
    logger.info("Device to use = %s", device)
    logger.info("Number of cpu to use = %d/%d", cpu_count, max_cpu_count)

    image_paths = sorted(glob.glob(os.path.join(args.data_dir,'sat')+'/*.png'))
    layer_paths = sorted(glob.glob(os.path.join(args.data_dir,'gt')+'/*.png'))

    transform_sat = transforms.Compose([
        transforms.ToTensor(),
        Unfold(crop_size=args.crop_size, stride=args.stride),
        RotateAll(),
        AdjustColor(n=3),
    ])

    transform_gt = transforms.Compose([
        transforms.ToTensor(),
        Unfold(crop_size=args.crop_size, stride=args.stride),
        RotateAll(),
        CloneCat(n=3)
    ])

    transformed_images = torch.tensor([], dtype=torch.float32)
    transformed_layers = torch.tensor([], dtype=torch.int64)

    logger.info("Image crop size & stride = %d, %d", args.crop_size, args.stride)
    for i in tqdm(range(len(image_paths)), 'Transforming Images'):
        image = Image.open(image_paths[i]).convert('RGB')
        layer = Image.open(layer_paths[i]).convert('RGB')
        layer = rgb_to_cat(np.array(layer))
        transformed_images = torch.cat((transformed_images, transform_sat(image)))
        transformed_layers = torch.cat((transformed_layers, transform_gt(layer)))
        
    trainset = TensorDataset(transformed_images, transformed_layers.type(torch.int64))
    train_idx, eval_idx = train_test_split(np.arange(len(trainset)), test_size=0.1, random_state=args.seed, shuffle=True)

    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, sampler=SubsetRandomSampler(train_idx), num_workers=cpu_count)
    eval_loader = DataLoader(trainset, batch_size=args.eval_batch_size, sampler=SubsetRandomSampler(eval_idx), num_workers=cpu_count)
    
    model = smp.Unet(encoder_name=args.encoder_name, classes=len(color2idx), activation='softmax').to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor((1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)).to(device))

    logger.info("***** Running training *****")
    logger.info("  Total examples = %d", len(trainset))
    logger.info("  Num train dataset = %d / Num eval = %d", len(train_idx), len(eval_idx))
    logger.info("  Train batch size = %d", args.train_batch_size)
    logger.info("  Eval batch size = %d", args.eval_batch_size)
    logger.info("  Num epochs = %d", args.n_epochs)
    logger.info("  Save epochs = %d", args.save_epochs)

    for epoch in range(args.n_epochs):
        tr_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval(model, eval_loader, loss_fn, device)
        logger.info(f'Epoch[{epoch+1}/{args.n_epochs}] Train loss : {tr_loss/len(train_idx)} / Eval loss : {val_loss/len(eval_idx)}')

        # Save model
        if (epoch+1) % args.save_epochs == 0:
            output_dir = os.path.join(args.save_dir, f'model_{args.encoder_name}_c{args.crop_size}_s{args.stride}_epoch{epoch+1}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            config = {
                'smp_args' : {'encoder_name' : args.encoder_name, 'classes' : len(color2idx)},
                'crop_size' : args.crop_size
            }

            with open(os.path.join(output_dir, 'config.json'), 'w') as json_file:
                json.dump(config, json_file)
            torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.pt'))
            logger.info(f" Saving model to {output_dir}")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', help='the directory which has sat and gt folders', default='data')
    parser.add_argument('--save_dir', help='the directory where model will be saved', default='model')  
    parser.add_argument('--encoder_name', help='pretrained backbone encoder name of Unet.', default='efficientnet-b0')
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save_epochs', type=int, default=2)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--cpu_count', type=int, default=1)


    main(parser.parse_args())