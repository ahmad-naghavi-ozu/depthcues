import os, glob
import argparse
import numpy as np
import timm
import wandb
from tqdm import tqdm

import torch
#from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, InterpolationMode

from utils import instantiate_from_config, get_transform_wo_crop, WarmupCosineSchedule, Accuracy
import json
import csv
import functools
from omegaconf import OmegaConf
from utils import parse_unknown_args

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        const=True,
        default=False,
        nargs="?",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="only test pre-trained probes",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    return parser



def train(feature_extractor, probe_model, train_loader, val_loader, criterion, metric, optimizer, lr_scheduler, logdir, config, device):
    best_score_val = 0 if metric.is_higher_better else np.inf
    best_score_train = 0 if metric.is_higher_better else np.inf
    global_step = 0
    for epoch in range(config['epochs']):
        probe_model.train()
        epoch_score = []
        epoch_loss = []
        # run training for this epoch
        with tqdm(train_loader, unit='batch') as tepoch:
            for i, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                imgs = batch['image']
                labels = batch['label']
                imgs = imgs.to(device) # (B,3,H,W)
                labels = labels.to(device) # (B,)

                optimizer.zero_grad()

                # extract features
                with torch.no_grad():
                    feats = feature_extractor(imgs, batch)
                preds = probe_model(feats).squeeze(-1) # (B,C) or (B,) in binary case

                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                score = metric(preds,labels)
                epoch_score.append(score.item())
                epoch_loss.append(loss.item())

                tepoch.set_postfix(loss=loss.item(), metric=score.item())

                if i % config['log_every_n_iter'] == 0:
                    wandb.log({'global_step': global_step,
                               'train_loss_iter': loss.item(), 
                               f'train_{metric.name}_iter': score.item(),})
                global_step += 1

        # validate for this epoch
        with torch.no_grad():
            val_score, val_loss, _ = evaluate(feature_extractor, probe_model, val_loader, criterion, metric, device, False)
            train_score = sum(epoch_score)/float(len(epoch_score))
            if (metric.is_higher_better and val_score > best_score_val) or \
               (not metric.is_higher_better and val_score < best_score_val):
                best_score_val = val_score
                best_score_train = train_score
                # save checkpoint
                torch.save({'state_dict': probe_model.state_dict(), 'epoch': epoch}, os.path.join(logdir, 'best.ckpt'))

        # log epoch stats
        wandb.log({
            'global_step': global_step,
            f'train_{metric.name}':train_score, 
            'train_loss':sum(epoch_loss)/float(len(epoch_loss)),
            f'val_{metric.name}':val_score, 
            'val_loss':val_loss,
            'lr': lr_scheduler.get_last_lr()[0],
            'epoch': epoch
        })
    
    return best_score_train, best_score_val


def test(feature_extractor, probe_model, test_loader, criterion, metric, logdir, device):
    with torch.no_grad():
        test_score, test_loss, record = evaluate(feature_extractor, probe_model, test_loader, criterion, metric, device, True)

    # save predictions
    fname = os.path.join(logdir, 'test_preds.csv')
    with open(fname, 'w') as f:
        n_cols = record.shape[1]
        header = ['path'] + ['pred']*((n_cols-1)//2) + ['label']*((n_cols-1)//2)
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(record.tolist())
        
    # log test stats
    wandb.log({
        f'test_{metric.name}':test_score, 
        'test_loss':test_loss, 
    })
    return test_score

def evaluate(feature_extractor, probe_model, data_loader, criterion, metric, device, return_record):
    probe_model.eval()
    epoch_loss = []
    preds_list = []
    preds_list_logits = []
    labels_list = []
    img_path_list = []
    
    for i, batch in enumerate(data_loader):
        imgs = batch['image']
        labels = batch['label']
        if return_record:
            img_path = batch['img_path']
        else:
            img_path = None
            
        imgs = imgs.to(device)
        labels = labels.to(device) # (B,)
        
        # extract features
        with torch.no_grad():
            feats = feature_extractor(imgs, batch)
        preds = probe_model(feats).squeeze(-1) # (B,C) or (B,) in binary case

        loss = criterion(preds, labels)
        epoch_loss.append(loss.item())
        
        if return_record:
            img_path_list += img_path
            preds_list.append(metric.get_preds(preds))
        preds_list_logits.append(preds)
        labels_list.append(labels)
    
    if return_record:
        img_path_record = np.array(img_path_list).squeeze().T # (N,)
        preds_record = torch.cat(preds_list).cpu().numpy().squeeze().T # (N,) or (2,N)
        labels_record = torch.cat(labels_list).cpu().numpy().squeeze().T # (N,) or (2,N)
        records = np.vstack([img_path_record, preds_record, labels_record]).T # (N,3) or (N,5)
    else:
        records = None
    preds = torch.cat(preds_list_logits).squeeze().cpu() # (N,C)
    labels = torch.cat(labels_list).squeeze().cpu() # (N)
    epoch_score = metric(preds, labels).item()
    return epoch_score, sum(epoch_loss)/float(len(epoch_loss)), records


        
if __name__ == "__main__":
    parser = get_parser()

    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()

    # read configs
    config_cli = OmegaConf.create(parse_unknown_args(unknown))
    configs = [OmegaConf.load(cfg) for cfg in opt.base] + [config_cli,] # append cli config to overwrite
    config = OmegaConf.merge(*configs)
    data_config = config['data_config']
    loss_config = config['loss_config']
    probe_config = config['probe_config']
    feature_extractor_config = config['feature_extractor_config']
    metric_config = config.get('metric_config', None)
    if data_config['val']['target'] == '__is_same_as_train__':
        bs_val = None
        if 'batch_size' in data_config['val']:
            bs_val = data_config['val']['batch_size']
        tmp_config = data_config['val']['params']
        data_config['val'] = data_config['train']
        data_config['val']['params'].update(tmp_config)
        if bs_val is not None:
            data_config['val']['batch_size'] = bs_val
    if data_config['test']['target'] == '__is_same_as_train__':
        bs_test = None
        if 'batch_size' in data_config['test']:
            bs_test = data_config['test']['batch_size']
        tmp_config = data_config['test']['params']
        data_config['test'] = data_config['train']
        data_config['test']['params'].update(tmp_config)
        if bs_test is not None:
            data_config['test']['batch_size'] = bs_test

    # set up log dir, save config, wandb args
    if not opt.test:
        logdir = os.path.join(opt.logdir, config['task'], config['model_name'], feature_extractor_config['params']['feat_type']+'_'+probe_config['type'])
        run_name = config['model_name'] + '_' + feature_extractor_config['params']['feat_type'] + '_' + probe_config['type']
        if opt.name:
            run_name += f'_{opt.name}'
            logdir = os.path.join(logdir, opt.name)
        os.makedirs(logdir, exist_ok=True)
        # save configs
        print(OmegaConf.to_yaml(config))
        OmegaConf.save(config, os.path.join(logdir, 'config.yaml'))
        # wandb args
        wandb_kwargs = {
            'project': f'depthcue-{config.task}',
            'name': run_name,
            'dir': os.path.abspath(logdir),
            'mode': 'offline' if opt.debug else 'online',
            'config': OmegaConf.to_container(config)
        }
    else:
        base = opt.base[0] # in test-only case only 1 cfg can be specified
        logdir = os.path.dirname(base)
        run_name = config['model_name'] + '_' + feature_extractor_config['params']['feat_type'] + '_' + probe_config['type']
        postfix = os.path.basename(logdir.strip('/'))
        if postfix not in run_name:
            run_name += f'_{postfix}'
        # wandb args
        wandb_kwargs = {
            'project': f'depthcue-{config.task}',
            'name': run_name,
            'dir': os.path.abspath(logdir),
            'mode': 'offline' if opt.debug else 'online',
            'config': OmegaConf.to_container(config),
            'id': glob.glob(os.path.join(logdir, "wandb", '*run-*'))[0].split('-')[-1],
            'resume': 'must'
        }
        print('Testing run', run_name)


    # initialise logger
    wandb.init(**wandb_kwargs)

    device = torch.device(f'cuda:{opt.gpu_id}')
    torch.manual_seed(opt.seed)
    
    # create backbone model
    model = instantiate_from_config(config['create_model_func'])
    model = model.to(device)
    model = model.eval()

    # create feature extractor that wraps backbone model
    feature_extractor = instantiate_from_config(feature_extractor_config, model=model, probe_type=probe_config['type'])

    # define probing model
    # check whether in_features depend on model layer
    if probe_config.params.in_features == '__check__model__':
        probe_config.params.in_features = model.feat_dim[feature_extractor.layer]
        print('===Probe input feature dimension is set to', probe_config.params.in_features)
    probe_model = instantiate_from_config(probe_config)
    probe_model = probe_model.to(device)

    # set up data transform
    transform = None
    if 'create_data_transform_func' in config:
        transform = instantiate_from_config(config['create_data_transform_func'])
    elif 'timm' in config['create_model_func']['target']:
        model_data_config = timm.data.resolve_model_data_config(model)
        transform = get_transform_wo_crop(model_data_config)
    elif 'probe3d_backbones' in config['create_model_func']['target']:
        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
            Resize((512,512), interpolation=InterpolationMode.NEAREST)
        ])
    
    # set up data loader
    train_dataset = instantiate_from_config(data_config['train'], transform=transform)
    val_dataset = instantiate_from_config(data_config['val'], transform=transform)
    test_dataset = instantiate_from_config(data_config['test'], transform=transform)

    print("Training data:", len(train_dataset))
    print("Validation data:", len(val_dataset))
    print("Test data:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=data_config['train']['batch_size'], num_workers=data_config['num_workers'], pin_memory=True, drop_last=False, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=data_config['val']['batch_size'], num_workers=data_config['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=data_config['test']['batch_size'], num_workers=data_config['num_workers'], pin_memory=True)
    
    criterion = instantiate_from_config(loss_config)
    steps_per_epoch = len(train_loader)
    
    optimizer = torch.optim.AdamW(probe_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr_scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=0.,
        start_lr=config['learning_rate'],
        ref_lr=config['learning_rate'],
        final_lr=0.,
        T_max=config['epochs']*steps_per_epoch
    )

    if metric_config is not None:
        metric = instantiate_from_config(metric_config)
    else: # set default to accuracy as it is the most common one
        metric = Accuracy()

    if not opt.test:
        best_score_train, best_score_val = train(feature_extractor, probe_model, train_loader, val_loader, criterion, metric, optimizer, lr_scheduler, logdir, config, device)
        print(f"Best {metric.name} train", best_score_train)
        print(f"Best {metric.name} val", best_score_val)

        filename = os.path.join(logdir, 'train_results.json')
        results = {f'best_{metric.name}_train': best_score_train, f'best_{metric.name}_val': best_score_val}
        with open(filename, 'w') as f:
            results_json = json.dumps(results, indent=4)
            f.write(results_json)
    
    else:
        # load checkpoint
        probe_ckpt = torch.load(os.path.join(logdir, 'best.ckpt'))
        probe_model.load_state_dict(probe_ckpt['state_dict'])
        score_test = test(feature_extractor, probe_model, test_loader, criterion, metric, logdir, device)
        print(f"{metric.name} test", score_test)

        filename = os.path.join(logdir, 'test_results.json')
        results = {f'{metric.name}_test': score_test}
        with open(filename, 'w') as f:
            results_json = json.dumps(results, indent=4)
            f.write(results_json)