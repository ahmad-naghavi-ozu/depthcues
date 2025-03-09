import os, glob
import argparse
import numpy as np
import wandb
from tqdm import tqdm
import json
import csv
import functools
from omegaconf import OmegaConf
import timm
import torch
#from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, InterpolationMode

from utils import instantiate_from_config, get_transform_wo_crop, WarmupCosineSchedule
from models.multi_cue_loss import MultiCueLoss
from data.multi_cue import MultiCueDataset, dataset_dict, MultiCueTrainValBatchSampler, MultiCueTestBatchSampler
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
        default="logs_finetune",
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
        "--ckpt",
        type=str,
        default="",
        help="file name of ckpt to test",
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




def train(adapter, train_loader, val_loader, criterion, metrics, optimizer, lr_scheduler, logdir, config, device, topk=3):
    k_best_score_val = {} # values are dicts {cue: float}
    k_best_score_train = {} # values are dicts {cue: float}
    k_best_total_val_loss = {} # values are float
    global_step = 0
    for epoch in range(config['epochs']):
        adapter.train()
        epoch_score = {cue: [] for cue in criterion.cues}
        epoch_loss = {cue: [] for cue in criterion.cues}
        # run training for this epoch
        with tqdm(train_loader, unit='batch') as tepoch:
            for i, (batch, cue) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                cue = cue[0]
                imgs = batch['image']
                labels = batch['label']
                imgs = imgs.to(device) # (B,3,H,W)
                labels = labels.to(device) # (B,)
                metric_name = metrics[cue].name

                optimizer.zero_grad()

                preds = adapter(imgs, batch)
                preds = preds[:, criterion.cue_indices[cue][0]:criterion.cue_indices[cue][1]].squeeze(-1) # get task relevant output

                loss = criterion.loss_weights[cue] * criterion.loss_functions[cue](preds, labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                score = metrics[cue](preds, labels)
                epoch_score[cue].append(score.item())
                epoch_loss[cue].append(loss.item())

                tepoch.set_postfix(cue=cue, loss=loss.item(), metric=score.item())

                if i % config['log_every_n_iter'] == 0:
                    wandb.log({'global_step': global_step,
                               f'train_{cue}_loss_iter': loss.item(), 
                               f'train_{metric_name}_{cue}_iter': score.item(),})
                global_step += 1

        # validate for this epoch
        with torch.no_grad():
            val_score, val_loss, _ = evaluate(adapter, val_loader, criterion, metrics, device, False)
            train_score = {cue: sum(epoch_score[cue])/float(len(epoch_score[cue])) for cue in epoch_score.keys()}

            total_val_loss = sum(val_loss[cue] for cue in val_loss.keys())
            if len(k_best_total_val_loss) < topk:
                k_best_total_val_loss[epoch] = total_val_loss
                k_best_score_val[epoch] = val_score
                k_best_score_train[epoch] = train_score
                torch.save({'state_dict': adapter.state_dict(), 'epoch': epoch}, os.path.join(logdir, f'best_{epoch}.ckpt'))
            elif epoch == config['epochs']-1: # save last ckpts
                torch.save({'state_dict': adapter.state_dict(), 'epoch': epoch}, os.path.join(logdir, f'best_{epoch}.ckpt'))
            elif len(val_score) == 1: # finetuning on only one cue, save by val score
                cue = list(val_score.keys())[0]
                tuples = [(e, s[cue]) for e, s in k_best_score_val.items()] + [(epoch, val_score[cue])]
                tuples = sorted(tuples, key=lambda x: x[1])
                worst_epoch = tuples[0][0] if metrics[cue].is_higher_better else tuples[-1][0] # this has the worst val score
                if worst_epoch != epoch: # cur epoch is in top k
                    # remove non-top-k ckpt records
                    k_best_total_val_loss.pop(worst_epoch)
                    k_best_score_val.pop(worst_epoch)
                    k_best_score_train.pop(worst_epoch)
                    os.remove(os.path.join(logdir, f'best_{worst_epoch}.ckpt'))
                    # save cur ckpt records as it is top-k
                    k_best_total_val_loss[epoch] = total_val_loss
                    k_best_score_val[epoch] = val_score
                    k_best_score_train[epoch] = train_score
                    torch.save({'state_dict': adapter.state_dict(), 'epoch': epoch}, os.path.join(logdir, f'best_{epoch}.ckpt'))
                    assert len(k_best_score_val) == topk
            else: # compare with current top k, remove the smallest one
                tuples = [(e, l) for e, l in k_best_total_val_loss.items()] + [(epoch, total_val_loss)]
                tuples = sorted(tuples, key=lambda x: x[1])
                worst_epoch = tuples[-1][0] # this has the largest total_val_loss
                if worst_epoch != epoch: # cur epoch is in top k
                    # remove non-top-k ckpt records
                    k_best_total_val_loss.pop(worst_epoch)
                    k_best_score_val.pop(worst_epoch)
                    k_best_score_train.pop(worst_epoch)
                    os.remove(os.path.join(logdir, f'best_{worst_epoch}.ckpt'))
                    # save cur ckpt records as it is top-k
                    k_best_total_val_loss[epoch] = total_val_loss
                    k_best_score_val[epoch] = val_score
                    k_best_score_train[epoch] = train_score
                    torch.save({'state_dict': adapter.state_dict(), 'epoch': epoch}, os.path.join(logdir, f'best_{epoch}.ckpt'))
                    assert len(k_best_total_val_loss) == topk

        # log epoch stats
        train_loss = {cue: sum(l)/float(len(l)) for cue, l in epoch_loss.items()}
        total_train_loss = sum(train_loss[cue] for cue in train_loss.keys())

        log_dict_loss_train = {f'train_loss_{cue}': train_loss[cue] for cue in train_loss.keys()}
        log_dict_loss_val = {f'val_loss_{cue}': val_loss[cue] for cue in val_loss.keys()}
        log_dict_score_train = {f'train_{metrics[cue].name}_{cue}': train_score[cue] for cue in train_score.keys()}
        log_dict_score_val = {f'val_{metrics[cue].name}_{cue}': val_score[cue] for cue in val_score.keys()}
        log_dict_meta = {
            'global_step': global_step,
            'train_loss_total': total_train_loss,
            'val_loss_total': total_val_loss,
            'lr': lr_scheduler.get_last_lr()[0],
            'epoch': epoch
        }
        wandb.log({
            **log_dict_meta,
            **log_dict_loss_train,
            **log_dict_loss_val,
            **log_dict_score_train,
            **log_dict_score_val,
        })
    
    return k_best_score_train, k_best_score_val


def test(adapter, test_loader, criterion, metrics, logdir, device):
    with torch.no_grad():
        test_score, test_loss, record = evaluate(adapter, test_loader, criterion, metrics, device, True)

    # save predictions
    for cue in criterion.cues:
        fname = os.path.join(logdir, f'test_preds_{cue}.csv')
        with open(fname, 'w') as f:
            n_cols = record[cue].shape[1]
            header = ['path'] + ['pred']*((n_cols-1)//2) + ['label']*((n_cols-1)//2)
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(record[cue].tolist())
        
    # log test stats
    log_dict_score_test = {f'test_{metrics[cue].name}_{cue}': test_score[cue] for cue in test_score.keys()}
    log_dict_loss_test = {f'test_loss_{cue}': test_loss[cue] for cue in test_loss.keys()}
    wandb.log({
        **log_dict_score_test,
        **log_dict_loss_test, 
    })
    return test_score

def evaluate(adapter, data_loader, criterion, metrics, device, return_record):
    adapter.eval()
    epoch_score = {cue: [] for cue in criterion.cues}
    epoch_loss = {cue: [] for cue in criterion.cues}
    preds_list = {cue: [] for cue in criterion.cues}
    preds_list_logits = {cue: [] for cue in criterion.cues}
    labels_list = {cue: [] for cue in criterion.cues}
    img_path_list = {cue: [] for cue in criterion.cues}
    
    for i, (batch, cue) in enumerate(data_loader):
        cue = cue[0]
        imgs = batch['image']
        labels = batch['label']
        imgs = imgs.to(device)
        labels = labels.to(device) # (B,)
        if return_record:
            img_path = batch['img_path']
        else:
            img_path = None

        preds = adapter(imgs, batch)
        preds = preds[:, criterion.cue_indices[cue][0]:criterion.cue_indices[cue][1]].squeeze(-1) # get task relevant output

        loss = criterion.loss_weights[cue] * criterion.loss_functions[cue](preds, labels)
        epoch_loss[cue].append(loss.item())

        if return_record:
            img_path_list[cue] += img_path
            preds_list[cue].append(metrics[cue].get_preds(preds))
        preds_list_logits[cue].append(preds)
        labels_list[cue].append(labels)
    
    if return_record:
        img_path_record = {cue: np.array(img_paths).squeeze().T for cue, img_paths in img_path_list.items()} # (N,)
        preds_record = {cue: torch.cat(preds).cpu().numpy().squeeze().T for cue, preds in preds_list.items()} # (N,) or (2,N)
        labels_record = {cue: torch.cat(labels).cpu().numpy().squeeze().T for cue, labels in labels_list.items()} # (N,) or (2,N)
        records = {cue: np.vstack([img_path_record[cue], preds_record[cue], labels_record[cue]]).T for cue in criterion.cues} # (N,3) or (N,5)
    else:
        records = None
    preds = {cue: torch.cat(preds_logits).squeeze().cpu() for cue, preds_logits in preds_list_logits.items()} # (N,C)
    labels = {cue: torch.cat(labels).squeeze().cpu() for cue, labels in labels_list.items()} # (N)
    epoch_score = {cue: metrics[cue](preds[cue], labels[cue]).item() for cue in preds.keys()}
    epoch_loss = {cue: sum(l)/float(len(l)) for cue, l in epoch_loss.items()}
    return epoch_score, epoch_loss, records


        
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
    metric_config = config['metric_config']
    adapter_config = config['adapter_config']
    if 'val_batch_size' not in data_config:
        data_config.val_batch_size = data_config.train_batch_size
    if 'test_batch_size' not in data_config:
        data_config.test_batch_size = data_config.train_batch_size

    # set up log dir, save config, wandb args
    if len(data_config.cues) == 6:
        cue_setup = 'all_cues'
    elif len(data_config.cues) == 1:
        kept = data_config.cues[0]
        cue_setup = f'only_{kept}'
    elif len(data_config.cues) == 5:
        leftout = list(dataset_dict.keys() - set(data_config.cues))[0]
        cue_setup = f'no_{leftout}'
    else:
        cue_setup = '_'.join(data_config.cues)
    if not opt.test:
        logdir = os.path.join(opt.logdir, 'finetune', config['model_name'], cue_setup)
        run_name = config['model_name'] + '_' + cue_setup
        if opt.name:
            run_name += f'_{opt.name}'
            logdir = os.path.join(logdir, opt.name)
        os.makedirs(logdir, exist_ok=True)
        # save configs
        print(OmegaConf.to_yaml(config))
        OmegaConf.save(config, os.path.join(logdir, 'config.yaml'))
        # wandb args
        wandb_kwargs = {
            'project': 'depthcue-finetune',
            'name': run_name,
            'dir': os.path.abspath(logdir),
            'mode': 'offline' if opt.debug else 'online',
            'config': OmegaConf.to_container(config)
        }
    else:
        assert opt.ckpt, 'Checkpoint file name must be provided in test mode!'
        base = opt.base[0] # in test-only case only 1 cfg can be specified
        logdir = os.path.dirname(base)
        run_name = config['model_name'] + '_' + cue_setup
        postfix = os.path.basename(logdir.strip('/'))
        if postfix not in run_name:
            run_name += f'_{postfix}'
        # wandb args
        wandb_kwargs = {
            'project': 'depthcue-finetune',
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

    # create adapter model
    adapter = instantiate_from_config(adapter_config, backbone=model)
    adapter = adapter.to(device)
    print('Frozen parameters:', sum(p.numel() for p in adapter.parameters() if not p.requires_grad))
    print('Trainable parameters:', sum(p.numel() for p in adapter.parameters() if p.requires_grad))

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
    train_dataset = MultiCueDataset(data_config.data_path, data_config.cues, transform=transform, split='train', return_path=False)
    val_dataset = MultiCueDataset(data_config.data_path, data_config.cues, transform=transform, split='val', return_path=False)
    test_dataset = MultiCueDataset(data_config.data_path, data_config.cues, transform=transform, split='test', return_path=True)

    train_batch_sampler = MultiCueTrainValBatchSampler(train_dataset, 
                                                       data_config.train_batch_size, 
                                                       steps_per_cue=data_config.steps_per_cue, 
                                                       drop_last=False, shuffle=True)
    val_batch_sampler = MultiCueTrainValBatchSampler(val_dataset, 
                                                     data_config.val_batch_size, 
                                                     steps_per_cue=data_config.steps_per_cue, 
                                                     drop_last=False, shuffle=False)
    test_batch_sampler = MultiCueTestBatchSampler(test_dataset, data_config.test_batch_size)

    print("Training data:", len(train_dataset))
    print("Validation data:", len(val_dataset))
    print("Test data:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=data_config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, num_workers=data_config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=data_config.num_workers, pin_memory=True)
    
    criterion = MultiCueLoss(loss_config, cues=data_config.cues)
    metrics = {cue: instantiate_from_config(metric_config[cue]) for cue in data_config.cues}
    steps_per_epoch = len(train_loader)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config['start_lr'], weight_decay=config['weight_decay'])
    lr_scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=config['warmup_epochs']*steps_per_epoch,
        start_lr=config['start_lr'],
        ref_lr=config['learning_rate'],
        final_lr=config['start_lr'],
        T_max=config['epochs']*steps_per_epoch
    )

    if not opt.test:
        k_best_score_train, k_best_score_val = train(adapter, train_loader, val_loader, criterion, metrics, optimizer, lr_scheduler, logdir, config, device)
        print(f"Best train scores\n", k_best_score_train)
        print(f"Best val scores\n", k_best_score_val)

        filename = os.path.join(logdir, 'train_results.json')
        results = {f'best_train_scores': k_best_score_train, f'best_val_scores': k_best_score_val}
        with open(filename, 'w') as f:
            results_json = json.dumps(results, indent=4)
            f.write(results_json)
    
    else:
        # load checkpoint
        ckpt = torch.load(os.path.join(logdir, opt.ckpt))
        adapter.load_state_dict(ckpt['state_dict'])
        score_test = test(adapter, test_loader, criterion, metrics, logdir, device)
        print(f"Test scores\n", score_test)

        filename = os.path.join(logdir, 'test_results.json')
        results = {f'test_scores': score_test}
        with open(filename, 'w') as f:
            results_json = json.dumps(results, indent=4)
            f.write(results_json)