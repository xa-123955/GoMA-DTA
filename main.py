import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='dgl.heterograph')
from model import GoMADTA
from time import time
from utils import set_seed, graph_collate_func, mkdir,worker_init_fn
from configs import get_cfg_defaults
from dataloder import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
import torch.multiprocessing as mp
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

mp.set_start_method('spawn', force=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="GPS-DTI for DTI prediction")
parser.add_argument('--cfg',  default='configs/GoMADTA.yaml', help="path to config file", type=str)
parser.add_argument('--data',  default='2016', type=str, metavar='TASK', help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'time', 'cluster'])



args = parser.parse_args()

def main():
    # wandb.init()
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    train_path = os.path.join(dataFolder, 'train_data.csv')
    val_path = os.path.join(dataFolder, "val_data.csv")
    test_path = os.path.join(dataFolder, "test_data.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = DTIDataset(df_train.index.values, df_train)
    val_dataset = DTIDataset(df_val.index.values, df_val)
    test_dataset = DTIDataset(df_test.index.values, df_test)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func, 'pin_memory': True,'persistent_workers': True if cfg.SOLVER.NUM_WORKERS > 0 else False}

    training_generator = DataLoader(train_dataset, **params)
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    params['shuffle'] = False
    params['drop_last'] = False

    model = GoMADTA(config=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005,weight_decay=0.0005) #2016 0.0005
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.SOLVER.MAX_EPOCH)
    print(type(scheduler))
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(model, opt,scheduler, device, training_generator, val_generator, test_generator, **cfg)
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")
    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
