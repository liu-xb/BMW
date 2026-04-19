from __future__ import absolute_import
import datetime
import os
import sys
import errno
import os.path as osp
import shutil
import torch
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
            self.file.flush()
            os.fsync(self.file.fileno())

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def save_model(model_ema, is_best, best_mAP, epoch, logs_dir):
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(logs_dir, f'Epoch{epoch}-checkpoint.pth.tar'))


def check_gpu(GPU, need_memory=20000, wait_level=1):
    print(f'checking free memory of gpu {GPU} ...', end='')
    GPU = GPU.split(',')
    GPU = [int(gpu) for gpu in GPU]
    NotOK = wait_level
    while NotOK:
        print(wait_level, end =' ')
        sys.stdout.flush()
        free_memory = []
        for device in GPU:
            this_free_memory = os.popen('nvidia-smi --id=' + str(device) + ' --query-gpu=memory.free --format=csv,noheader').readlines()[0].split(' ')[0]
            free_memory.append(float(this_free_memory))
        free_memory = min(free_memory)
        if free_memory > need_memory:
            NotOK -= 1
        else:
            time.sleep(600)
            NotOK = wait_level
    print('\r training on gpu: ', GPU, '--------------')


def time_now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        # checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def show_args(args):
    print(time_now())
    print("==========")
    args_keys = list(args.__dict__.keys())
    args_keys.sort()
    for k in args_keys:
        print(f'{k} : {args.__dict__[k]}')
    print("==========")