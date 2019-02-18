'''
Run/checkpoint directory management
'''

BEST_CHECKPOINT_NAME = "checkpoint_best_model.t7"

import json
import os

class TopLevelDir(object):
    def __init__(self, base_path):
        self.base_path = base_path

    def get_arch_path(self, arch_name):
        return os.path.join(self.base_path, arch_name)

'''
All the checkpoints for a specific architecture.
'''
class ArchDir(object):
    def __init__(self, arch_path):
        self.arch_path = arch_path

    def most_recent_train(self):
        '''
        get the checkpoint directory corresponding to the most recent training run
        '''
        timestamps = sorted(os.listdir(self.arch_path))
        assert len(timestamps) > 0
        run_dir = os.path.join(self.arch_path, timestamps[-1])
        return run_dir

    def by_checkpoint_id(self, checkpoint):
        dir_path = os.join(self.arch_path, checkpoint)
        return dir_path

    def new_checkpoint_dir(self, name=None):
        if name:
            dir_path = os.path.join(self.arch_path, name)
        else:
            timestamp = "{:.0f}".format(datetime.utcnow().timestamp())
            dir_path = os.path.join(self.arch_path, timestamp)

        assert (not os.path.exists(dir_path))
        os.makedirs(dir_path)

        return dir_path

class CheckpointDir(object):
    def __init__(self, run_path):
        self.run_path = run_path

    def latest_checkpoint_path(self):
        pass

    def best_checkpoint_path(self):
        return os.path.join(self.run_path, BEST_CHECKPOINT_NAME)

    def config_path(self):
        return os.path.join(self.run_path, CONFIG_NAME)

    def load_config(self):
        with open(self.config_path(), 'r') as f:
            return json.load(f)

    def save_config(self, kv_dict):
        with open(self.config_path(), 'w') as f:
            json.dump(kv_dict, f)
    
    def epoch_path(self, epoch):
        return os.path.join(self.run_path, f'checkpoint_{epoch}.f7')

    def save_checkpoint(self, model):
        pass
    
class EnsembleDir(CheckpointDir):
    def init_subdirs(self):
        files = os.listdir(self.run_path)
        self.subdirs = {}
        for f in files:
            if f.isdigit():
                self.subdirs[int(f)] = CheckpointDir(os.join(self.run_path, f))
    
    def __init__(self, ensemble_path):
        super().__init__(ensemble_path)

        self.init_subdirs()
        
    def __getitem__(self, key):
        assert isinstance(key, int)
        return self.subdirs[key]


def load_latest_checkpoint(basedir, arch, checkpoint_id='latest'):
    topdir = TopLevelDir(basedir)
    arch_dir = ArchDir(topdir.get_arch_path(arch))
    checkpoint_dir = CheckpointDir(arch_dir.most_recent_train())
    return checkpoint_dir

def load_ensemble(basedir):
    pass

def new_checkpoint_dir(basedir, arch):
    topdir = TopLevelDir(basedir)
    arch_dir = ArchDir(topdir.get_arch_path(arch))
    return arch_dir.new_checkpoint_dir()
