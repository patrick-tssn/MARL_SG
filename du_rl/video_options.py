from __future__ import absolute_import
from __future__ import division
import os

class Option:
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        with open (os.path.join(self.this_expsdir, "option.txt"), "w", encoding='UTF-8') as f:
            for key, value in sorted(self.__dict__.items(), key = lambda x: x[0]):
                f.write("{}, {}\n".format(key, str(value)))

def read_options():
    d = {}
    d['state_embed_size'] = 50
    d['mlp_hidden_size'] = 100
    d['use_entity_embed'] = True
    d['grad_clip_norm'] = 5
    d['train_times'] = 20 # roll outs
    d['test_times'] = 100
    d['train_batch'] = 200
    d['max_out'] = 10
    d['max_step_length'] = 2
    d['l2_reg_const'] = 1e-2
    d['learning_rate'] = 1e-3
    d['batch_size'] = 256
    d['decay_weight'] = 0.02
    d['decay_batch'] = 100
    d['decay_rate'] = 0.95
    d['gamma'] = 0.5
    d['Lambda'] = 0.02
    d['beta'] = 0.02
    d['vrd_thres'] = 0.5
    d['log_path'] = 'vrd_log/'
    d['train_rollouts'] = 5
    d['test_rollouts'] = 5
    option = Option(d)
    return option
