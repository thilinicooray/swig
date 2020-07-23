import torch
import numpy as np
import pdb


def load_net(fname, net_list, prefix_list = None):
    '''
    loading a pretrained model weights from a file
    '''

    for i in range(0, len(net_list)):
        if not torch.cuda.is_available():
            dict_all = torch.load(fname, map_location='cpu')
        else:
            dict_all = torch.load(fname)
        dict = dict_all['state_dict']
        unnecessary_modules = ['vocab_linear', 'vocab_linear_2', 'verb_embeding', 'noun_embedding', 'regressionModel',
                               'classificationModel', 'rnn', 'rnn_linear']
        try:
            for k, v in net_list[i].state_dict().items():
                prefix = k.split('.')[0]
                if prefix in unnecessary_modules:
                    continue
                if k in dict:
                    #print('copied ', k)
                    param = torch.from_numpy(np.asarray(dict[k].cpu()))
                    v.copy_(param)
                else:
                    print('[Missed]: {}'.format(k), v.size())
        except Exception as e:
            print(e)
            pdb.set_trace()
            print ('[Loaded net not complete] Parameter[{}] Size Mismatch...'.format(k))

def set_trainable(model, requires_grad):
    '''
    set model parameters' training mode on/off
    '''
    set_trainable_param(model.parameters(), requires_grad)

def set_trainable_param(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad