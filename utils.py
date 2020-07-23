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
        #print('loaded ', dict)
        #print('current ',net_list[0].state_dict() )

        try:
            for k, v in net_list[i].state_dict().items():
                if k in dict:
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