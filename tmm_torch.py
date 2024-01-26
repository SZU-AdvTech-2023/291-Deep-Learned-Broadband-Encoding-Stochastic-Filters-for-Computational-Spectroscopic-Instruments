import numpy as np
import math
import torch 
import torch.nn as nn
from pathlib import Path
import csv


matPath = Path(__file__).parent.resolve()/ Path('material')

def load_nk(csv_file,lambda_list,lambda_unit='nm'):
    '''
    读入材料nk值csv文件，再根据lambda_list插值出nk曲线
    @param csv_file: csv文件名
    @param lambda_list: 插值的lambda列表
    @param lambda_unit: csv文件中lambda的单位，'nm'或'um'，默认为'nm'
    @return: 插值后的nk曲线
    '''
    raw_wl = []
    raw_data = []
    multiplier = 1
    if lambda_unit == 'um':
        multiplier = 1000
    with open(csv_file) as csvFile:
        csvReader = csv.reader(csvFile)
        for row in csvReader:
            if row[0] == '':
                break
            raw_wl.append(float(row[0])*multiplier)
            raw_data.append(complex(float(row[1]), float(row[2])))

    
    return np.interp(lambda_list,raw_wl,raw_data)

def make_nx2x2_array(n, a, b, c, d, **kwargs):
    """
    Makes a nx2x2 tensor [[a,b],[c,d]] x n
    """

    my_array = torch.zeros((n, 2, 2), **kwargs)
    my_array[:, 0, 0] = a
    my_array[:, 0, 1] = b
    my_array[:, 1, 0] = c
    my_array[:, 1, 1] = d
    return my_array


class TMM_predictor(nn.Module):
    def __init__(self, lambda_list, n_array):
        super(TMM_predictor, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0), requires_grad=False)
        device = self.dummy_param.device
        self.lam_vac_list = torch.tensor(lambda_list, dtype=torch.complex64, device=device)
        self.n_array = torch.tensor(n_array, dtype=torch.complex64, device=device)

        torch.pi = math.pi
        self.num_layers = self.n_array.shape[1]
        self.num_lam = self.lam_vac_list.shape[0]
        self.kz_array = 2 * torch.pi * n_array / self.lam_vac_list.reshape(-1,1)
        self.kz_array = self.kz_array.to(device).to(torch.complex64)
        self.t_array = torch.zeros((self.num_lam , self.num_layers), dtype=torch.complex64, device=device)
        self.r_array = torch.zeros((self.num_lam , self.num_layers), dtype=torch.complex64, device=device)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.kz_array = self.kz_array.to(*args, **kwargs)
        self.t_array = self.t_array.to(*args, **kwargs)
        self.r_array = self.r_array.to(*args, **kwargs)
        self.n_array = self.n_array.to(*args, **kwargs)
        return self        
        
    def forward(self, d_array:torch.Tensor):
        device = self.dummy_param.device

        d_array = d_array.to(torch.complex64)

        num_f = d_array.shape[0]
        delta = torch.zeros((num_f, self.num_lam, self.num_layers), dtype=torch.complex64, device=device)
        delta[:,:,[0,-1]] = torch.nan+1j*torch.nan
        for l in range(1,self.num_layers-1):
            delta[:,:,l] = torch.matmul( d_array[:,l-1].reshape(-1,1) , self.kz_array[:,l].reshape(1,-1))

        delta = delta.reshape(num_f * self.num_lam, self.num_layers)

        for i in range(self.num_layers-1):
            self.t_array[:,i] = 2 * self.n_array[:,i] / (self.n_array[:,i] + self.n_array[:,i+1]) 
            self.r_array[:,i] = ((self.n_array[:,i] - self.n_array[:,i+1] ) / (self.n_array[:,i]  + self.n_array[:,i+1]))

        t_array = self.t_array.repeat(num_f,1)
        r_array = self.r_array.repeat(num_f,1)

        Mtilde_array = make_nx2x2_array(num_f * self.num_lam,1, 0, 0, 1, dtype=torch.complex64, device=device)
        for i in range(1, self.num_layers-1):
            _m = (1/t_array[:,i].reshape(-1,1,1)) * torch.matmul(
                make_nx2x2_array(num_f * self.num_lam, torch.exp(-1j*delta[:,i]), 0, 0, torch.exp(1j*delta[:,i]), dtype=torch.complex64, device=device),
                make_nx2x2_array(num_f * self.num_lam, 1, r_array[:,i], r_array[:,i], 1, dtype=torch.complex64, device=device)
            )

            Mtilde_array = torch.matmul(Mtilde_array, _m)
        # Mtilde = np.dot(make_2x2_array(1, r_list[0,1], r_list[0,1], 1,
        #                                dtype=complex)/t_list[0,1], Mtilde)
        Mtilde_array = torch.matmul(
            make_nx2x2_array(num_f * self.num_lam, 1, r_array[:,0], r_array[:,0], 1, dtype=torch.complex64, device=device)
            /t_array[:,0].reshape(-1,1,1),
            Mtilde_array
        )

        # Net complex transmission and reflection amplitudes
        # r = Mtilde[1,0]/Mtilde[0,0]
        # t = 1/Mtilde[0,0]

        t_list = 1/Mtilde_array[:,0,0]

        # Net transmitted and reflected power, as a proportion of the incoming light
        # power.
        _n = self.n_array.repeat(num_f,1)
        _r = _n[:,-1] / _n[:,0]
        T = abs(t_list**2) * _r
        del _n, _r, t_list, Mtilde_array, t_array, r_array, delta, d_array,

        return T.real.reshape(num_f, self.num_lam)

if __name__=='__main__':
    import argparse
    import json
    import scipy.io as sio

    parser = argparse.ArgumentParser(description='Calculate the transmission of a thin film stack.')
    parser.add_argument('-l','--layers', type=int, help='Number of layers in the stack.', required=True)
    parser.add_argument('-r','--resolution', type=float, help='Spectral Resolution (nm).', required=True)
    parser.add_argument('-t','--thickness',nargs=2, type=float, help='Thickness of each layer.', required=True)
    parser.add_argument('path',metavar='path', nargs='?', type=str, default='.', help='Path to save the model.')

    args = parser.parse_args()

    num_layers = args.layers

    thickness = args.thickness
    min_thickness = thickness[0]
    max_thickness = thickness[1]

    config = {
        'fnet':{
            'TFNum': num_layers,
            'StartWL': 400,
            'EndWL': 1000 + float(args.resolution),
            'Resolution': args.resolution,
            'params_min': min_thickness,
            'params_max': max_thickness
        }
    }

    resolution = args.resolution
    lambda_list = np.arange(400,1000+resolution,resolution)

    parent = Path('nets/fnet')
    folder = args.path/parent/f'TMM_L{num_layers}_R{resolution:.1f}'
    folder.mkdir(parents=True,exist_ok=True)

    # 定义空气
    air = np.ones_like(lambda_list,dtype=complex)

    # 载入材料nk值
    sio2 = load_nk(matPath/'SiO2new.csv',lambda_list,'nm')
    tio2 = load_nk(matPath/'TiO2new.csv',lambda_list,'nm')

    # 载入玻璃nk值
    glass = load_nk(matPath/'bolijingyuan.csv',lambda_list,'nm')


    n_list = [air]
    for j in range(num_layers):
        if j % 2 == 0:
            n_list.append(sio2)
        else:
            n_list.append(tio2)
    n_list.append(glass)

    n_array = np.array(n_list).T

    predictor = TMM_predictor(lambda_list, n_array)

    # export to .pkl
    torch.save(predictor, folder/'fnet.pkl')

    # export to .json
    with open(folder/'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    sio.savemat(folder/'n.mat',{'n':n_array})


