"""
quant_layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .quantizer import *


def odd_symm_quant(input, nbit, dequantize=True, posQ=False):
    z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}

    alpha_w = get_scale(input, z_typical[f'{nbit}bit']).item()
    output = input.clamp(-alpha_w, alpha_w)

    if posQ:
        output = output + alpha_w

    scale, zero_point = symmetric_linear_quantization_params(nbit, abs(alpha_w), restrict_qrange=True)

    output = linear_quantize(output, scale, zero_point)
    
    if dequantize:
        output = linear_dequantize(output, scale, zero_point)

    return output, alpha_w, scale

def activation_quant(input, nbit, sat_val, dequantize=True):
    with torch.no_grad():
        scale, zero_point = quantizer(nbit, 0, sat_val)
    
    output = linear_quantize(input, scale, zero_point)

    if dequantize:
        output = linear_dequantize(output, scale, zero_point)

    return output, scale

class sawb_w2_Func(torch.autograd.Function):

    def __init__(self, alpha):
        super(sawb_w2_Func, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        self.save_for_backward(input)
        
        output = input.clone()
        output[input.ge(self.alpha - self.alpha/3)] = self.alpha
        output[input.lt(-self.alpha + self.alpha/3)] = -self.alpha
        
        output[input.lt(self.alpha - self.alpha/3)*input.ge(0)] = self.alpha/3
        output[input.ge(-self.alpha + self.alpha/3)*input.lt(0)] = -self.alpha/3

        return output
    
    def backward(self, grad_output):
    
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0

        return grad_input

class sawb_ternFunc(torch.autograd.Function):
    def __init__(self, th):
        super(sawb_ternFunc,self).__init__()
        self.th = th

    def forward(self, input):
        self.save_for_backward(input)

        # self.th = self.tFactor*max_w #threshold
        output = input.clone().zero_()
        output[input.ge(self.th - self.th/2)] = self.th
        output[input.lt(-self.th + self.th/2)] = -self.th

        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class zero_skp_quant(torch.autograd.Function):
    def __init__(self, nbit, coef, group_ch):
        super(zero_skp_quant, self).__init__()
        self.nbit = nbit
        self.coef = coef
        self.group_ch = group_ch
    
    def forward(self, input):
        self.save_for_backward(input)

        # alpha_w_original = get_scale(input, z_typical[f'{int(self.nbit)}bit'])
        alpha_w_original = get_scale_2bit(input)
        interval = 2*alpha_w_original / (2**self.nbit - 1) / 2
        self.th = self.coef * interval

        cout = input.size(0)
        cin = input.size(1)
        kh = input.size(2)
        kw = input.size(3)
        num_group = (cout * cin) // self.group_ch

        w_t = input.view(num_group, self.group_ch*kh*kw)

        grp_values = w_t.norm(p=2, dim=1)                                               # L2 norm
        mask_1d = grp_values.gt(self.th*self.group_ch*kh*kw).float()
        # mask_dropout = torch.rand_like(mask_1d_small).lt(self.prob).float()

        # apply the probablistic sampling:
        # mask_1d = 1 - mask_1d_small * mask_dropout
        mask_2d = mask_1d.view(w_t.size(0),1).expand(w_t.size()) 

        w_t = w_t * mask_2d

        non_zero_idx = torch.nonzero(mask_1d).squeeze(1)                             # get the indexes of the nonzero groups
        non_zero_grp = w_t[non_zero_idx]                                             # what about the distribution of non_zero_group?
        
        weight_q = non_zero_grp.clone()
        alpha_w = get_scale_2bit(weight_q)

        weight_q[non_zero_grp.ge(alpha_w - alpha_w/3)] = alpha_w
        weight_q[non_zero_grp.lt(-alpha_w + alpha_w/3)] = -alpha_w
        
        weight_q[non_zero_grp.lt(alpha_w - alpha_w/3)*non_zero_grp.ge(0)] = alpha_w/3
        weight_q[non_zero_grp.ge(-alpha_w + alpha_w/3)*non_zero_grp.lt(0)] = -alpha_w/3

        w_t[non_zero_idx] = weight_q
        
        output = w_t.clone().resize_as_(input)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class ClippedReLU(nn.Module):
    def __init__(self, num_bits, alpha=8.0, inplace=False, dequantize=True):
        super(ClippedReLU, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]))     
        self.num_bits = num_bits
        self.inplace = inplace
        self.dequantize = dequantize
        
    def forward(self, input):
        # print(f'ClippedRELU: input mean: {input.mean()} | input std: {input.std()}')
        input = F.relu(input)
        input = torch.where(input < self.alpha, input, self.alpha)
        
        with torch.no_grad():
            scale, zero_point = quantizer(self.num_bits, 0, self.alpha)
        input = STEQuantizer.apply(input, scale, zero_point, self.dequantize, self.inplace)
        return input

    def extra_repr(self):
        return super(ClippedReLU, self).extra_repr() + 'nbit={}, alpha_init={}'.format(self.num_bits, self.alpha.detach().item())

class ClippedHardTanh(nn.Module):
    def __init__(self, num_bits, alpha=8.0, inplace=False, dequantize=True):
        super(ClippedHardTanh, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=True)     
        self.num_bits = num_bits
        self.inplace = inplace
        self.dequantize = dequantize
        
    def forward(self, input):
        input = F.hardtanh(input)
        with torch.no_grad():
            scale, zero_point = symmetric_linear_quantization_params(self.num_bits, 1.0, False)
        input = STEQuantizer_weight.apply(input, scale, zero_point, self.dequantize, self.inplace, self.num_bits, False)
        print(torch.unique(input))
        return input

    # def extra_repr(self):
    #     return super(ClippedHardTanh, self).extra_repr() + 'nbit={}, alpha_init={}'.format(self.num_bits, self.alpha.detach().item())
    
class clamp_conv2d(nn.Conv2d):

    def forward(self, input):
        
        num_bits = [2]
        z_typical = {'2bit': [0.311, 0.678], '4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}

        # w_mean = self.weight.mean()

        weight_c = self.weight

        q_scale = get_scale(self.weight, z_typical[f'{num_bits[0]}bit']).item()

        weight_th = weight_c.clamp(-q_scale, q_scale)

        weight_th = q_scale * weight_th / 2 / torch.max(torch.abs(weight_th)) + q_scale / 2

        scale, zero_point = quantizer(num_bits[0], 0, abs(q_scale))
        weight_th = STEQuantizer.apply(weight_th, scale, zero_point, True, False)

        weight_q = 2 * weight_th - q_scale

        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output

class sawb_w2_Func(torch.autograd.Function):

    def __init__(self, alpha):
        super(sawb_w2_Func, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        self.save_for_backward(input)
        
        output = input.clone()
        output[input.ge(self.alpha - self.alpha/3)] = self.alpha
        output[input.lt(-self.alpha + self.alpha/3)] = -self.alpha
        
        output[input.lt(self.alpha - self.alpha/3)*input.ge(0)] = self.alpha/3
        output[input.ge(-self.alpha + self.alpha/3)*input.lt(0)] = -self.alpha/3

        return output
    
    def backward(self, grad_output):
    
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0

        return grad_input

class int_quant_func(torch.autograd.Function):
    def __init__(self, nbit, alpha_w, restrictRange=True, ch_group=16, push=False):
        super(int_quant_func, self).__init__()
        self.nbit = nbit
        self.restrictRange = restrictRange
        self.alpha_w = alpha_w
        self.ch_group = ch_group
        self.push = push

    def forward(self, input):
        self.save_for_backward(input)
        output = input.clamp(-self.alpha_w.item(), self.alpha_w.item())
        scale, zero_point = symmetric_linear_quantization_params(self.nbit, self.alpha_w, restrict_qrange=self.restrictRange)
        output = STEQuantizer_weight.apply(output, scale, zero_point, True, False, self.nbit, self.restrictRange)   

        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class int_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, nbit=4, mode='mean', k=2, ch_group=16, push=False):
        super(int_conv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.nbit = nbit
        self.mode = mode
        self.k = k
        self.ch_group = ch_group
        self.push = push
        self.iter = 0
        self.mask = torch.ones_like(self.weight).cuda()

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(w_l, z_typical[f'{int(self.nbit)}bit'])
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")                         
            
        weight_q = int_quant_func(nbit=self.nbit, alpha_w=self.alpha_w, restrictRange=True, ch_group=self.ch_group, push=self.push)(w_l)
        w_p = weight_q.clone()
        num_group = w_p.size(0) * w_p.size(1) // self.ch_group
        # if self.push and self.iter is 0:
        if self.push and (self.iter+1) % 4000 == 0:
            print("Inference Prune!")
            kw = weight_q.size(2)
            num_group = w_p.size(0) * w_p.size(1) // self.ch_group
            w_p = w_p.contiguous().view((num_group, self.ch_group, kw, kw))
            
            self.mask = torch.ones_like(w_p)

            for j in range(num_group):
                idx = torch.nonzero(w_p[j, :, :, :])
                r = len(idx) / (self.ch_group * kw * kw)
                internal_sparse = 1 - r

                if internal_sparse >= 0.85 and internal_sparse != 1.0:
                    # print(internal_sparse)
                    self.mask[j, :, :, :] = 0.0

            w_p = w_p * self.mask
            w_p = w_p.contiguous().view((num_group, self.ch_group * kw * kw))
            grp_values = w_p.norm(p=2, dim=1)
            non_zero_idx = torch.nonzero(grp_values) 
            num_nonzeros = len(non_zero_idx)
            zero_groups = num_group - num_nonzeros 
            # print(f'zero groups = {zero_groups}')

            self.mask = self.mask.clone().resize_as_(weight_q)
        
        if not self.push:
            self.mask = torch.ones_like(self.weight)
        
        weight_q = self.mask * weight_q

        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.iter += 1
        return output
    
    def extra_repr(self):
        return super(int_conv2d, self).extra_repr() + ', nbit={}, mode={}, k={}, ch_group={}, push={}'.format(self.nbit, self.mode, self.k, self.ch_group, self.push)


class int_linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, nbit=8, mode='mean', k=2, ch_group=16, push=False):
        super(int_linear, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.nbit=nbit
        self.mode = mode
        self.k = k
        self.ch_group = ch_group
        self.push = push

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(w_l, z_typical[f'{int(self.nbit)}bit'])
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")                         
            
        weight_q = int_quant_func(nbit=self.nbit, alpha_w=self.alpha_w, restrictRange=True, ch_group=self.ch_group, push=self.push)(w_l)
        output = F.linear(input, weight_q, self.bias)

        w_tmp = weight_q.clone()
        grp_val = w_tmp.norm(p=2, dim=1)
        num_non_zero = len(torch.nonzero(grp_val.contiguous().view(-1)))
        num_zero_grp = w_tmp.size(0) - num_non_zero

        # output vector
        out_tmp = F.linear(input, weight_q, bias=torch.Tensor([0]).cuda())
        non_zero_output = len(torch.nonzero(out_tmp.contiguous().view(-1)))
        return output

    def extra_repr(self):
        return super(int_linear, self).extra_repr() + ', nbit={}, mode={}, k={}, ch_group={}, push={}'.format(self.nbit, self.mode, self.k, self.ch_group, self.push)

"""
2-bit quantization
"""

def w2_quant(input, mode='mean', k=2):
    if mode == 'mean':
            alpha_w = k * input.abs().mean()
    elif mode == 'sawb':
        alpha_w = get_scale_2bit(w_l)
    else:
        raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")
    
    output = input.clone()
    output[input.ge(alpha_w - alpha_w/3)] = alpha_w
    output[input.lt(-alpha_w + alpha_w/3)] = -alpha_w

    output[input.lt(alpha_w - alpha_w/3)*input.ge(0)] = alpha_w/3
    output[input.ge(-alpha_w + alpha_w/3)*input.lt(0)] = -alpha_w/3

    return output

class sawb_w2_Func(torch.autograd.Function):
    def __init__(self, alpha_w):
        super(sawb_w2_Func, self).__init__()
        self.alpha_w = alpha_w 

    def forward(self, input):
        self.save_for_backward(input)
        
        output = input.clone()
        output[input.ge(self.alpha_w - self.alpha_w/3)] = self.alpha_w
        output[input.lt(-self.alpha_w + self.alpha_w/3)] = -self.alpha_w

        output[input.lt(self.alpha_w - self.alpha_w/3)*input.ge(0)] = self.alpha_w/3
        output[input.ge(-self.alpha_w + self.alpha_w/3)*input.lt(0)] = -self.alpha_w/3

        return output
    
    def backward(self, grad_output):
    
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0

        return grad_input

class Conv2d_2bit(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, mode='sawb', k=2):
        super(Conv2d_2bit, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.mode = mode
        self.k = k
        self.alpha_w = 1.

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            self.alpha_w = get_scale_2bit(w_l)
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ") 

        weight = sawb_w2_Func(alpha_w=self.alpha_w)(w_l)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output

class Linear2bit(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, mode='mean', k=2):
        super(Linear2bit, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.mode = mode
        self.k = k

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            self.alpha_w = get_scale_2bit(w_l)
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ") 
        
        weight_q = sawb_w2_Func(alpha_w=self.alpha_w)(w_l)
        output = F.linear(input, weight_q, self.bias)
        return output

    def extra_repr(self):
        return super(Linear2bit, self).extra_repr() + ', mode={}, k={}'.format(self.mode, self.k)

class HardTanh2bit(nn.Module):
    def __init__(self, num_bits, inplace=False, dequantize=True):
        super(HardTanh2bit, self).__init__()
        self.num_bits = num_bits
        self.inplace = inplace
        self.dequantize = dequantize
    def forward(self, input):
        input = F.hardtanh(input)
        input_q = sawb_w2_Func(alpha_w=1.0)(input)
        return input_q
    def extra_repr(self):
        return super(HardTanh2bit, self).extra_repr() + 'nbit={}, inplace={}'.format(self.num_bits, self.inplace)