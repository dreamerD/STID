import torch
from torch import nn
import torch.nn.functional as F
from .mlp import MultiLayerPerceptron
import numpy as np
import pywt
class nconv(nn.Module):
    """Graph conv operation."""

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    """Linear layer."""

    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)
    
class gcn(nn.Module):
    """Graph convolution network."""

    def __init__(self, c_in=128, c_out=128, dropout=0., support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a.to(x.device))
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a.to(x.device))
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        # h = F.dropout(h, self.dropout, training=self.training)
        return h
    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride,padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
class STID_test(nn.Module):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.supports = model_args["supports"]

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(288, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(7, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer0 = nn.Conv2d(
            in_channels=(self.input_dim) * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.time_series_emb_layer1 = nn.Conv2d(
            in_channels=(self.input_dim) * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.time_series_emb_layer2 = nn.Conv2d(
            in_channels=(self.input_dim) * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.fusion = nn.Conv2d(self.embed_dim,self.embed_dim,(1,4), padding="valid")
        self.hidden_dim = 128
        self.encoder0 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(1)])
        self.encoder1 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(1)])
        self.encoder2 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(1)])

        self.encoder4 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(1)])
        self.encoder5 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(1)])
        self.encoder6 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(1)])
        
        # regression
        self.regression_layer0 = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        self.regression_layer1 = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)        
        # feature fusion
        self.speed_fusion = nn.Conv2d(
            in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.occupy_fusion = nn.Conv2d(
            in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.speed_gate = nn.Sigmoid()
        self.occupy_gate = nn.Sigmoid()
        self.gcn0 = gcn(support_len=3,order=2)
        self.gcn1 = gcn(support_len=3,order=2)
        self.gcn2 = gcn(support_len=3,order=2)
        self.nodevec1 = nn.Parameter(
                    torch.randn(307, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(
                    torch.randn(10, 307), requires_grad=True)
        # self.dwt = pywt.DWT1DForward(wave='db4', J=2)
        # self.idwt = pywt.DWT1DInverse(wave='db4')
        self.mv = moving_avg(3,1)
    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """

        # prepare data
        main_input_data = history_data[...,0:5]
        # ya0 = history_data[...,0]
        # ya0 = ya0.permute(0,2,1)
        # for i in range(ya0.shape[0]):
        #     for j in range(ya0.shape[1]):
        #         x_test = ya0[i,j,:]
        #         coeffs = pywt.wavedec(x_test.cpu().numpy(), 'db4', level=2) # 2阶小波分解
        #         ya0[i,j,:] = torch.Tensor(pywt.waverec(np.multiply(coeffs, [1, 1, 0]).tolist(), 'db4')).cuda()

        # ya1 = history_data[...,1]
        # ya1 = ya1.permute(0,2,1)
        # for i in range(ya1.shape[0]):
        #     for j in range(ya1.shape[1]):
        #         x_test = ya1[i,j,:]
        #         coeffs = pywt.wavedec(x_test.cpu().numpy(), 'db4', level=2) # 2阶小波分解
        #         ya1[i,j,:] = torch.Tensor(pywt.waverec(np.multiply(coeffs, [1, 1, 0]).tolist(), 'db4')).cuda()
                                
        # ya2 = history_data[...,2]
        # ya2 = ya2.permute(0,2,1)
        # for i in range(ya2.shape[0]):
        #     for j in range(ya2.shape[1]):
        #         x_test = ya2[i,j,:]
        #         coeffs = pywt.wavedec(x_test.cpu().numpy(), 'db4', level=2) # 2阶小波分解
        #         ya2[i,j,:] = torch.Tensor(pywt.waverec(np.multiply(coeffs, [1, 1, 0]).tolist(), 'db4')).cuda()
                


        
        # main_input_data = torch.cat([ya0.permute(0,2,1).unsqueeze(-1),ya1.permute(0,2,1).unsqueeze(-1),ya2.permute(0,2,1).unsqueeze(-1),history_data[...,3:5]],dim=3)
        
        adp = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = self.supports + [adp]
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 3]
            time_in_day_emb = self.time_in_day_emb[(
                t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 4]
            day_in_week_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = main_input_data.shape
        main_input_data = main_input_data.transpose(1, 2).contiguous()
        main_input_data = main_input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb0 = self.time_series_emb_layer0(main_input_data)
        
        time_series_emb = time_series_emb0
        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        hidden = hidden.reshape(hidden.shape[0],hidden.shape[1], hidden.shape[2])
        new_hidden = self.mv(hidden)
        old_hidden = hidden - new_hidden
        new_hidden = new_hidden.reshape(hidden.shape[0],hidden.shape[1], hidden.shape[2],1)
        old_hidden = old_hidden.reshape(hidden.shape[0],hidden.shape[1], hidden.shape[2],1)
        # encoding
        new_hidden = self.encoder0(new_hidden)
        # hidden = self.gcn0(hidden, new_supports) + hidden
        new_hidden = self.encoder1(new_hidden)
        # hidden = self.gcn1(hidden, new_supports) + hidden
        new_hidden = self.encoder2(new_hidden)
        # hidden = self.gcn2(hidden, new_supports) + hidden

        old_hidden = self.encoder4(old_hidden)
        # hidden = self.gcn0(hidden, new_supports) + hidden
        old_hidden = self.encoder5(old_hidden)
        # hidden = self.gcn1(hidden, new_supports) + hidden
        # old_hidden = self.encoder6(old_hidden)
        # hidden = self.gcn2(hidden, new_supports) + hidden
        
        # yh = self.encoder00(yh)
        # hidden = self.gcn0(hidden, new_supports) + hidden
        # yh = self.encoder01(yh)
        # hidden = self.gcn1(hidden, new_supports) + hidden
        # yh = self.encoder02(yh)
        # hidden = self.gcn2(hidden, new_supports) + hidden

        # regression
        prediction = self.regression_layer0(new_hidden) + self.regression_layer1(old_hidden)

        return prediction
