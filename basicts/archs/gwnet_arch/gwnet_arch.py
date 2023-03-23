import torch
from torch import nn
import torch.nn.functional as F
from .mlp import MultiLayerPerceptron
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

    def __init__(self, c_in=32, c_out=32, dropout=0., support_len=3, order=2):
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
    
class GraphWaveNet(nn.Module):
    """
    Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling
    Link: https://arxiv.org/abs/1906.00121
    Ref Official Code: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
    """

    def __init__(self, num_nodes, dropout=0.3, supports=None,
                    gcn_bool=True, addaptadj=True, aptinit=None,
                    in_dim=3, out_dim=12, residual_channels=32,
                    dilation_channels=32, skip_channels=128, end_channels=512,
                    kernel_size=2, blocks=1, layers=2):
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv_main = nn.Conv2d(in_channels=in_dim-2,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 3),padding='same')
        self.start_conv_con0 = nn.Conv2d(in_channels=in_dim-2,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 3),padding='same')
        self.start_conv_con1 = nn.Conv2d(in_channels=in_dim-2,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 3),padding='same')
        self.cat0 = nn.Conv2d(in_channels=2*residual_channels,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 3),padding='same')
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.gate_0 = nn.Sigmoid()
        self.gate_1 = nn.Sigmoid()

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Feedforward function of Graph WaveNet.

        Args:
            history_data (torch.Tensor): shape [B, L, N, C]

        Returns:
            torch.Tensor: [B, L, N, 1]
        """

        input = history_data.transpose(1, 3).contiguous()
        in_len = input.size(3)
        if in_len < self.receptive_field:
            y = nn.functional.pad(
                input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            y = input
        x_main = self.start_conv_main(torch.cat([y[:,0:1,:,:],y[:,3:4,:,:],y[:,4:5,:,:]], dim=1))
        x = x_main
        x_0 = self.gate_0(self.start_conv_con0(torch.cat([y[:,1:2,:,:],y[:,3:4,:,:],y[:,4:5,:,:]], dim=1)))*x_main
        x_1 = self.gate_1(self.start_conv_con1(torch.cat([y[:,2:3,:,:],y[:,3:4,:,:],y[:,4:5,:,:]], dim=1)))*x_main
        x = self.cat0(torch.cat([x_0,x_1],dim=1))+x
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        # x = F.relu(self.end_conv_1(x))
        # x = self.end_conv_2(x)
        return x
    
class STID(nn.Module):
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
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.supports = [i.cuda() for i in model_args["supports"]]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        
        # spatial embeddings
        self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb)
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_day_in_week) + \
            self.temp_dim_diw*int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(128+32, 128+32) for _ in range(3)])
        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=128+32, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        
        self.graphwavenet = GraphWaveNet(num_nodes = 307, supports=self.supports, in_dim=5)
        self.gcn = gcn(support_len=2,order=1)
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=128 * 9, out_channels=128, kernel_size=(1, 1), bias=True)
   
    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """

        batch_size, _, num_nodes, _ = history_data.shape
        node_emb = self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
        # node_emb = self.gcn(node_emb, self.supports)
        x = self.graphwavenet(history_data)
        batch_size, _, num_nodes, _ = x.shape
        x = x.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(x)
        x = torch.cat([time_series_emb,node_emb],dim=1)
        x = self.encoder(x)
        prediction = self.regression_layer(x)
        
        return prediction
