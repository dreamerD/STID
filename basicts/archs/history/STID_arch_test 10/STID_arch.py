import torch
from torch import nn
import torch.nn.functional as F
from .mlp import MultiLayerPerceptron


class nconv(nn.Module):
    """Graph conv operation."""

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        if len(A.shape) == 2:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        else:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
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
        # self.bn = torch.nn.BatchNorm2d(128)

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
                 in_dim=5, out_dim=32, residual_channels=8,
                 dilation_channels=8, skip_channels=32, end_channels=32,
                 kernel_size=2, blocks=4, layers=2):
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs_r = nn.ModuleList()
        self.gate_convs_r = nn.ModuleList()
        self.residual_convs_r = nn.ModuleList()
        self.skip_convs_r = nn.ModuleList()
        self.bn_r = nn.ModuleList()

        self.filter_convs_i = nn.ModuleList()
        self.gate_convs_i = nn.ModuleList()
        self.residual_convs_i = nn.ModuleList()
        self.skip_convs_i = nn.ModuleList()
        self.bn_i = nn.ModuleList()

        self.filter_convs_j = nn.ModuleList()
        self.gate_convs_j = nn.ModuleList()
        self.residual_convs_j = nn.ModuleList()
        self.skip_convs_j = nn.ModuleList()
        self.bn_j = nn.ModuleList()

        self.filter_convs_k = nn.ModuleList()
        self.gate_convs_k = nn.ModuleList()
        self.residual_convs_k = nn.ModuleList()
        self.skip_convs_k = nn.ModuleList()
        self.bn_k = nn.ModuleList()

        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1), padding='same')
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
            channels = 0
            for i in range(layers):
                # dilated convolutions
                if i == 0 and b == 0:
                    channels = 1
                else:
                    channels = residual_channels
                self.filter_convs_r.append(nn.Conv2d(in_channels=channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs_r.append(nn.Conv1d(in_channels=channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs_r.append(nn.Conv1d(in_channels=residual_channels,
                                                       out_channels=residual_channels,
                                                       kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs_r.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=skip_channels,
                                                   kernel_size=(1, 1)))
                self.bn_r.append(nn.BatchNorm2d(residual_channels))

                # dilated convolutions
                self.filter_convs_i.append(nn.Conv2d(in_channels=channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs_i.append(nn.Conv1d(in_channels=channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs_i.append(nn.Conv1d(in_channels=residual_channels,
                                                       out_channels=residual_channels,
                                                       kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs_i.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=skip_channels,
                                                   kernel_size=(1, 1)))
                self.bn_i.append(nn.BatchNorm2d(residual_channels))

                # dilated convolutions
                self.filter_convs_j.append(nn.Conv2d(in_channels=channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs_j.append(nn.Conv1d(in_channels=channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs_j.append(nn.Conv1d(in_channels=residual_channels,
                                                       out_channels=residual_channels,
                                                       kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs_j.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=skip_channels,
                                                   kernel_size=(1, 1)))
                self.bn_j.append(nn.BatchNorm2d(residual_channels))

                # dilated convolutions
                self.filter_convs_k.append(nn.Conv2d(in_channels=channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs_k.append(nn.Conv1d(in_channels=channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs_k.append(nn.Conv1d(in_channels=residual_channels,
                                                       out_channels=residual_channels,
                                                       kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs_k.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=skip_channels,
                                                   kernel_size=(1, 1)))
                self.bn_k.append(nn.BatchNorm2d(residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=4*skip_channels,
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
        input = history_data.transpose(1, 3).contiguous()
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(
                input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input
        # x = self.start_conv(x)
        skip_r = 0
        skip_i = 0
        skip_j = 0
        skip_k = 0

        x_r = torch.zeros_like(x[:, 0:1, :, :])
        x_i = x[:, 0:1, :, :]
        x_j = x[:, 1:2, :, :]
        x_k = x[:, 2:3, :, :]
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

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual_r = x_r
            residual_i = x_i
            residual_j = x_j
            residual_k = x_k
            # dilated convolution
            filter_r = self.filter_convs_r[i](
                residual_r) - self.filter_convs_i[i](residual_i)-self.filter_convs_j[i](residual_j)-self.filter_convs_k[i](residual_k)
            filter_r = torch.tanh(filter_r)
            gate_r = self.gate_convs_r[i](
                residual_r) - self.gate_convs_i[i](residual_i)-self.gate_convs_j[i](residual_j)-self.gate_convs_k[i](residual_k)
            gate_r = torch.sigmoid(gate_r)
            x_r = filter_r * gate_r

            # residual_i = x_i
            # dilated convolution
            filter_i = self.filter_convs_i[i](
                residual_r) + self.filter_convs_r[i](residual_i) + self.filter_convs_k[i](residual_j) - self.filter_convs_j[i](residual_k)
            filter_i = torch.tanh(filter_i)
            gate_i = self.gate_convs_i[i](
                residual_r) + self.gate_convs_r[i](residual_i) + self.gate_convs_k[i](residual_j) - self.gate_convs_j[i](residual_k)
            gate_i = torch.sigmoid(gate_i)
            x_i = filter_i * gate_i

            # residual_j = x_j
            # dilated convolution
            filter_j = self.filter_convs_j[i](
                residual_r)+self.filter_convs_r[i](residual_j)+self.filter_convs_i[i](residual_k)+self.filter_convs_k[i](residual_i)
            filter_j = torch.tanh(filter_j)
            gate_j = self.gate_convs_j[i](
                residual_r)+self.gate_convs_r[i](residual_j)+self.gate_convs_i[i](residual_k)-self.gate_convs_k[i](residual_i)
            gate_j = torch.sigmoid(gate_j)
            x_j = filter_j * gate_j

            # residual_k = x_k
            # dilated convolution
            filter_k = self.filter_convs_k[i](residual_r)+self.filter_convs_r[i](
                residual_k)+self.filter_convs_j[i](residual_i)-self.filter_convs_i[i](residual_j)
            filter_k = torch.tanh(filter_k)
            gate_k = self.gate_convs_k[i](residual_r)+self.gate_convs_r[i](
                residual_k)+self.gate_convs_j[i](residual_i)-self.gate_convs_i[i](residual_j)
            gate_k = torch.sigmoid(gate_k)
            x_k = filter_k * gate_k
            # parametrized skip connection

            s_r = x_r
            s_i = x_i
            s_j = x_j
            s_k = x_k
            s_r = self.skip_convs_r[i](s_r)
            s_i = self.skip_convs_i[i](s_i)
            s_j = self.skip_convs_j[i](s_j)
            s_k = self.skip_convs_k[i](s_k)
            try:
                skip_r = skip_r[:, :, :,  -s_r.size(3):]
                skip_i = skip_i[:, :, :,  -s_i.size(3):]
                skip_j = skip_j[:, :, :,  -s_j.size(3):]
                skip_k = skip_k[:, :, :,  -s_k.size(3):]
            except:
                skip_r = 0
                skip_i = 0
                skip_j = 0
                skip_k = 0
            skip_r = s_r + skip_r
            skip_i = s_i + skip_i
            skip_j = s_j + skip_j
            skip_k = s_k + skip_k
            x_r = self.residual_convs_r[i](x_r)
            x_i = self.residual_convs_i[i](x_i)
            x_j = self.residual_convs_j[i](x_j)
            x_k = self.residual_convs_k[i](x_k)

            x_r = x_r + residual_r[:, :, :, -x_r.size(3):]
            x_i = x_i + residual_i[:, :, :, -x_i.size(3):]
            x_j = x_j + residual_j[:, :, :, -x_j.size(3):]
            x_k = x_k + residual_k[:, :, :, -x_k.size(3):]

            x_r = self.bn_r[i](x_r)
            x_i = self.bn_i[i](x_i)
            x_j = self.bn_j[i](x_j)
            x_k = self.bn_k[i](x_k)

        x = F.relu(torch.cat([skip_r, skip_i, skip_j, skip_k], dim=1))
        x = F.relu(self.end_conv_1(x))
        # x = F.relu(self.end_conv_2(x))
        return x


class STID_test(nn.Module):
    """The implementation of CIKM 2022 short paper
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
            in_channels=32, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.time_series_emb_layer1 = nn.Conv2d(
            in_channels=(self.input_dim) * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.time_series_emb_layer2 = nn.Conv2d(
            in_channels=(self.input_dim) * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.fusion = nn.Conv2d(
            self.embed_dim, self.embed_dim, (1, 4), padding="valid")
        self.hidden_dim = 128
        self.encoder0 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(1)])
        self.encoder1 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(1)])
        self.encoder2 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(1)])
        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

        # feature fusion
        self.speed_fusion = nn.Conv2d(
            in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.occupy_fusion = nn.Conv2d(
            in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.speed_gate = nn.Sigmoid()
        self.occupy_gate = nn.Sigmoid()
        self.gcn0 = gcn(support_len=3, order=2)
        self.gcn1 = gcn(support_len=3, order=2)
        self.gcn2 = gcn(support_len=3, order=2)
        self.nodevec1 = nn.Parameter(
            torch.randn(307, 12), requires_grad=True)
        self.nodevec2 = nn.Parameter(
            torch.randn(12, 307), requires_grad=True)
        self.temporal_nodevec1 = nn.Parameter(
            torch.randn(288, 12), requires_grad=True)
        self.week_nodevec1 = nn.Parameter(
            torch.randn(7, 12), requires_grad=True)
        self.temporal_nodevec2 = nn.Parameter(
            torch.randn(288, 12), requires_grad=True)
        self.week_nodevec2 = nn.Parameter(
            torch.randn(7, 12), requires_grad=True)

        self.gw = GraphWaveNet(307)
        self.bn0 = torch.nn.BatchNorm2d(128)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.bn2 = torch.nn.BatchNorm2d(128)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        # """Feed forward of STID.

        # Args:
        #     history_data (torch.Tensor): history data with shape [B, L, N, C]

        # Returns:
        #     torch.Tensor: prediction wit shape [B, L, N, C]
        # """

        # prepare data
        main_input_data = history_data[..., 0:5]
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 3]
            time_in_day_emb = self.time_in_day_emb[(
                t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
            temporal_nodevec1 = self.temporal_nodevec1[(  # (B N 5)
                t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
            temporal_nodevec2 = self.temporal_nodevec2[(  # (B N 5)
                t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 4]
            day_in_week_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
            week_nodevec1 = self.week_nodevec1[(  # (B N 2)
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
            week_nodevec2 = self.week_nodevec2[(  # (B N 2)
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = main_input_data.shape
        main_input_data = self.gw(main_input_data)
        main_input_data = main_input_data.transpose(1, 2).contiguous()
        main_input_data = main_input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        # time_series_emb0 = self.time_series_emb_layer0(main_input_data)

        new_supports = None
        adp = F.softmax(
            F.relu(torch.bmm(self.nodevec1.expand(batch_size, -1, -1)+temporal_nodevec1+week_nodevec1, self.nodevec2.expand(batch_size, -1, -1) + temporal_nodevec1.permute(0, 2, 1)+week_nodevec1.permute(0, 2, 1))), dim=2)
        new_supports = self.supports + [adp]

        time_series_emb = main_input_data
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
        # hidden = self.fusion(hidden)
        # encoding
        hidden = self.encoder0(hidden)
        hidden = self.gcn0(hidden, new_supports) + hidden
        # hidden = self.bn0(hidden)
        hidden = self.encoder1(hidden)
        hidden = self.gcn1(hidden, new_supports) + hidden
        # hidden = self.bn1(hidden)
        hidden = self.encoder2(hidden)
        hidden = self.gcn2(hidden, new_supports) + hidden
        # hidden = self.bn2(hidden)
        # regression
        prediction = self.regression_layer(hidden)

        return prediction
