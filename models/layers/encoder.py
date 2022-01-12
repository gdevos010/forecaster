import torch
import torch.nn as nn
import torch.nn.functional as F


# GD does not match TCCT
class SelfAttentionDistil(nn.Module):
    def __init__(self, c_in, d):
        super().__init__()
        # GD not sure if this section is correct
        self.conv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=1, padding_mode="circular")
        self.pad1 = nn.Conv1d(
            in_channels=c_in, out_channels=c_in, kernel_size=1, padding=0, stride=1
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.d = d
        self.dropout = nn.Dropout(0.1)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # print(self.d)
        if self.d == 1:
            x = self.conv(x.permute(0, 2, 1))
            x = self.norm(x)
            x = self.activation(x)
            x = self.max_pool(x)
            x = torch.transpose(x, 1, 2)
            return x
        elif self.d == 2:
            x_i = x.clone()
            x_p = x.permute(0, 2, 1)
            x1 = x[:, 0::2, :]
            x1_p1 = self.conv(x1.permute(0, 2, 1))
            x1_p2 = self.pad1(x1[:, 0:2, :].permute(0, 2, 1))
            x1_p = torch.cat((x1_p1, x1_p2), 2)
            x2 = x[:, 1::2, :]
            x2_p1 = self.conv(x2.permute(0, 2, 1))
            x2_p2 = self.pad1(x2[:, 0:2, :].permute(0, 2, 1))
            x2_p = torch.cat((x2_p1, x2_p2), 2)
            for i in range(x_p.shape[2]):
                if i % 2 == 0:
                    x_p[:, :, i] = x1_p[:, :, i // 2]
                else:
                    x_p[:, :, i] = x2_p[:, :, i // 2]
            x = self.norm(x_p)
            x = self.dropout(self.activation(x))
            x = x + x_i.permute(0, 2, 1)
            x = self.max_pool(x)
            x = x.transpose(1, 2)
            return x
        else:
            x_i = x.clone()
            x_p = x.permute(0, 2, 1)
            for i in range(self.d):
                x1 = x[:, i :: self.d, :]
                x1_p1 = self.downConv(x1.permute(0, 2, 1))
                x1_p2 = self.pad1(x1[:, 0:2, :].permute(0, 2, 1))
                x1_p = torch.cat((x1_p1, x1_p2), 2)
                for j in range(x_p.shape[2]):
                    if j % self.d == i:
                        x_p[:, :, j] = x1_p[:, :, j // self.d]
            x = self.norm(x_p)
            x = self.dropout(self.activation(x))
            x = x + x_i.permute(0, 2, 1)
            x = self.max_pool(x)
            x = x.transpose(1, 2)
            return x


class EncoderLayer(nn.Module):
    # TODO change activation to pass function instead of string
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", ECSP=False):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_model // 2, out_channels=d_model // 2, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model // 2) if ECSP else nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.csp = ECSP

    def forward(self, x, attention_mask=None):
        # x [B, L, D]

        # split in half
        if self.csp:
            # print('using csp')
            split_x = torch.split(x, x.shape[2] // 2, dim=2)
            csp_x = split_x[1].clone()
            norm_x = split_x[0].clone()
            norm_x = self.conv3(norm_x.permute(0, 2, 1))
            norm_x = norm_x.transpose(1, 2)
            new_x, attention = self.attention(csp_x, csp_x, csp_x, attention_mask=attention_mask)
            csp_x = csp_x + self.dropout(new_x)
            csp_x = self.norm1(csp_x)
            x = torch.cat((csp_x, norm_x), 2)

            y = x
        else:
            new_x, attention = self.attention(x, x, x, attention_mask=attention_mask)
            x = x + self.dropout(new_x)

            y = x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attention


class FocusLayer(nn.Module):
    # Focus l information into d-space
    def __init__(self, c1, c2, k=1):
        super().__init__()
        # self.conv = nn.Conv1d(in_channels=c1*2, out_channels=c2, kernel_size=1)

    def forward(self, x):  # x(b,d,l) -> y(b,2d,l/2)
        return torch.cat([x[..., ::2], x[..., 1::2]], dim=1)


class Encoder(nn.Module):
    def __init__(
        self,
        attention_layers,
        conv_layers=None,
        norm_layer=None,
        Focus_layer=None,
        Passthrough_layer=None,
    ):
        super().__init__()
        self.passnum = len(attention_layers)
        self.attention_layers = nn.ModuleList(attention_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.f_F = Focus_layer
        self.passthrough = Passthrough_layer

    def forward(self, x, attention_mask=None):
        attentions = []
        x_out_list = []
        if self.conv_layers is not None:
            if self.f_F is not None:
                print("using focus")
                i = self.passnum
                for attention_layer, conv_layer in zip(self.attention_layers, self.conv_layers):
                    x, attention = attention_layer(x, attention_mask=attention_mask)
                    i -= 1
                    x_out = (x.clone()).permute(0, 2, 1)
                    for _ in range(i):
                        x_out = self.f_F(x_out)
                    x_out_list.append(x_out.transpose(1, 2))
                    x = conv_layer(x)
                    attentions.append(attention)
                x, attention = self.attention_layers[-1](x)
                x_out_list.append(x)
                attentions.append(attention)
            else:
                for attention_layer, conv_layer in zip(self.attention_layers, self.conv_layers):
                    x, attention = attention_layer(x, attention_mask=attention_mask)
                    x = conv_layer(x)
                    attentions.append(attention)
                x, attention = self.attention_layers[-1](x)
                attentions.append(attention)
        else:
            for attention_layer in self.attention_layers:
                x, attention = attention_layer(x, attention_mask=attention_mask)
                attentions.append(attention)
        if (self.passthrough is not None) and (self.conv_layers is not None):
            x_pass = torch.cat(x_out_list, -1)
            x_pass = x_pass.permute(0, 2, 1)
            x_final = self.passthrough(x_pass)
            x = x_final.transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, attentions


class EncoderStack(nn.Module):
    def __init__(self, encoders):
        super(EncoderStack).__init__()
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x, attention_mask=None):
        inp_len = x.size(1)
        x_stack = []
        attentions = []
        for encoder in self.encoders:
            if encoder is None:
                inp_len //= 2
                continue
            x, attention = encoder(x[:, -inp_len:, :])
            x_stack.append(x)
            attentions.append(attention)
            inp_len //= 2
        x_stack = torch.cat(x_stack, -2)
        return x_stack, attentions
