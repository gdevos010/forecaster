import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray import tune

from models.layers.AutoCorrelation import AutoCorrelation
from models.layers.AutoCorrelation import AutoCorrelationLayer
from models.layers.Autoformer_EncDec import Decoder
from models.layers.Autoformer_EncDec import DecoderLayer
from models.layers.Autoformer_EncDec import Encoder
from models.layers.Autoformer_EncDec import EncoderLayer
from models.layers.Autoformer_EncDec import my_Layernorm
from models.layers.Autoformer_EncDec import series_decomp
from models.layers.embedding import DataEmbedding
from models.layers.embedding import DataEmbedding_wo_pos


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(
        self,
        seq_len,
        label_len,
        pred_len,
        output_attention,
        moving_avg,
        enc_in,
        dec_in,
        d_model,
        embedding_type,
        frequency,
        dropout,
        factor,
        n_heads,
        d_ff,
        activation,
        num_encoder_layers,
        num_decoder_layers,
        c_out,
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(
            enc_in, d_model, embedding_type, frequency, dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            dec_in, d_model, embedding_type, frequency, dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for el in range(num_encoder_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True, factor, attention_dropout=dropout, output_attention=False
                        ),
                        d_model,
                        n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False, factor, attention_dropout=dropout, output_attention=False
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for dl in range(num_decoder_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True),
        )

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len :, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--enc_in", type=int, default=7, help="Input size of encoder")
        parser.add_argument("--dec_in", type=int, default=7, help="Input size of decoder")
        parser.add_argument("--c_out", type=int, default=7, help="Output size")
        parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model")
        parser.add_argument("--n_heads", type=int, default=8, help="Number of heads")
        parser.add_argument(
            "--num_encoder_layers", type=int, default=2, help="Number of encoder layers"
        )
        parser.add_argument(
            "--num_decoder_layers", type=int, default=1, help="Number of decoder layers"
        )
        parser.add_argument("--d_ff", type=int, default=2048, help="Dimension of FCN")
        parser.add_argument("--factor", type=int, default=5, help="ProbSparse Attention factor")
        parser.add_argument(
            "--no_distil",
            action="store_true",
            help="Whether to use distilling in the encoder",
        )
        parser.add_argument("--dropout", type=float, default=0.05, help="Dropout probability")

        parser.add_argument(
            "--embedding_type",
            "--embed",
            type=str,
            default="timefeature",
            choices=["timefeature", "fixed", "learned"],
            help="Type of time features encoding",
        )
        parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
        parser.add_argument(
            "--output_attention",
            action="store_true",
            help="Whether to output attention in the encoder",
        )
        parser.add_argument(
            "--moving_avg", type=int, default=25, help="window size of moving average"
        )
        return parser

    @staticmethod
    def get_tuning_params():
        config = {
            "enc_in": tune.choice([5, 7, 9]),
            "dec_in": tune.choice([5, 7, 9]),
            "d_model": tune.choice([32, 256, 512, 1024]),
            "n_heads": tune.choice([1, 4, 8, 16]),
            "num_encoder_layers": tune.choice([2, 3, 4]),
            "num_decoder_layers": tune.choice([1, 2, 3]),
            "d_ff": tune.choice([128, 1024, 2048, 4096, 8192]),
            "dropout": tune.choice([0.01, 0.05, 0.1, 0.15]),
            "activation": tune.choice(["gelu", "relu"]),
        }
        return config
