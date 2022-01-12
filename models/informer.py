import argparse

import torch.nn as nn
from ray import tune

from models.layers.attention import AttentionLayer
from models.layers.attention import FullAttention
from models.layers.attention import LogSparseAttention
from models.layers.attention import ProbSparseAttention
from models.layers.decoder import Decoder
from models.layers.decoder import DecoderLayer
from models.layers.embedding import DataEmbedding
from models.layers.encoder import Encoder
from models.layers.encoder import EncoderLayer
from models.layers.encoder import EncoderStack
from models.layers.encoder import FocusLayer
from models.layers.encoder import SelfAttentionDistil


class BaseInformer(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=2,
        d_ff=512,
        dropout=0.0,
        attention_type="prob",
        embedding_type="fixed",
        frequency="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix_attention=False,
        csp=False,
        dilated=False,
        passthrough=False,
        **kwargs
    ):
        super().__init__()
        self.pred_len = out_len
        self.attention_type = attention_type
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(enc_in, d_model, embedding_type, frequency, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embedding_type, frequency, dropout)

        self.csp = 2 if csp else 1

        if attention_type == "prob":
            attention = ProbSparseAttention
            # print('prob')
        elif attention_type == "log":
            attention = LogSparseAttention
            # print('log')
        else:
            attention = FullAttention
            # print('full')

        self.encoder = None
        self.dilated = dilated
        self.passthrough = passthrough

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        attention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model // self.csp,
                        n_heads,
                        mix=mix_attention,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False, factor, attention_dropout=dropout, output_attention=False
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    DCSP=csp,
                )
                for _ in range(num_decoder_layers)
            ],
            nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, c_out)

    def forward(
        self,
        x_enc,
        x_enc_mark,
        x_dec,
        x_dec_mark,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        enc_out = self.enc_embedding(x_enc, x_enc_mark)
        enc_out, attentions = self.encoder(enc_out, attention_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_dec_mark)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attentions
        return dec_out[:, -self.pred_len :, :]

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
            "--attention_type",
            "--attn",
            type=str,
            default="prob",
            choices=["prob", "full", "log"],
            help="Type of attention used in the encoder",
        )
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
            "--mix_attention",
            "--mix",
            action="store_true",
            help="Whether to mix attention in generative decoder",
        )
        parser.add_argument(
            "--csp",
            "--CSP",
            action="store_true",
            help="whether to use CSPAttention, default=False",
            default=False,
        )
        parser.add_argument(
            "--dilated",
            action="store_true",
            help="whether to use dilated causal convolution in encoder, default=False",
            default=False,
        )
        parser.add_argument(
            "--passthrough",
            action="store_true",
            help="whether to use passthrough mechanism in encoder, default=False",
            default=False,
        )

        return parser

    @staticmethod
    def get_tuning_params():
        config = {
            "enc_in": tune.choice([5, 7, 9]),
            "dec_in": tune.choice([5, 7, 9]),
            "d_model": tune.choice([256, 512, 1024]),
            "n_heads": tune.choice([4, 8, 16]),
            "num_encoder_layers": tune.choice([2, 3, 4]),
            "num_decoder_layers": tune.choice([1, 2, 3]),
            "d_ff": tune.choice([1024, 2048, 4096]),
            "dropout": tune.choice([0.01, 0.05, 0.1, 0.15]),
            "attention_type": tune.choice(["prob", "full", "log"]),
            "activation": tune.choice(["gelu", "relu"]),
            "mix": tune.choice([True, False]),
            "csp": tune.choice([True, False]),
            "dilated": tune.choice([True, False]),
            "passthrough": tune.choice([True, False]),
        }
        return config


class Informer(BaseInformer):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=2,
        d_ff=512,
        dropout=0.0,
        attention_type="prob",
        embedding_type="fixed",
        frequency="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix_attention=False,
        csp=False,
        dilated=False,
        passthrough=False,
        **kwargs
    ):
        super().__init__(
            enc_in,
            dec_in,
            c_out,
            out_len,
            factor=factor,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            attention_type=attention_type,
            embedding_type=embedding_type,
            frequency=frequency,
            activation=activation,
            output_attention=output_attention,
            distil=distil,
            mix_attention=mix_attention,
            csp=csp,
            dilated=dilated,
            passthrough=passthrough,
        )

        # Attention
        self.csp = 2 if csp else 1

        if attention_type == "prob":
            attention = ProbSparseAttention
            # print('prob')
        elif attention_type == "log":
            attention = LogSparseAttention
            # print('log')
        else:
            attention = FullAttention
            # print('full')

        self.pconv = 0
        for i in range(num_encoder_layers):
            e_num = 2 ** i
            self.pconv += e_num

        self.dilated = dilated
        self.passthrough = passthrough
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        attention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        (d_model // self.csp),
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    ECSP=csp,
                )
                for _ in range(num_encoder_layers)
            ],
            [
                SelfAttentionDistil(d_model, ((2 ** el) if self.dilated else 1))
                for el in range(num_encoder_layers - 1)
            ]
            if distil
            else None,
            nn.LayerNorm(d_model),
            Focus_layer=FocusLayer(d_model, d_model) if self.passthrough else None,
            Passthrough_layer=nn.Conv1d(self.pconv * d_model, d_model, 1)
            if self.passthrough
            else None,
        )


class InformerStack(BaseInformer):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=2,
        d_ff=512,
        dropout=0.0,
        attention_type="prob",
        embedding_type="fixed",
        frequency="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix_attention=False,
        CSP=False,
        dilated=False,
        passthrough=False,
        **kwargs
    ):
        super().__init__(
            enc_in,
            dec_in,
            c_out,
            out_len,
            factor=factor,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            attention_type=attention_type,
            embedding_type=embedding_type,
            frequency=frequency,
            activation=activation,
            output_attention=output_attention,
            distil=distil,
            mix_attention=mix_attention,
        )
        if attention_type == "prob":
            attention = ProbSparseAttention
            # print('prob')
        elif attention_type == "log":
            attention = LogSparseAttention
            # print('log')
        else:
            attention = FullAttention
            # print('full')

        stacks = list(range(num_encoder_layers, 2, -1))  # customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            attention(
                                False,
                                factor,
                                attention_dropout=dropout,
                                output_attention=output_attention,
                            ),
                            d_model,
                            n_heads,
                            mix=False,
                        ),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for _ in range(el)
                ],
                [SelfAttentionDistil(d_model) for _ in range(el - 1)] if distil else None,
                nn.LayerNorm(d_model),
            )
            for el in stacks
        ]
        self.encoder = EncoderStack(encoders)
