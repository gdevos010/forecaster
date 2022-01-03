import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.attention import AttentionLayer
from models.layers.attention import FullAttention
from models.layers.attention import LogSparseAttention
from models.layers.attention import ProbSparseAttention
from models.layers.embedding import DataEmbedding
from models.layers.ydecoder import Decoder
from models.layers.ydecoder import DecoderLayer
from models.layers.ydecoder import DeConvLayer
from models.layers.ydecoder import YformerDecoder
from models.layers.ydecoder import YformerDecoderLayer
from models.layers.yencoder import ConvLayer
from models.layers.yencoder import Encoder
from models.layers.yencoder import EncoderLayer
from models.layers.yencoder import EncoderStack
from models.layers.yencoder import YformerEncoder

debug = False


class Yformer(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn="prob",
        embed="fixed",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        device=torch.device("cuda:0"),
        **kwargs
    ):
        super().__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        # TODO: change the embedding so that there is a simple shared embedding for timestamp
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.fut_enc_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbSparseAttention if attn == "prob" else FullAttention
        # Encoder
        self.encoder = YformerEncoder(
            [
                # uses probSparse attention
                EncoderLayer(
                    AttentionLayer(
                        Attn(
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
                    dropout=dropout,
                    activation=activation,
                )
                for el in range(e_layers)
            ],
            [ConvLayer(d_model) for el in range(e_layers)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Future encoder
        self.future_encoder = YformerEncoder(
            [
                # uses masked attention
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for el in range(e_layers)
            ],
            [ConvLayer(d_model) for el in range(e_layers)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Decoder
        self.ydecoder = YformerDecoder(
            [
                # single attention block in the decoder compared to 2 in the informer
                YformerDecoderLayer(
                    AttentionLayer(
                        Attn(
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
                    dropout=dropout,
                    activation=activation,
                )
                for dl in range(d_layers)
            ],
            [DeConvLayer(d_model) for dl in range(d_layers)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.seq_len_projection = nn.Linear(
            d_model, c_out, bias=True
        )  # (bs, 336, 512) -> (bs, 336 + 336, 7)
        self.pred_len_projection = nn.Linear(
            d_model, c_out, bias=True
        )  # (bs, 336, 512) -> (bs, 336 + 336, 7)

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

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns, x_list = self.encoder(enc_out, attention_mask=enc_self_mask)
        x_list.reverse()
        # print("input shape x_dec, x_mark_dec",  x_dec.shape, x_mark_dec.shape)

        # Future Encoder
        fut_enc_out = self.fut_enc_embedding(x_dec, x_mark_dec)
        fut_enc_out, attns, fut_x_list = self.future_encoder(fut_enc_out, attention_mask=enc_self_mask)
        fut_x_list.reverse()

        # Decoder
        dec_out, attns = self.ydecoder(x_list, fut_x_list, attention_mask=dec_self_mask)

        seq_len_dec_out = self.pred_len_projection(dec_out)[:, -self.seq_len :, :]
        pre_len_dec_out = self.seq_len_projection(dec_out)[:, -self.pred_len :, :]

        dec_out = torch.cat((seq_len_dec_out, pre_len_dec_out), dim=1)  # 336 -> 336 + 336
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B, L, D]

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
