import torch
import torch.nn as nn
from utils.encoding import tokenizer
from utils.misc import load_model_from_ckpt

class TEIM(nn.Module):
    def __init__(self, config):
        self.dim_hidden = dim_hidden = config.dim_hid # hidden dimension:; [64, 128, 256]
        self.layers_inter = layers_inter = config.layers_inter  # layers of the inter-layer: [2, 3, 4]
        self.dim_seqlevel = dim_seqlevel = config.dim_seqlevel  # dim of seq level output layer: [64, 128]
        self.inter_type =  config.inter_type  # type of the inter-map combination: [cat, add, mul, outer] 
        self.ae_model_cfg =  config.ae_model  # configs of ae model
        super().__init__()

        ## embedding
        embedding = tokenizer.embedding_mat()
        vocab_size, dim_emb = embedding.shape
        self.embedding_module = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), padding_idx=0, )

        ## feature extractor
        self.seq_cdr3 = nn.Sequential(
            nn.Conv1d(dim_emb, dim_hidden, 1,),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
        )
        self.seq_epi =nn.Sequential(
            nn.Conv1d(dim_emb, dim_hidden, 1,),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
        )

        ## interaction map extractor
        if self.inter_type == 'cat':
            self.combine_layer = nn.Conv2d(dim_hidden*2, dim_hidden, 1, bias=False)
        elif self.inter_type == 'outer':
            self.combine_layer = nn.Conv2d(dim_hidden*dim_hidden, dim_hidden, 1, bias=False)

        self.inter_layers = nn.ModuleList([
            nn.Sequential(  # first cnn layer
                ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                nn.BatchNorm2d(dim_hidden),
                nn.ReLU(),
            ),
            nn.ModuleList([  # second layer, this layer add the ae pretrained vector
                ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                nn.Sequential(
                    nn.BatchNorm2d(dim_hidden),
                    nn.ReLU(),
                ),
            ]),
            *[  # more cnn layers
                nn.Sequential(
                ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                nn.BatchNorm2d(dim_hidden),
                nn.ReLU(),
            ) for _ in range(layers_inter - 2)],
        ])
        ##  ae linear
        if self.ae_model_cfg.path != '':
            ae_model = AutoEncoder(self.ae_model_cfg.dim_hid, self.ae_model_cfg.len_epi)
            self.ae_encoder = load_model_from_ckpt(self.ae_model_cfg.path, ae_model)
            for param in self.ae_encoder.parameters():
                param.requires_grad = False
            self.ae_linear = nn.Linear(self.ae_model_cfg.dim_hid, dim_hidden, bias=False)
        else:
            self.ae_encoder = None
        
        ## seq-level prediction
        self.seqlevel_outlyer = nn.Sequential(
            # nn.AdaptiveMaxPool2d(1),
            # nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(dim_seqlevel, 1),
            nn.Sigmoid()
        )

        ## res-level prediction
        self.reslevel_outlyer = nn.Conv2d(
            in_channels=dim_hidden,
            out_channels=2,
            kernel_size=2*layers_inter+1,
            padding=layers_inter
        )
        

    def forward(self, inter_map, addition=None):
        
        ## output layers
        # seq-level prediction
        seqlevel_out = self.seqlevel_outlyer(inter_map)
        # res-level prediction
        reslevel_out = self.reslevel_outlyer(inter_map)
        out_dist = torch.relu(reslevel_out[:, 0, :, :])
        out_bd = torch.sigmoid(reslevel_out[:, 1, :, :])
        reslevel_out = torch.cat([out_dist.unsqueeze(-1), out_bd.unsqueeze(-1)], axis=-1)

        return {
            'seqlevel_out': seqlevel_out,
            'reslevel_out': reslevel_out,
            'inter_map': inter_map,
        }


class ResNet(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def forward(self, data):
        tmp_data = self.cnn(data)
        out = tmp_data + data
        return out


class AutoEncoder(nn.Module):
    def __init__(self, 
        dim_hid,
        len_seq,
    ):
        super().__init__()
        embedding = tokenizer.embedding_mat()
        vocab_size, dim_emb = embedding.shape
        self.embedding_module = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), padding_idx=0, )
        self.encoder = nn.Sequential(
            nn.Conv1d(dim_emb, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.Conv1d(dim_hid, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )

        self.seq2vec = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len_seq * dim_hid, dim_hid),
            nn.ReLU()
        )
        self.vec2seq = nn.Sequential(
            nn.Linear(dim_hid, len_seq * dim_hid),
            nn.ReLU(),
            View(dim_hid, len_seq)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(dim_hid, vocab_size)

    def forward(self, inputs, latent_only=False):
        # inputs = inputs.long()
        seq_emb = self.embedding_module(inputs)
        seq_enc = self.encoder(seq_emb.transpose(1, 2))
        vec = self.seq2vec(seq_enc)
        seq_repr = self.vec2seq(vec)
        seq_dec = self.decoder(seq_repr)
        out = self.out_layer(seq_dec.transpose(1, 2))
        if latent_only:
            return vec
        else:
            return out, seq_enc, vec


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        shape = [input.shape[0]] + list(self.shape)
        return input.view(*shape)