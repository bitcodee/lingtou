import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit import Transformer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x




class TransImgPred(nn.Module):
    
    def __init__(self, seq_len, feature_size, cls_num,
                 nlayers, nhead, nhid=512,
                 dropout=0.1,
                 merge=False):
        super(TransImgPred, self).__init__()
        self.model_type = 'Transformer'
        self.merge = merge

        self.pos_encoder = PositionalEncoding(feature_size, seq_len, dropout=dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=nlayers)

        # self.generator = F.log_softmax

        if merge:
            self.merge_proj = nn.Sequential(
                nn.Linear(feature_size*seq_len, feature_size*seq_len//2),
                nn.Linear(feature_size*seq_len//2, feature_size*seq_len//4),
                nn.Linear(feature_size*seq_len//4, feature_size*seq_len//8)
            )
            self.proj = nn.Linear(feature_size*seq_len//8, cls_num)
        else:
            self.proj = nn.Linear(feature_size, cls_num)

        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, _src, tgt, src_mask=None, tgt_mask=None):
        src = self.pos_encoder(_src)
        memory = self.encoder(src, src_mask)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)

        if self.merge:
            output = output.permute(1,0,2).reshape(output.size(1),-1)
            output = self.merge_proj(output)

        logits = self.proj(output)
        # pred = self.generator(logits, dim=-1)

        return logits, logits

    def encode(self, src, src_mask):
        # 调用encoder来进行编码，传入的参数embedding的src和src_mask
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        # 调用decoder
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class TransImgPredEn(nn.Module):
    
    def __init__(self, seq_len, feature_size, cls_num,
                 nlayers, nhead, nhid=512,
                 dropout=0.1,
                 merge=False):
        super(TransImgPredEn, self).__init__()
        self.model_type = 'Transformer'
        self.merge = merge

        self.pos_encoder = PositionalEncoding(feature_size, seq_len, dropout=dropout)
        # self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, feature_size))


        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.decoder = nn.Linear(feature_size, cls_num)


        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, _src, src_mask=None, tgt_mask=None):
        src = self.pos_encoder(_src)
        memory = self.transformer_encoder(src, src_mask)
        output = self.decoder(memory)

        return output

    def encode(self, src, src_mask):
        # 调用encoder来进行编码，传入的参数embedding的src和src_mask
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        # 调用decoder
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class TPT(nn.Module):
    def __init__(self,
                *, 
                seq_len, feature_size, 
                featMap_size,
                num_classes, 
                depth, heads, 
                mlp_dim, 
                pool = 'cls', 
                channels = 3, 
                dim_head = 64, 
                dropout = 0., 
                emb_dropout = 0.
                ):
        super(TPT, self).__init__()
        num_patches = seq_len
        dim = feature_size
        patch_size = featMap_size
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b s c p1 p2 -> b s (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        #     nn.Linear(patch_dim, dim),
        # )

        self.to_patch_embedding_img = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding_img(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x_pool = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x_pool)

        x = self.mlp_head(x)

        x = torch.nn.Sigmoid()(x)

        return x


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()



    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden

class ClassNet(nn.Module):
    def __init__(self, input_size=256, cls_num=2):
        super(ClassNet, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        self.fc3 = nn.Linear(input_size // 4, input_size // 8)
        self.fc4 = nn.Linear(input_size // 8, cls_num)


    def forward(self, x_):

        x1 = x_[:,:5]
        x2 = x_[:,5:]
        x = x2 - x1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x






## =================================== ResNet feature network ====================================== ##


class featureNet(nn.Module):

    def __init__(self, feature_size=256):
        super(featureNet, self).__init__()

        net = models.resnet50(pretrained=True)
        num_fits = net.fc.in_features
        self.feaNet = nn.Sequential(
            *list(net.children())[:-1],
            # nn.Flatten()
            )
        # self.feaDown = nn.Linear(num_fits, feature_size)  
        # self.fc = nn.Linear(feature_size, 3)

    
    def forward(self, x_):
        img_size = x_.size()
        x = torch.reshape(x_, (-1, *(img_size[2:])))

        x_fea = self.feaNet[:-1](x)
        # x_fea = nn.Flatten()(x_fea)

        # x_fea = nn.ReLU()(x_fea)
        # x_fea = self.feaDown(x_fea)
        # x = self.fc(x_fea)

        x_fea = torch.reshape(x_fea, (*img_size[:2], *x_fea.size()[-3:]))
        return x_fea


class downFeaNet(nn.Module):

    def __init__(self, num_fits=2048, feature_size=256):
        super(downFeaNet, self).__init__()

        
        # self.feaDown = nn.Sequential(
        #                     # nn.Linear(num_fits, 512),
        #                     nn.Linear(num_fits, feature_size),
        #                     nn.Dropout(p=0.5)
        # )

        self.feaDownConv = nn.Sequential(
                            nn.Conv2d(num_fits, feature_size, 1, 1),
                            nn.BatchNorm2d(feature_size),
                            nn.ReLU(),
        )

        # self.feaDownConv = nn.Sequential(
        #                     # nn.Conv2d(num_fits, feature_size, 1, 1),
        #                     nn.Conv2d(num_fits, num_fits//2, 1, )
        #                     nn.BatchNorm2d(num_fits//2),
        #                     nn.ReLU(),
        # )        
        
        self.featConv1x1 = nn.Conv2d(feature_size, feature_size, 7, 1)
        # self.featFC = nn.Sequential(
        #                     nn.Flatten(),
        #                     nn.Linear(feature_size*7*7, feature_size)
        # )
        self.tanh = nn.Tanh()


    def forward(self, x_):

        img_size = x_.size()
        x = torch.reshape(x_, (-1, *(img_size[2:])))

        x_fea = self.feaDownConv(x)
        x_fea = self.featConv1x1(x_fea)
        x_fea = self.tanh(x_fea)
        # x_fea = self.feaDown(x_fea)
        # x = self.fc(x_fea)


        x_fea = torch.reshape(x_fea, (*img_size[:2], -1))
        return x_fea



class downFeaNet_TPT(nn.Module):

    def __init__(self, num_fits=2048, feature_size=256):
        super(downFeaNet_TPT, self).__init__()

        
        # self.feaDown = nn.Sequential(
        #                     # nn.Linear(num_fits, 512),
        #                     nn.Linear(num_fits, feature_size),
        #                     nn.Dropout(p=0.5)
        # )

        self.feaDownConv = nn.Sequential(
                            nn.Conv2d(num_fits, feature_size, 1, 1),
                            nn.BatchNorm2d(feature_size),
                            nn.ReLU(),
        )

        # self.feaDownConv = nn.Sequential(
        #                     # nn.Conv2d(num_fits, feature_size, 1, 1),
        #                     nn.Conv2d(num_fits, num_fits//2, 1, )
        #                     nn.BatchNorm2d(num_fits//2),
        #                     nn.ReLU(),
        # )        
        
        # self.featConv1x1 = nn.Conv2d(feature_size, feature_size, 7, 1)
        # self.featFC = nn.Sequential(
        #                     nn.Flatten(),
        #                     nn.Linear(feature_size*7*7, feature_size)
        # )

    
    def forward(self, x):

        x_fea = self.feaDownConv(x)
        # x_fea = self.featConv1x1(x_fea)
        # x_fea = self.feaDown(x_fea)
        # x = self.fc(x_fea)
        return x_fea



class downFeaNet_mnist(nn.Module):
    
    def __init__(self, num_fits=2048, feature_size=256):
        super(downFeaNet_mnist, self).__init__()

        
        self.feaDown = nn.Linear(num_fits, feature_size)  
        self.class_num = nn.Linear(feature_size, 10)
        # self.fc = nn.Linear(feature_size, 3)

    
    def forward(self, x):

        x_fea = self.feaDown(x)
        x_fea = nn.Tanh()(x_fea)
        x_out = self.class_num(x_fea)
        
        # x_fea = self.feaDown(x_fea)
        # x = self.fc(x_fea)
        return x_fea, x_out

class simpleFeatNet(nn.Module):
    def __init__(self, feature_size=256):
        super(simpleFeatNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*128, 1024)
        self.fc2 = nn.Linear(1024, 256)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = nn.Tanh()(x)
        return x


## ==================================== Auto-encoder pretrained ======================= ##

import functools

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

def load_networks(net, epoch):
    """Load all the networks from the disk.

    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    name = 'G'
    if isinstance(name, str):
        # load_filename = '%s_net_%s.pth' % (epoch, name)
        # load_path = os.path.join(self.save_dir, load_filename)
        load_path = epoch

        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            __patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)

    return net





def define_AE(input_nc, output_nc, ngf, netG, norm='batch', load_suffix=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'simple':
        net = AEfeatNet(input_nc, output_nc, ngf, norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    if load_suffix:
        return load_networks(net, load_suffix)
    else:
        return net




class AEfeatNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.BatchNorm2d):
        super(AEfeatNet, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.model = [
            # 224
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            # 112
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.LeakyReLU(0.2, True),
            # 56
            nn.Conv2d(ngf * 2, ngf*4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.LeakyReLU(0.2, True),
            # 28
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # 14
            nn.Conv2d(ngf*8, min(512, ngf*16), kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(min(512, ngf*16)),
            nn.LeakyReLU(0.2, True),
            # 7
        ]

        self.encoder = nn.Sequential(*self.model)

        self.net = [
            # 7
            nn.ConvTranspose2d(min(512, ngf*16), ngf*8, kernel_size=4, stride=2, padding=1),
            norm_layer(ngf*8),
            nn.LeakyReLU(0.2, True),
            # 14
            nn.ConvTranspose2d(ngf*8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.LeakyReLU(0.2, True),
            # 28
            nn.ConvTranspose2d(ngf * 4, ngf*2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.LeakyReLU(0.2, True),
            # 56
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.LeakyReLU(0.2, True),
            # 112
            nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # 224
        ]

        self.decoder = nn.Sequential(*self.net)


    def forward(self, x_):

        img_size = x_.size()
        x = torch.reshape(x_, (-1, *(img_size[2:])))

        x_fea = self.encoder(x)

        x = self.decoder(x_fea)

        output = {'rec': x, 'fea': torch.reshape(x_fea, (*img_size[:2], *x_fea.size()[-3:]))}

        return output













