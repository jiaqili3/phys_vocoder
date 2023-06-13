from easydict import EasyDict as edict

config = edict()
config.model = edict()

# ECAPATDNN
config.ECAPATDNN = edict()
config.ECAPATDNN.C = 1024

# ECAPATDNNGLOBc1024
config.ECAPATDNNGLOBc1024 = edict()
config.ECAPATDNNGLOBc1024.feat_dim = 80
config.ECAPATDNNGLOBc1024.embed_dim = 192
config.ECAPATDNNGLOBc1024.num_class = 1251 + 5994
config.ECAPATDNNGLOBc1024.pooling_func = 'TSTP'

# ResNet34
config.ResNet34 = edict()
config.ResNet34.feat_dim = 80
config.ResNet34.embed_dim = 256
config.ResNet34.num_class = 1251
config.ResNet34.pooling_func = 'TSTP'
config.ResNet34.two_emb_layer = False

# XVEC
config.XVEC = edict()
config.XVEC.feat_dim = 80
config.XVEC.embed_dim = 192
config.XVEC.num_class = 1251
config.XVEC.pooling_func = 'TSTP'

# Xvector
config.Xvector = edict()
config.Xvector.tdnn_blocks = 5
config.Xvector.tdnn_channels = [512, 512, 512, 512, 1500]
config.Xvector.tdnn_kernel_sizes = [5, 3, 3, 1, 1]
config.Xvector.tdnn_dilations = [1, 2, 3, 1, 1]
config.Xvector.lin_neurons = 512
config.Xvector.in_channels = 24

# # RawNet3_
# config.RawNet3 = edict()
# config.RawNet3.block = 'Bottle2neck'
# config.RawNet3.model_scale = 8
# config.RawNet3.context = True
# config.RawNet3.summed = True
# config.RawNet3.log_sinc = True
# config.RawNet3.embed_dim = 512
# config.RawNet3.num_class = 1251
# config.RawNet3.norm_sinc = 'mean'
# config.RawNet3.grad_mult = 1
# config.RawNet3.encoder_type = 'ECA'
# config.RawNet3.sinc_stride = 10

# RawNet3
config.RawNet3 = edict()
config.RawNet3.block = 'Bottle2neck'
config.RawNet3.model_scale = 8
config.RawNet3.context = True
config.RawNet3.summed = True
config.RawNet3.out_bn = False
config.RawNet3.log_sinc = True
config.RawNet3.norm_sinc = "mean"
config.RawNet3.grad_mult = 1
config.RawNet3.encoder_type = 'ECA'
config.RawNet3.sinc_stride = 10
config.RawNet3.nOut = 256

# ResNetSE34V2
config.ResNetSE34V2 = edict()
config.ResNetSE34V2.nOut = 512
config.ResNetSE34V2.encoder_type = 'ASP'
config.ResNetSE34V2.n_mels = 64
config.ResNetSE34V2.log_input = True
config.ResNetSE34V2.block = "SEBasicBlock"
config.ResNetSE34V2.layers = [3, 4, 6, 3]
config.ResNetSE34V2.num_filters = [32, 64, 128, 256]

