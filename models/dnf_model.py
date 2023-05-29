from torch import nn

from .dnf_utils.base import DNFBase
from .dnf_utils.cid import CID
from .dnf_utils.mcc import MCC
from .dnf_utils.fuse import PDConvFuse, GFM
from .dnf_utils.resudual_switch import ResidualSwitchBlock
from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class SingleStageNet(DNFBase):
    def __init__(self, f_number, *, 
                 block_size=1, 
                 layers=4, 
                 denoising_block='CID', 
                 color_correction_block='MCC') -> None:
        super().__init__(f_number, block_size=block_size, layers=layers, 
                         denoising_block=denoising_block, color_correction_block=color_correction_block)
    
    def _pass_features_to_color_decoder(self, x, f_short_cut, encoder_features):
        return x, None, encoder_features


@MODEL_REGISTRY.register()
class MultiStageNet(DNFBase):
    def __init__(self, f_number, *, 
                 block_size=1, 
                 layers=4, 
                 denoising_block='CID', 
                 color_correction_block='MCC') -> None:
        super().__init__(f_number, block_size=block_size, layers=layers, 
                         denoising_block=denoising_block, color_correction_block=color_correction_block)

        aux_outchannel = 3 if block_size == 1 else block_size * block_size
        self.aux_denoising_blocks = nn.ModuleList([
            ResidualSwitchBlock(
                self.denoising_block_class(
                    f_number=f_number * (2**idx),
                    padding_mode=self.padding_mode
                )
            )
            for idx in range(layers)
        ])
        self.aux_upsamples = nn.ModuleList([
            self.upsample_class(
                f_number * (2**idx), 
                padding_mode=self.padding_mode
            )
            for idx in range(1, layers)
        ])
        self.denoising_decoder_fuses = nn.ModuleList([
            self.decoder_fuse_class(in_channels=f_number * (2 ** idx)) for idx in range(layers - 1)
        ])

        self.aux_conv_fuse_0 = nn.Conv2d(f_number, f_number, 3, 1, 1, bias=True, padding_mode=self.padding_mode)
        self.aux_conv_fuse_1 = nn.Conv2d(f_number, aux_outchannel, 1, 1, 0, bias=True)
        
        inchannel = 3 if block_size == 1 else block_size * block_size
        self.aux_feature_conv_0 = nn.Conv2d(inchannel, f_number, 5, 1, 2, bias=True, padding_mode=self.padding_mode)
        self.aux_feature_conv_1 = nn.Conv2d(f_number, f_number, 5, 1, 2, bias=True, padding_mode=self.padding_mode)
        
        head = [2 ** layer for layer in range(layers)]
        self.aux_color_correction_blocks = nn.ModuleList([
            self.color_correction_block_class(
                f_number=f_number * (2 ** idx),
                num_heads=head[idx],
                padding_mode=self.padding_mode,
            )
            for idx in range(layers)
        ])
        self.aux_downsamples = nn.ModuleList([
            self.downsample_class(
                f_number * (2**idx), 
                padding_mode=self.padding_mode
            )
            for idx in range(layers - 1)
        ])
        
    def _pass_features_to_color_decoder(self, x, f_short_cut, encoder_features):
        denoise_decoder_features = []
        for denoise, up, fuse, encoder_feature in reversed(list(zip(
            self.aux_denoising_blocks[1:], 
            self.aux_upsamples, 
            self.denoising_decoder_fuses,
            encoder_features    
        ))):
            x = denoise(x, 1)
            denoise_decoder_features.append(x)
            x = up(x)
            x = fuse(x, encoder_feature)
        x = self.aux_denoising_blocks[0](x, 1)
        denoise_decoder_features.append(x)
        x = x + f_short_cut
        x = self.act(self.aux_conv_fuse_0(x))
        x = self.aux_conv_fuse_1(x)
        res1 = x

        encoder_features = []
        x = self.act(self.aux_feature_conv_0(res1))
        x = self.aux_feature_conv_1(x)
        for color_correction, down in zip(self.aux_color_correction_blocks[:-1], self.aux_downsamples):
            x = color_correction(x)
            encoder_features.append(x)
            x = down(x)
        x = self.aux_color_correction_blocks[-1](x)
        return x, res1, encoder_features


@MODEL_REGISTRY.register()
class DNF(DNFBase):
    def __init__(self, f_number, *,
                block_size=1,
                layers=4,
                denoising_block='CID',
                color_correction_block='MCC',
                feedback_fuse='GFM'
                ) -> None:
        super(DNF, self).__init__(f_number=f_number, block_size=block_size, layers=layers,
                                  denoising_block=denoising_block, color_correction_block=color_correction_block)
        def get_class(class_or_class_str):
            return eval(class_or_class_str) if isinstance(class_or_class_str, str) else class_or_class_str
        
        self.feedback_fuse_class = get_class(feedback_fuse)

        self.feedback_fuses = nn.ModuleList([
            self.feedback_fuse_class(in_channels=f_number * (2 ** idx)) for idx in range(layers)
        ])

        aux_outchannel = 3 if block_size == 1 else block_size * block_size
        self.aux_denoising_blocks = nn.ModuleList([
            ResidualSwitchBlock(
                self.denoising_block_class(
                    f_number=f_number * (2**idx),
                    padding_mode=self.padding_mode
                )   
            )
            for idx in range(layers)
        ])
        self.aux_upsamples = nn.ModuleList([
            self.upsample_class(
                f_number * (2**idx), 
                padding_mode=self.padding_mode
            )
            for idx in range(1, layers)
        ])
        self.denoising_decoder_fuses = nn.ModuleList([
            self.decoder_fuse_class(in_channels=f_number * (2 ** idx)) for idx in range(layers - 1)
        ])

        self.aux_conv_fuse_0 = nn.Conv2d(f_number, f_number, 3, 1, 1, bias=True, padding_mode=self.padding_mode)
        self.aux_conv_fuse_1 = nn.Conv2d(f_number, aux_outchannel, 1, 1, 0, bias=True)

    def _pass_features_to_color_decoder(self, x, f_short_cut, encoder_features):
        ## denoising decoder
        denoise_decoder_features = []
        for denoise, up, fuse, encoder_feature in reversed(list(zip(
            self.aux_denoising_blocks[1:], 
            self.aux_upsamples, 
            self.denoising_decoder_fuses,
            encoder_features    
        ))):
            x = denoise(x, 1)
            denoise_decoder_features.append(x)
            x = up(x)
            x = fuse(x, encoder_feature)
        x = self.aux_denoising_blocks[0](x, 1)
        denoise_decoder_features.append(x)
        x = x + f_short_cut
        x = self.act(self.aux_conv_fuse_0(x))
        x = self.aux_conv_fuse_1(x)
        res1 = x

        ## feedback, local residual switch on
        encoder_features = []
        denoise_decoder_features.reverse()
        x = f_short_cut
        for fuse, denoise, down, decoder_feedback_feature in zip(
            self.feedback_fuses[:-1], 
            self.denoising_blocks[:-1], 
            self.downsamples,
            denoise_decoder_features[:-1]
        ):
            x = fuse(x, decoder_feedback_feature)
            x = denoise(x, 1)  # residual switch on
            encoder_features.append(x)
            x = down(x)
        x = self.feedback_fuses[-1](x, denoise_decoder_features[-1])
        x = self.denoising_blocks[-1](x, 1)  # residual switch on
        
        return x, res1, encoder_features
