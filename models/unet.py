import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    '''
    inconv only changes the number of channels
    '''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv(in_ch, out_ch),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        self.bilinear=bilinear
        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch//2, 1),)
        else:
            self.up =  nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
        


class SelfCompleteNet1raw1ofAnyPredict(nn.Module):  # 1raw1of
    '''
    rawRange: Int, the idx of raw inputs to be predicted
    '''

    def __init__(self, features_root=64, tot_raw_num=5, tot_of_num=1, border_mode='predict', rawRange=None,
                 useFlow=True, padding=True, useCluster=True, clip_predict=1):
        super(SelfCompleteNet1raw1ofAnyPredict, self).__init__()

        self.clip_predict = clip_predict
        assert tot_of_num <= tot_raw_num
        if border_mode == 'predict':
            self.raw_center_idx = tot_raw_num - self.clip_predict
            self.of_center_idx = tot_of_num - self.clip_predict
        else:
            self.raw_center_idx = (tot_raw_num - self.clip_predict) // 2
            self.of_center_idx = (tot_of_num - self.clip_predict) // 2
        if rawRange is None:
            self.rawRange = range(tot_raw_num)
        else:
            if rawRange < 0:
                rawRange += tot_raw_num
            assert rawRange < tot_raw_num
            self.rawRange = range(rawRange, rawRange + self.clip_predict)
        self.raw_channel_num = 3  # RGB channel no.
        self.of_channel_num = 2  # optical flow channel no.
        self.tot_of_num = tot_of_num
        self.tot_raw_num = tot_raw_num
        self.raw_of_offset = self.raw_center_idx - self.of_center_idx

        self.useFlow = useFlow
        self.padding = padding
        self.useCluster = useCluster
        assert self.raw_of_offset >= 0

        if self.padding:
            in_channels = self.raw_channel_num * tot_raw_num
        else:
            in_channels = self.raw_channel_num * (tot_raw_num - self.clip_predict)

        raw_out_channels = self.raw_channel_num * self.clip_predict
        of_out_channels = self.of_channel_num * self.clip_predict

        self.inc = inconv(in_channels, features_root)
        self.down1 = down(features_root, features_root * 2)
        self.down2 = down(features_root * 2, features_root * 4)
        self.down3 = down(features_root * 4, features_root * 8)
        # 0
        self.up1 = up(features_root * 8, features_root * 4)
        self.up2 = up(features_root * 4, features_root * 2)
        self.up3 = up(features_root * 2, features_root)
        self.outc = outconv(features_root, raw_out_channels)

        if useFlow:
            self.inc_of = inconv(in_channels, features_root)
            self.down_of1 = down(features_root, features_root * 2)
            self.down_of2 = down(features_root * 2, features_root * 4)
            self.down_of3 = down(features_root * 4, features_root * 8)

            self.up_of1 = up(features_root * 8, features_root * 4)
            self.up_of2 = up(features_root * 4, features_root * 2)
            self.up_of3 = up(features_root * 2, features_root)
            self.outc_of = outconv(features_root, of_out_channels)

    def forward(self, x, x_of):
        # use incomplete inputs to yield complete inputs
        of_i = self.tot_raw_num - self.clip_predict - self.raw_of_offset
        if self.padding:
            incomplete_x = x.clone()
            incomplete_x[:, (self.tot_raw_num - self.clip_predict) * self.raw_channel_num:, :, :] = 0
            incomplete_of = x_of.clone()
            incomplete_of[:, :of_i * self.of_channel_num, :, :] = 0
        else:
            incomplete_x = x[:, :(self.tot_raw_num - self.clip_predict) * self.raw_channel_num, :, :]
            incomplete_of = x_of[:, :of_i * self.of_channel_num, :, :]
        raw_target = x[:, (self.tot_raw_num - self.clip_predict) * self.raw_channel_num:, :, :]

        x1 = self.inc(incomplete_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        raw = self.up1(x4, x3)
        raw = self.up2(raw, x2)
        raw = self.up3(raw, x1)
        raw_output = self.outc(raw)

        # of_i = self.tot_raw_num - self.clip_predict - self.raw_of_offset
        if self.useFlow:
            ofx1 = self.inc_of(incomplete_x)
            ofx2 = self.down_of1(ofx1)
            ofx3 = self.down_of2(ofx2)
            ofx4 = self.down_of3(ofx3)

            of = self.up_of1(ofx4, ofx3)
            of = self.up_of2(of, ofx2)
            of = self.up_of3(of, ofx1)
            of_output = self.outc_of(of)

            of_target = x_of[:, of_i * self.of_channel_num:(of_i + self.clip_predict) * self.of_channel_num, :, :]

        if self.useCluster:
            return of_output, raw_output, of_target, raw_target, x4, ofx4
        else:
            return of_output, raw_output, of_target, raw_target


class SelfCompleteNet1raw1ofAnyPredictStage1(nn.Module):  # 1raw1of
    '''
    rawRange: Int, the idx of raw inputs to be predicted
    '''

    def __init__(self, features_root=64, tot_raw_num=5, tot_of_num=1, border_mode='predict', rawRange=None,
                 useFlow=True, padding=True, useCluster=True, negativePool=None, clip_predict=1):
        super(SelfCompleteNet1raw1ofAnyPredictStage1, self).__init__()
        self.clip_predict = clip_predict
        assert tot_of_num <= tot_raw_num
        if border_mode == 'predict':
            self.raw_center_idx = tot_raw_num - self.clip_predict
            self.of_center_idx = tot_of_num - self.clip_predict
        else:
            self.raw_center_idx = (tot_raw_num - self.clip_predict) // 2
            self.of_center_idx = (tot_of_num - self.clip_predict) // 2
        if rawRange is None:
            self.rawRange = range(tot_raw_num)
        else:
            if rawRange < 0:
                rawRange += tot_raw_num
            assert rawRange < tot_raw_num
            self.rawRange = range(rawRange, rawRange + self.clip_predict)
        self.raw_channel_num = 3  # RGB channel no.
        self.of_channel_num = 2  # optical flow channel no.
        self.tot_of_num = tot_of_num
        self.tot_raw_num = tot_raw_num
        self.raw_of_offset = self.raw_center_idx - self.of_center_idx

        self.useFlow = useFlow
        self.padding = padding
        self.useCluster = useCluster
        self.negativePool = negativePool        # todo: negative pool, noise or unrelated image
        assert self.raw_of_offset >= 0

        if self.padding:
            in_channels = self.raw_channel_num * tot_raw_num
        else:
            in_channels = self.raw_channel_num * (tot_raw_num - self.clip_predict)

        raw_out_channels = self.raw_channel_num * self.clip_predict
        of_out_channels = self.of_channel_num * self.clip_predict

        self.inc = inconv(in_channels, features_root)
        self.down1 = down(features_root, features_root * 2)
        self.down2 = down(features_root * 2, features_root * 4)
        self.down3 = down(features_root * 4, features_root * 8)
        # 0
        self.up1 = up(features_root * 8, features_root * 4)
        self.up2 = up(features_root * 4, features_root * 2)
        self.up3 = up(features_root * 2, features_root)
        self.outc = outconv(features_root, raw_out_channels)

        if useFlow:
            self.inc_of = inconv(in_channels, features_root)
            self.down_of1 = down(features_root, features_root * 2)
            self.down_of2 = down(features_root * 2, features_root * 4)
            self.down_of3 = down(features_root * 4, features_root * 8)

            self.up_of1 = up(features_root * 8, features_root * 4)
            self.up_of2 = up(features_root * 4, features_root * 2)
            self.up_of3 = up(features_root * 2, features_root)
            self.outc_of = outconv(features_root, of_out_channels)

    def forward(self, x, x_of):
        of_i = self.tot_raw_num - self.clip_predict - self.raw_of_offset
        # use incomplete inputs to yield complete inputs
        if self.padding:
            incomplete_x = x.clone()
            incomplete_x[:, (self.tot_raw_num - self.clip_predict) * self.raw_channel_num:, :, :] = 0
            incomplete_of = x_of.clone()
            incomplete_of[:, :of_i * self.of_channel_num, :, :] = 0
        else:
            incomplete_x = x[:, :(self.tot_raw_num - self.clip_predict) * self.raw_channel_num, :, :]
            incomplete_of = x_of[:, :of_i * self.of_channel_num, :, :]
            negative_x = incomplete_x.clone()
            # negative pool, either use noise or use shuffled frames
            if random.randint(0,1) == 0:
                negative_x[:, (self.tot_raw_num - self.clip_predict-1) * self.raw_channel_num: (self.tot_raw_num - self.clip_predict) * self.raw_channel_num, :, :]\
                    = torch.randn(negative_x[:, (self.tot_raw_num - self.clip_predict-1) * self.raw_channel_num: (self.tot_raw_num - self.clip_predict) * self.raw_channel_num, :, :].shape) # todo
            else:
                shuffle_indices = (list(range(self.tot_raw_num - self.clip_predict)))
                while True:
                    random.shuffle(shuffle_indices)
                    if shuffle_indices != (list(range(self.tot_raw_num - self.clip_predict))):
                        break

                for i,idx in enumerate(shuffle_indices):
                    negative_x[:,i*self.raw_channel_num:(i+1)*self.raw_channel_num]=incomplete_x[:,idx*self.raw_channel_num:(idx+1)*self.raw_channel_num]
        raw_target = x[:, (self.tot_raw_num - self.clip_predict) * self.raw_channel_num:, :, :]

        x1 = self.inc(incomplete_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        negative_x4 = self.inc(negative_x)
        negative_x4 = self.down1(negative_x4)
        negative_x4 = self.down2(negative_x4)
        negative_x4 = self.down3(negative_x4)

        raw = self.up1(x4, x3)
        raw = self.up2(raw, x2)
        raw = self.up3(raw, x1)
        raw_output = self.outc(raw)

        # of_i = self.tot_raw_num - 1 - self.raw_of_offset
        if self.useFlow:
            ofx1 = self.inc_of(incomplete_x)
            ofx2 = self.down_of1(ofx1)
            ofx3 = self.down_of2(ofx2)
            ofx4 = self.down_of3(ofx3)

            negative_ofx4 = self.inc(negative_x)
            negative_ofx4 = self.down1(negative_ofx4)
            negative_ofx4 = self.down2(negative_ofx4)
            negative_ofx4 = self.down3(negative_ofx4)

            of = self.up_of1(ofx4, ofx3)
            of = self.up_of2(of, ofx2)
            of = self.up_of3(of, ofx1)
            of_output = self.outc_of(of)

            of_target = x_of[:, of_i * self.of_channel_num:(of_i + self.clip_predict) * self.of_channel_num, :, :]

        if self.useCluster:
            return of_output, raw_output, of_target, raw_target, x4, ofx4, negative_x4, negative_ofx4
        else:
            return of_output, raw_output, of_target, raw_target



class MLP_Projection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_Projection, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_dim,1024), nn.BatchNorm1d(1024),nn.ReLU(),
                                     nn.Linear(1024, out_dim))
    def forward(self, x):

        return F.normalize(self.project(x), dim = -1)


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.rawProjection = MLP_Projection(4096,512)
        self.ofProjection = MLP_Projection(4096,512)

        self.l0 = nn.Linear(1024, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, raw_emb, of_emb):
        raw = self.rawProjection(raw_emb)
        of = self.ofProjection(of_emb)
        h = torch.cat((raw,of),dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


# used for predict mse loss, i.e. reconstruction difficulty
class MLP_Predictor(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(MLP_Predictor, self).__init__()
        self.predict = nn.Sequential(nn.Linear(in_dim,1024), nn.BatchNorm1d(1024),nn.ReLU(),
                                     nn.Linear(1024, 512), nn.BatchNorm1d(512),nn.ReLU(),
                                     nn.Linear(512, 128),nn.ReLU(),nn.Linear(128, out_dim))
    def forward(self, x):
        return self.predict(x)