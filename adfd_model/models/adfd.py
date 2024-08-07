from adfd_model.util import train_utils
from torch.optim import Adam
from adfd_model.criteria.earlystopping import EarlyStopping
from adfd_model.criteria.aging_loss import AgingLoss
from adfd_model.criteria.lpips.lpips import LPIPS
from adfd_model.criteria import id_loss
from adfd_model.models.psp import pSp
from adfd_model.util.common import tensor2im
from adfd_model.config import paths_config
from adfd_model.config.config import *
import sys
import numpy as np
from argparse import Namespace
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(".")
sys.path.append("..")


class ADFD:
    def __init__(self, opts):
        self.test_opts = opts

        # update test options with options used during training
        ckpt = torch.load(self.test_opts.checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts.update(vars(self.test_opts))
        self.opts = Namespace(**opts)

        self.net = pSp(self.opts)
        self.net.eval()
        self.net.cuda()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        # Initialize loss
        self.mse_loss = nn.MSELoss().cuda().eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').cuda().eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().cuda().eval()
        if self.opts.aging_lambda > 0:
            self.aging_loss = AgingLoss()

        self.transforms_inference = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def infer(self, np_arr):
        results = None
        img = self.transforms_inference(Image.fromarray(np_arr).convert("RGB"))
        for age_transformer in AGE_TRANSFORMERS:
            print(f"Running on target age: {age_transformer.target_age}")
            input_age_batch = [age_transformer(img.cpu()).to('cuda')]
            input_age_batch = torch.stack(input_age_batch)
            for input_image in input_age_batch:
                result, sv = self.net(input_image.unsqueeze(0).to("cuda").float(
                ), randomize_noise=False, resize=self.opts.resize_outputs, return_latents=False, return_s=True)
                resize_amount = (
                    IMAGE_SIZE, IMAGE_SIZE) if self.opts.resize_outputs else (1024, 1024)
                initial_age = self.aging_loss.extract_ages(result)

                input_image = input_image[:3].unsqueeze(0).to("cuda").float()
                result = self.face_pool(result)

                s_mod = self.age_id_based_perturbation(sv)
                if self.opts.div_opt == 'adam':
                    optimizer = Adam([s for i, s in enumerate(s_mod) if i %
                                      3 != 1], self.opts.div_lr)
                earlystopping = EarlyStopping(
                    patience=self.opts.patience, delta=self.opts.es_delta)
                for step in tqdm(range(self.opts.max_steps_adfd)):
                    optimizer.zero_grad()
                    if step == 0:
                        initial_result = result.detach().clone()
                        target_ages = initial_age.detach().clone() / 100.
                        input_ages = target_ages
                    else:
                        input_ages = self.aging_loss.extract_ages(
                            y_hat) / 100.

                    y_hat, _ = self.net.decoder([s_mod],
                                                input_is_latent=False,
                                                input_is_stylespace=True,
                                                randomize_noise=False,)
                    y_hat_rs = self.face_pool(y_hat)
                    loss, _, _ = self.calc_loss(input_image,
                                                initial_result,
                                                y_hat_rs,
                                                target_ages=target_ages,
                                                input_ages=input_ages)

                    earlystopping(loss.item())
                    if earlystopping.counter == 0:
                        y_hat_final = y_hat
                    if earlystopping.early_stop:
                        break
                    else:
                        loss.backward()
                        optimizer.step()
                res = tensor2im(y_hat_final[0]).resize(resize_amount)
                # Initialize results with the first result_image shape
                if results is None:
                    results = np.array(res)
                else:
                    # Concatenate the results along the height
                    results = np.concatenate((results, np.array(res)), axis=0)
        return results

    def calc_loss(self, x, y, y_hat, target_ages, input_ages):
        loss_dict = {}
        id_logs = []
        loss = 0.0
        if self.opts.id_lambda > 0:
            weights = None
            if self.opts.use_weighted_id_loss:  # compute weighted id loss only on forward pass
                age_diffs = torch.abs(target_ages - input_ages)
                weights = train_utils.compute_cosine_weights(x=age_diffs)
            loss_id, sim_improvement, id_logs = self.id_loss(
                y_hat, y, x, weights=weights)
            loss_dict[f'loss_id'] = float(loss_id)
            loss_dict[f'id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict[f'loss_l2'] = float(loss_l2)
            l2_lambda = self.opts.l2_lambda
            loss += loss_l2 * l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict[f'loss_lpips'] = float(loss_lpips)
            lpips_lambda = self.opts.lpips_lambda
            loss += loss_lpips * lpips_lambda
        if self.opts.aging_lambda > 0:
            aging_loss, id_logs = self.aging_loss(
                y_hat, y, target_ages, id_logs)
            loss_dict[f'loss_aging'] = float(aging_loss)
            loss += aging_loss * self.opts.aging_lambda
        loss_dict[f'loss'] = float(loss)
        return loss, loss_dict, id_logs

    def age_id_based_perturbation(self, style_vector):
        ca = pd.read_pickle(
            paths_config.analyzation_path['correlation_analysis'])
        ca_idx = list(ca.index)

        ca_idx_d = {}
        for ch in ca_idx:
            pos_tmp = ch
            if ch < 512*15:
                layer = ch // 512
                channel = ch % 512
            elif ch < 512*15 + 256*3:
                pos_tmp -= 512*15
                layer = 15 + pos_tmp // 256
                channel = pos_tmp % 256
            elif ch < 512*15 + 256*3 + 128*3:
                pos_tmp -= 512*15 + 256*3
                layer = 18 + pos_tmp // 128
                channel = pos_tmp % 128
            elif ch < 512*15 + 256*3 + 128*3 + 64*3:
                pos_tmp -= 512*15 + 256*3 + 128*3
                layer = 21 + pos_tmp // 64
                channel = pos_tmp % 64
            else:
                pos_tmp -= 512*15 + 256*3 + 128*3 + 64*3
                layer = 24 + pos_tmp // 32
                channel = pos_tmp % 32
            ca_idx_d[ch] = [layer, channel]

        z_rnd = np.random.randn(1, 512)
        z_rnd = torch.from_numpy(z_rnd.astype(np.float32)).clone().cuda()
        _, o_rnd = self.net.decoder([z_rnd],
                                    input_is_latent=False,
                                    randomize_noise=False,
                                    return_s=True)

        s_copy = [l.detach().clone() for l in style_vector]
        o_mask = [torch.zeros_like(l) for l in style_vector]

        sigma = ca['COEF_AGE'] + (1 - ca['COEF_ID'])
        sigma_max = sigma.max()
        sigma_min = sigma.min()
        o_mask_trgb = (sigma - sigma_min) / (sigma_max - sigma_min)

        for k, v in ca_idx_d.items():
            o_mask[v[0]][0][0][v[1]][0][0] = o_mask_trgb.at[k]

        o_rnd_d = []
        for l1, l2 in zip(o_mask, o_rnd):
            o_rnd_d.append(l1 * l2)

        s_init = []
        for l1, l2 in zip(s_copy, o_rnd_d):
            s_init.append(torch.nn.Parameter((l1 + l2).data))

        return s_init
