import os
from collections import OrderedDict
from os import path as osp
import contextlib
from basicsr.utils.profiler import profile, profiler

import numpy as np
import torch
import torch.distributed as torch_dist
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
import math
import basicofr.metrics  # noqa: F401 - register LPIPS, DISTS, BRISQUE, CLIP-IQA metrics
from basicsr.models.video_base_model import VideoBaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from basicofr.models.chunk_batch import plan_chunk_batches
from basicofr.utils.ckpt_fingerprint import (
    checkpoint_fingerprint,
    fingerprint_matches,
    write_fingerprint,
)


def _resolve_save_best_settings(train_opt, val_opt):
    val_metrics = (val_opt or {}).get('metrics') or {}
    default_metric = 'psnr' if 'psnr' in val_metrics else next(iter(val_metrics), None)
    save_best_enabled = train_opt.get('save_best', bool(val_metrics))
    save_best_metric = train_opt.get('save_best_metric', default_metric)

    if save_best_enabled:
        if not val_metrics:
            raise ValueError('save_best requires val.metrics.')
        if save_best_metric not in val_metrics:
            raise ValueError(f'save_best_metric {save_best_metric!r} is not in val.metrics.')

    return save_best_enabled, save_best_metric


@MODEL_REGISTRY.register()
class RTNGANModel(VideoBaseModel):
    """RTN video restoration GAN model with flow-aware optimization settings."""

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type.lower() == 'adamw':
            return torch.optim.AdamW(params, lr, **kwargs)
        return super().get_optimizer(optim_type, params, lr, **kwargs)

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.net_g.train()

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        # discriminator network
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        self.net_d.train()
        self._discriminator_expects_3d = any(isinstance(module, torch.nn.Conv3d) for module in self.net_d.modules())

        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # loss functions
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get('perceptual_opt') else None
        if not train_opt.get('gan_opt'):
            raise ValueError('GAN loss configuration `gan_opt` must be provided for RTNGANModel.')
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.g_gan_weight = train_opt.get('g_gan_weight', 1.0)
        self.d_gan_weight = train_opt.get('d_gan_weight', 1.0)
        self.g_grad_clip = train_opt.get('g_grad_clip', 0.0)
        self.d_grad_clip = train_opt.get('d_grad_clip', 0.0)
        self.d_output_clamp = train_opt.get('d_output_clamp', 0.0)
        self.net_d_start_iter = int(max(train_opt.get('net_d_start_iter', 0), self.net_d_init_iters))
        self.flow_lr_mul = train_opt.get('flow_lr_mul', 1.0)
        self.flow_modules = train_opt.get('flow_modules', ['spynet'])
        # fix_flow_iters 可能从 YAML 读取为字符串（如 '5e-4'），需要转换为 float
        self.fix_flow_iters = float(train_opt.get('fix_flow_iters', 0))
        self._flow_frozen = False

        # 梯度累积与混合精度配置（保持向后兼容）
        # 累积步数：accumulation_steps=1 时等价于原始行为
        self.accumulation_steps = train_opt.get('accumulation_steps', 1)
        self.accumulation_counter = 0

        # AMP 配置：use_amp=False 时不启用混合精度
        self.use_amp = train_opt.get('use_amp', False)
        if self.use_amp:
            # 延迟导入，避免在 CPU 或未安装 AMP 环境下报错
            from torch.amp import GradScaler
            self.scaler_g = GradScaler('cuda')
            self.scaler_d = GradScaler('cuda')
        else:
            self.scaler_g = None
            self.scaler_d = None

        # setup optimizers / schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        # Save best by default when validation metrics exist; explicit false is the opt-out.
        self.save_best_enabled, self.save_best_metric = _resolve_save_best_settings(train_opt, self.opt.get('val', {}))
        if self.save_best_enabled:
            logger = get_root_logger()
            logger.info(f'[SaveBest] Enabled: metric={self.save_best_metric}')

    def setup_optimizers(self):
        train_opt = self.opt['train']

        optim_g_opt = train_opt['optim_g'].copy()
        optim_type_g = optim_g_opt.pop('type')
        lr_g = optim_g_opt.pop('lr')

        if self.flow_lr_mul != 1.0:
            normal_params, flow_params = [], []
            for name, param in self.net_g.named_parameters():
                if not param.requires_grad:
                    continue
                if any(flow_key in name for flow_key in self.flow_modules):
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            param_groups = []
            if normal_params:
                param_groups.append({'params': normal_params, 'lr': lr_g})
            if flow_params:
                param_groups.append({'params': flow_params, 'lr': lr_g * self.flow_lr_mul})
        else:
            param_groups = [{'params': [p for p in self.net_g.parameters() if p.requires_grad]}]

        self.optimizer_g = self.get_optimizer(optim_type_g, param_groups, lr=lr_g, **optim_g_opt)
        self.optimizers.append(self.optimizer_g)

        optim_d_opt = train_opt['optim_d'].copy()
        optim_type_d = optim_d_opt.pop('type')
        lr_d = optim_d_opt.pop('lr')
        self.optimizer_d = self.get_optimizer(optim_type_d, self.net_d.parameters(), lr=lr_d, **optim_d_opt)
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        # 显式保持 float32，AMP 会在 autocast 中自动转换
        self.lq = data['lq'].to(self.device, non_blocking=True).float().contiguous()
        if 'gt' in data:
            self.gt = data['gt'].to(self.device, non_blocking=True).float().contiguous()

    def prepare_validation_chunk(self, val_data, start_idx, end_idx, frame_names):
        """Optional project hook for chunk-level validation side inputs."""
        return None

    def _compute_dataset_level_metrics(self, dataloader, save_img, current_iter=None, tb_logger=None):
        """计算数据集级别指标（如 FID）。

        FID 需要在所有图像保存后，基于目录计算。
        """
        val_opt = self.opt.get('val', {})
        # metrics 键存在但值为 null（NR 测试）时 .get 返回 None，需兜底为 {}
        metrics_opt = val_opt.get('metrics') or {}

        # 检查是否配置了 FID
        if 'fid' not in metrics_opt:
            return

        # FID 需要保存图像
        if not save_img:
            logger = get_root_logger()
            logger.warning('FID 需要 save_img=True，跳过 FID 计算')
            return

        # 获取路径
        dataset_name = dataloader.dataset.opt['name']
        visualization_path = self.opt['path'].get('visualization', 'results')
        generated_path = osp.join(visualization_path, dataset_name)

        # GT 路径从数据集配置获取
        gt_path = dataloader.dataset.opt.get('dataroot_gt', None)
        if gt_path is None:
            logger = get_root_logger()
            logger.warning('未找到 dataroot_gt，跳过 FID 计算')
            return

        # 计算 FID
        logger = get_root_logger()
        logger.info(f'Computing FID: {generated_path} vs {gt_path}')

        try:
            from basicofr.metrics import calculate_fid
            fid_opt = metrics_opt['fid']
            fid_value = calculate_fid(
                generated_path=generated_path,
                gt_path=gt_path,
                **{k: v for k, v in fid_opt.items() if k != 'type'}
            )
            # 格式与其他指标一致
            logger.info(f'Validation {dataset_name}\n\t # fid: {fid_value:.4f}')

            # 记录到 TensorBoard
            if tb_logger is not None:
                tb_logger.add_scalar(f'metrics/fid', fid_value, current_iter)

        except Exception as e:
            logger.warning(f'FID 计算失败: {e}')

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        if self.opt['dist']:
            rank, _ = get_dist_info()
            if rank == 0:
                self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
            if torch_dist.is_available() and torch_dist.is_initialized():
                torch_dist.barrier()
        else:
            super().validation(dataloader, current_iter, tb_logger, save_img)

        # 计算数据集级别指标 (FID)
        self._compute_dataset_level_metrics(dataloader, save_img, current_iter, tb_logger)

    def optimize_parameters(self, current_iter):
        """
        优化步骤：
        1) 判别器 D：每次迭代都更新，保证 GAN 稳定性；支持 AMP。
        2) 生成器 G：支持梯度累积与 AMP；仅在累计到设定步数时 step。
        """
        self._manage_flow_freeze(current_iter)

        use_gan = current_iter >= self.net_d_start_iter

        amp_autocast = contextlib.nullcontext
        if self.use_amp:
            from functools import partial
            from torch.amp import autocast
            amp_autocast = partial(autocast, 'cuda')

        with amp_autocast():
            with profile('G forward'):
                self.output = self.net_g(self.lq)

        loss_dict = OrderedDict()

        gt_for_d = self.gt
        output_for_d = self.output
        if not getattr(self, '_discriminator_expects_3d', False):
            if self.gt.dim() == 5:
                b, t, c, h, w = self.gt.size()
                gt_for_d = self.gt.reshape(b * t, c, h, w)
            if self.output.dim() == 5:
                b, t, c, h, w = self.output.size()
                output_for_d = self.output.reshape(b * t, c, h, w)

        # ========== 判别器更新（支持延迟启动） ==========
        if use_gan:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()

            with amp_autocast():
                with profile('D forward (real)'):
                    real_pred = self.net_d(gt_for_d)
                with profile('D forward (fake)'):
                    fake_pred = self.net_d(output_for_d.detach())
                if self.d_output_clamp > 0:
                    real_pred = real_pred.clamp(-self.d_output_clamp, self.d_output_clamp)
                    fake_pred = fake_pred.clamp(-self.d_output_clamp, self.d_output_clamp)
                loss_d_real = self.cri_gan(real_pred, True, is_disc=True)
                loss_d_fake = self.cri_gan(fake_pred, False, is_disc=True)
                l_d_total = 0.5 * (loss_d_real + loss_d_fake) * self.d_gan_weight

            loss_dict['l_d_real'] = loss_d_real
            loss_dict['out_d_real'] = torch.mean(real_pred.detach())
            loss_dict['l_d_fake'] = loss_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_pred.detach())
            loss_dict['l_d'] = l_d_total

            if self.use_amp:
                with profile('D backward'):
                    self.scaler_d.scale(l_d_total).backward()
                if self.d_grad_clip > 0:
                    self.scaler_d.unscale_(self.optimizer_d)
                    torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), self.d_grad_clip)
                with profile('D step'):
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
            else:
                with profile('D backward'):
                    l_d_total.backward()
                if self.d_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), self.d_grad_clip)
                with profile('D step'):
                    self.optimizer_d.step()
        else:
            for p in self.net_d.parameters():
                p.requires_grad = False
            zero = torch.zeros(1, device=self.device)
            loss_dict['l_d_real'] = zero
            loss_dict['out_d_real'] = zero
            loss_dict['l_d_fake'] = zero
            loss_dict['out_d_fake'] = zero
            loss_dict['l_d'] = zero

        # ========== 生成器更新（梯度累积 + AMP） ==========
        for p in self.net_d.parameters():
            p.requires_grad = False

        # 仅在一个累积周期开始时清零梯度
        if self.accumulation_counter == 0:
            self.optimizer_g.zero_grad()

        # 汇总生成器损失
        with amp_autocast():
            l_g_total = 0.0

            if self.cri_pix:
                with profile('Pixel loss (cri_pix)'):
                    l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            if self.cri_perceptual:
                with profile('Perceptual loss + VGG'):
                    if self.output.dim() == 5:
                        b, t, c, h, w = self.output.size()
                        pred = self.output.reshape(b * t, c, h, w)
                        gt = self.gt.reshape(b * t, c, h, w)
                        percep_result = self.cri_perceptual(pred, gt)
                    else:
                        percep_result = self.cri_perceptual(self.output, self.gt)

                # 感知损失可能返回 (percep, style, ...)，只取第一个
                l_g_percep = percep_result[0] if isinstance(percep_result, tuple) else percep_result
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep

            if use_gan:
                with profile('GAN loss (G cri_gan)'):
                    fake_g_pred = self.net_d(output_for_d)
                    if self.d_output_clamp > 0:
                        fake_g_pred = fake_g_pred.clamp(-self.d_output_clamp, self.d_output_clamp)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False) * self.g_gan_weight
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan
            else:
                loss_dict['l_g_gan'] = torch.zeros(1, device=self.device)

            offset_loss_fn = getattr(self.net_g, 'get_offset_loss', None)
            if offset_loss_fn is not None:
                l_g_offset = offset_loss_fn()
                l_g_total += l_g_offset
                loss_dict['l_g_offset'] = l_g_offset

            # 梯度累积按步数缩放损失，确保等价于大 batch
            if self.accumulation_steps > 1:
                l_g_total = l_g_total / float(self.accumulation_steps)

        # 反向传播（AMP/FP32）
        if self.use_amp:
            with profile('G backward'):
                self.scaler_g.scale(l_g_total).backward()
        else:
            with profile('G backward'):
                l_g_total.backward()

        # 仅在累计到指定步数时 step + 清零
        if (self.accumulation_counter + 1) % self.accumulation_steps == 0:
            if self.use_amp:
                if self.g_grad_clip > 0:
                    self.scaler_g.unscale_(self.optimizer_g)
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.g_grad_clip)
                with profile('G step'):
                    self.scaler_g.step(self.optimizer_g)
                    self.scaler_g.update()
            else:
                if self.g_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.g_grad_clip)
                with profile('G step'):
                    self.optimizer_g.step()
            self.optimizer_g.zero_grad()

        # 更新累积计数器
        self.accumulation_counter = (self.accumulation_counter + 1) % self.accumulation_steps

        # 记录日志与 EMA
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # 训练可视化保存
        self._save_training_visualization(current_iter)

        # 打印本 iter 的剖析结果
        profiler.report_and_reset(current_iter)

        # AMP step 状态标记，用于 scheduler 同步
        self._optimizer_stepped = True

    def _save_training_visualization(self, current_iter):
        """保存训练可视化图像（GT/LQ/Output 对比图）。

        子类可直接调用此方法以复用可视化逻辑。
        同时支持保存到文件和写入 TensorBoard。
        """
        train_vis_iter = self.opt.get('train', {}).get('train_visualization_iter', 0)
        if train_vis_iter <= 0 or current_iter % train_vis_iter != 0:
            return

        # 仅 rank=0 或非分布式环境保存
        if self.opt.get('dist', False):
            rank, _ = get_dist_info()
            if rank != 0:
                return

        import torchvision.utils as vutils
        from PIL import Image, ImageDraw, ImageFont
        import io

        vis_dir = osp.join(self.opt['path'].get('visualization', 'experiments/visualization'), 'training')
        os.makedirs(vis_dir, exist_ok=True)

        # 取 batch 第一个样本
        gt_vis = self.gt[0] if self.gt.dim() == 5 else self.gt
        lq_vis = self.lq[0] if self.lq.dim() == 5 else self.lq
        out_vis = self.output[0] if self.output.dim() == 5 else self.output

        # 拼接：GT, LQ, Output（3 行）
        comparison = torch.cat([gt_vis, lq_vis, out_vis], dim=0)


        # 归一化处理（如果使用了 normalizing）
        if self.opt.get('datasets', {}).get('train', {}).get('normalizing', False):
            comparison = (comparison + 1.) / 2.

        num_frames = gt_vis.size(0)

        # 先使用 vutils 生成基础图像
        buf = io.BytesIO()
        vutils.save_image(comparison.cpu(), buf, nrow=num_frames, padding=0, normalize=False, format='PNG')
        buf.seek(0)

        # 使用 PIL 添加标签
        img = Image.open(buf)
        row_height = img.height // 3
        label_width = 80  # 标签区域宽度

        # 创建带标签区域的新图像
        new_img = Image.new('RGB', (img.width + label_width, img.height), color=(255, 255, 255))
        new_img.paste(img, (label_width, 0))

        draw = ImageDraw.Draw(new_img)
        labels = ['GT', 'LQ', 'Restored']

        # 尝试加载字体，失败则使用默认
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for i, label in enumerate(labels):
            y_center = i * row_height + row_height // 2
            # 获取文本边界框
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (label_width - text_width) // 2
            y = y_center - text_height // 2
            draw.text((x, y), label, fill=(0, 0, 0), font=font)

        save_path = osp.join(vis_dir, f'training_show_{current_iter}.png')
        new_img.save(save_path)

        # 写入 TensorBoard
        self._write_to_tensorboard(comparison, num_frames, current_iter)

    def _get_tb_logger(self):
        """获取或创建 TensorBoard logger（延迟初始化）。"""
        if hasattr(self, '_tb_logger') and self._tb_logger is not None:
            return self._tb_logger

        # 检查是否启用 TensorBoard
        if not self.opt.get('logger', {}).get('use_tb_logger', False):
            return None
        if 'debug' in self.opt.get('name', ''):
            return None

        # 获取 TensorBoard 日志路径
        tb_log_dir = self.opt.get('path', {}).get('tb_logger')
        if tb_log_dir is None:
            experiments_root = self.opt.get('path', {}).get('experiments_root', 'experiments')
            tb_log_dir = osp.join(experiments_root, 'tb_logger')
        tb_log_dir = osp.join(tb_log_dir, self.opt.get('name', 'default'))

        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_logger = SummaryWriter(log_dir=tb_log_dir)
            return self._tb_logger
        except ImportError:
            return None

    def _write_to_tensorboard(self, comparison, num_frames, current_iter):
        """将训练可视化图像写入 TensorBoard。

        Args:
            comparison: 拼接后的图像张量 (3*T, C, H, W)
            num_frames: 帧数
            current_iter: 当前迭代次数
        """
        import torchvision.utils as vutils

        tb_logger = self._get_tb_logger()
        if tb_logger is None:
            return

        # 创建网格图像
        grid = vutils.make_grid(comparison.cpu(), nrow=num_frames, padding=2, normalize=False)

        # 写入 TensorBoard
        tb_logger.add_image('training/GT_LQ_Restored', grid, current_iter)

    def test(self):
        use_ema = hasattr(self, 'ema_decay') and self.ema_decay > 0
        net = self.net_g_ema if use_ema else self.net_g

        prev_training_state = net.training
        net.eval()

        collect_debug = getattr(self, '_collect_visual_debug', False)
        self._last_visual_masks = None
        self._last_visual_features = None

        with torch.no_grad():
            self.output = net(self.lq)

            if collect_debug:
                masks = None
                features = None

                if hasattr(net, 'visualiza_mask'):
                    try:
                        backward_mask, forward_mask = net.visualiza_mask(self.lq)
                        masks = {
                            'backward': backward_mask.detach().cpu()[0],
                            'forward': forward_mask.detach().cpu()[0],
                        }
                    except Exception:
                        masks = None

                if hasattr(net, 'visualiza_feature'):
                    try:
                        feature_outputs = net.visualiza_feature(self.lq)
                    except Exception:
                        feature_outputs = None
                    if feature_outputs is not None:
                        features = {}
                        if isinstance(feature_outputs, (list, tuple)):
                            if len(feature_outputs) == 4:
                                b_feat, b_state, f_feat, f_state = feature_outputs
                            elif len(feature_outputs) == 2:
                                b_feat = b_state = None
                                f_feat, f_state = feature_outputs
                            else:
                                b_feat = b_state = f_feat = f_state = None
                        else:
                            b_feat = b_state = None
                            f_feat = f_state = feature_outputs
                        if b_feat is not None:
                            features['backward_mean_mask'] = b_feat.detach().cpu()[0]
                        if b_state is not None:
                            features['backward_state'] = b_state.detach().cpu()[0]
                        if f_feat is not None:
                            features['forward_mean_mask'] = f_feat.detach().cpu()[0]
                        if f_state is not None:
                            features['forward_state'] = f_state.detach().cpu()[0]

                self._last_visual_masks = masks
                self._last_visual_features = features

        if not use_ema and prev_training_state:
            net.train()

    def _mask_to_uint8(self, tensor):
        arr = tensor.squeeze().detach().cpu().numpy()
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).round().astype(np.uint8)

    def _feature_to_uint8(self, tensor):
        arr = tensor.squeeze().detach().cpu().numpy()
        if arr.size == 0:
            return arr.astype(np.uint8)
        min_val = float(arr.min())
        max_val = float(arr.max())
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val)
        else:
            arr = np.zeros_like(arr)
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).round().astype(np.uint8)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img=False):
        """分块验证：将长视频序列分成多个小块处理，避免 OOM。

        通过 opt['val']['chunk_size'] 控制每块的帧数：
        - chunk_size > 0: 按指定帧数分块处理
        - chunk_size <= 0: 不分块，一次性处理整个序列（需要足够显存）
        """
        logger = get_root_logger()
        dataset = dataloader.dataset
        dataset_name = dataset.opt.get('name', 'val')

        # 获取验证分块大小，默认不分块（-1）
        chunk_size_cfg = self.opt['val'].get('val_chunk_size', -1)
        try:
            chunk_size_cfg = int(chunk_size_cfg)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'val_chunk_size must be an integer, got {chunk_size_cfg!r}.') from exc
        drop_tail_cfg = self.opt['val'].get('drop_tail_incomplete_chunk', False)
        if isinstance(drop_tail_cfg, str):
            drop_tail_incomplete_chunk = drop_tail_cfg.lower() in {'1', 'true', 'yes', 'y'}
        else:
            drop_tail_incomplete_chunk = bool(drop_tail_cfg)

        metrics_cfg = self.opt['val'].get('metrics')
        with_metrics = bool(metrics_cfg)
        if with_metrics:
            self._initialize_best_metric_results(dataset_name)
            aggregated_metrics = {metric: [] for metric in metrics_cfg}
        else:
            aggregated_metrics = {}

        vis_root = None
        if save_img:
            vis_root = osp.join(self.opt['path']['visualization'], dataset_name)
            os.makedirs(vis_root, exist_ok=True)
        suffix = self.opt['val'].get('suffix')
        # tensors are normalized to [-1, 1] when datasets enable `normalizing`
        vis_min_max = (-1, 1) if dataset.opt.get('normalizing') else (0, 1)

        self._collect_visual_debug = save_img

        # Fix C: in test mode (current_iter is not an int), stamp/verify which
        # checkpoint produced each clip's outputs so resume never reuses stale
        # images left by a different checkpoint (the a1/a2/a3 incident).
        is_test_mode = not isinstance(current_iter, int)
        ckpt_fingerprint = (
            checkpoint_fingerprint(self.opt.get('path', {}).get('pretrain_network_g'))
            if (save_img and is_test_mode) else None
        )

        for val_data in dataloader:
            # 获取完整序列
            full_lq = val_data['lq']  # (B, T, C, H, W)
            full_gt = val_data.get('gt')

            if full_lq.dim() == 4:
                full_lq = full_lq.unsqueeze(0)
            if full_gt is not None and full_gt.dim() == 4:
                full_gt = full_gt.unsqueeze(0)

            batch_size, total_frames = full_lq.size(0), full_lq.size(1)
            if batch_size != 1:
                raise ValueError(f'RTNGANModel validation expects batch_size=1, but got {batch_size}.')

            key = val_data.get('key', ['sample'])
            if isinstance(key, list):
                clip_id = key[0]
            else:
                clip_id = key
            clip_id = str(clip_id)
            # 保存目录名：
            # - 若 key 类似 "clip/frame"，则取 clip 部分
            # - 若 key 本身已是 clip（如 "video_id/clip_id"），则保留完整 key，避免不同 clip 覆盖
            folder = clip_id
            if '/' in clip_id:
                parts = clip_id.split('/')
                last_part = parts[-1]
                _, ext = osp.splitext(last_part)
                if last_part.isdigit() or ext.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
                    folder = '/'.join(parts[:-1])

            name_list = val_data.get('name_list')
            if isinstance(name_list, list):
                if len(name_list) and isinstance(name_list[0], (list, tuple)):
                    frame_names = [str(n[0]) for n in name_list]
                else:
                    frame_names = [str(n) for n in name_list]
            else:
                frame_names = None
            if not frame_names or len(frame_names) != total_frames:
                frame_names = [f'{idx:05d}' for idx in range(total_frames)]

            # 计算分块数量：drop_tail_incomplete_chunk 只在显式开启时丢弃尾部不足一块的帧。
            if chunk_size_cfg <= 0:
                chunk_size = total_frames
                effective_frames = total_frames
                num_chunks = 1
                dropped_tail = 0
            else:
                # 光流计算至少需要 2 帧
                chunk_size = max(chunk_size_cfg, 2)
                if drop_tail_incomplete_chunk:
                    effective_frames = (total_frames // chunk_size) * chunk_size
                    dropped_tail = total_frames - effective_frames
                    if effective_frames < 2:
                        logger.warning(
                            f'Skipping {folder}: {total_frames} frames cannot form a {chunk_size}-frame '
                            'validation chunk with drop_tail_incomplete_chunk=true.')
                        continue
                    num_chunks = effective_frames // chunk_size
                else:
                    effective_frames = total_frames
                    dropped_tail = 0
                    num_chunks = (total_frames + chunk_size - 1) // chunk_size
                    # 确保最后一个 chunk 至少有 2 帧（余数为 1 时，将最后 1 帧并入前一块）
                    last_chunk_frames = total_frames - (num_chunks - 1) * chunk_size
                    if last_chunk_frames < 2 and num_chunks > 1:
                        num_chunks -= 1
            eval_frame_names = frame_names[:effective_frames]

            # 可选：若数据集做了 padding，对保存/评测裁回原始尺寸
            def _to_int(v):
                if v is None:
                    return None
                if torch.is_tensor(v):
                    if v.numel() == 0:
                        return None
                    return int(v.reshape(-1)[0].item())
                if isinstance(v, (list, tuple)):
                    if not v:
                        return None
                    return _to_int(v[0])
                return int(v)

            orig_h = _to_int(val_data.get('orig_h'))
            orig_w = _to_int(val_data.get('orig_w'))

            # 准备保存目录
            if save_img:
                folder_dir = osp.join(vis_root, folder)
                if isinstance(current_iter, int):
                    # 训练验证时：添加 iter 前缀区分不同迭代的结果
                    iter_suffix = f'{current_iter:08d}'
                    base_dir = osp.join(folder_dir, f'iter_{iter_suffix}')
                else:
                    # 测试时：直接保存到 folder 目录下，不需要 iter 前缀
                    base_dir = folder_dir
                output_dir = osp.join(base_dir, 'outputs')
                mask_dir = osp.join(base_dir, 'masks')
                feature_dir = osp.join(base_dir, 'features')
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(mask_dir, exist_ok=True)
                os.makedirs(feature_dir, exist_ok=True)

                # ★ Resume 逻辑：跳过已完成的 folder
                existing_imgs = [f for f in os.listdir(output_dir) if f.endswith('.png')]
                logger.info(
                    f'[Resume check] {folder}: output_dir={output_dir}, existing={len(existing_imgs)}, '
                    f'total_frames={total_frames}, effective_frames={effective_frames}, dropped_tail={dropped_tail}')
                # Fix C: only honor the skip when the recorded checkpoint
                # fingerprint matches the current one (test mode); otherwise
                # re-infer so outputs always belong to the current checkpoint.
                _resume_ok = (not is_test_mode) or fingerprint_matches(output_dir, ckpt_fingerprint)
                if len(existing_imgs) >= effective_frames and not _resume_ok:
                    logger.warning(
                        f'[Resume] {folder}: {len(existing_imgs)} PNG(s) exist but the checkpoint '
                        f'fingerprint is missing/different — re-inferring so results match the '
                        f'current checkpoint (no stale cross-checkpoint reuse).')
                if len(existing_imgs) >= effective_frames and _resume_ok:
                    logger.info(
                        f'Skipping {folder}: {len(existing_imgs)} images already exist (need {effective_frames})')
                    # 从已保存的图片读取指标
                    if with_metrics:
                        import cv2
                        seq_metrics_local = {metric: [] for metric in metrics_cfg}
                        no_ref_metrics = {'calculate_niqe', 'calculate_brisque', 'calculate_clipiqa', 'calculate_clipiqa_plus'}
                        dataset_level_metrics = {'calculate_fid', 'fid'}
                        for frame_name in sorted(eval_frame_names):
                            img_stem = osp.splitext(frame_name)[0]
                            if suffix:
                                save_name = f'{img_stem}_{suffix}.png'
                            else:
                                save_name = f'{img_stem}_{self.opt["name"]}.png'
                            img_path = osp.join(output_dir, save_name)
                            if not osp.exists(img_path):
                                continue
                            result_img = cv2.imread(img_path)
                            if result_img is None:
                                continue
                            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                            # GT 图片（如果有参考指标需要）
                            gt_img_metric = None
                            if full_gt is not None:
                                frame_idx = frame_names.index(frame_name)
                                gt_tensor = full_gt[0, frame_idx]
                                gt_img_metric = tensor2img([gt_tensor], min_max=vis_min_max)
                                if orig_h is not None and orig_w is not None:
                                    if gt_img_metric.shape[0] >= orig_h and gt_img_metric.shape[1] >= orig_w:
                                        gt_img_metric = gt_img_metric[:orig_h, :orig_w]
                            for metric_name, metric_opt in metrics_cfg.items():
                                metric_type = metric_opt.get('type', metric_name)
                                if metric_type in dataset_level_metrics or metric_name in dataset_level_metrics:
                                    continue
                                is_no_ref = metric_type in no_ref_metrics
                                if is_no_ref:
                                    value = calculate_metric({'img': result_img}, metric_opt)
                                    if not math.isnan(value):
                                        aggregated_metrics[metric_name].append(value)
                                        seq_metrics_local[metric_name].append(value)
                                elif gt_img_metric is not None:
                                    value = calculate_metric({'img': result_img, 'img2': gt_img_metric}, metric_opt)
                                    if not math.isnan(value):
                                        aggregated_metrics[metric_name].append(value)
                                        seq_metrics_local[metric_name].append(value)
                        # 输出当前序列的评价指标
                        if any(seq_metrics_local.values()):
                            parts = []
                            for mn, vals in seq_metrics_local.items():
                                if vals:
                                    parts.append(f'{mn}: {sum(vals)/len(vals):.4f}')
                            if parts:
                                logger.info(f'  -> {folder} metrics (from cache): {", ".join(parts)}')
                    continue

            if chunk_size_cfg <= 0:
                logger.info(f'Processing {folder}: {total_frames} frames (no chunking)')
            else:
                logger.info(
                    f'Processing {folder}: {total_frames} frames in {num_chunks} chunks '
                    f'(chunk_size={chunk_size}, dropped_tail={dropped_tail}, effective_frames={effective_frames})')

            # 当前序列的指标收集器
            if with_metrics:
                seq_metrics = {metric: [] for metric in metrics_cfg}

            # 收集所有帧的结果
            all_result_tensors = []
            all_gt_tensors = []

            # ====== chunk 级 batch 推理（test_batch_size 控制并行 chunk 数）======
            # default test_batch_size=1 退化为逐 chunk，训练 validation 路径行为不变。
            test_batch_size = self.opt['val'].get('test_batch_size', 1)
            try:
                test_batch_size = int(test_batch_size)
            except (TypeError, ValueError):
                test_batch_size = 1
            batch_groups = plan_chunk_batches(num_chunks, chunk_size, effective_frames, test_batch_size)

            # 阶段 1：按 batch group 前向推理，缓存每个 chunk 的结果（CPU）
            chunk_cache = {}
            first_mask_pack = None
            first_feature_pack = None
            batch_pbar = tqdm(batch_groups, desc=f'{folder}', unit='batch', leave=True)
            for group in batch_pbar:
                batch_pbar.set_postfix(frames=f'{group[-1][2]}/{effective_frames}')

                lq_list = []
                gt_list = []
                for (gi, gs, ge) in group:
                    clq = full_lq[:, gs:ge, :, :, :]
                    cgt = full_gt[:, gs:ge, :, :, :] if full_gt is not None else None
                    # 极端情况：只有 1 帧时复制一帧，避免光流模块报错（仅用于推理）
                    if clq.size(1) < 2:
                        clq = torch.cat([clq, clq[:, -1:, :, :, :].clone()], dim=1)
                        if cgt is not None:
                            cgt = torch.cat([cgt, cgt[:, -1:, :, :, :].clone()], dim=1)
                    lq_list.append(clq)
                    if cgt is not None:
                        gt_list.append(cgt)

                self.lq = torch.cat(lq_list, dim=0).to(self.device, non_blocking=True).float().contiguous()
                if gt_list:
                    self.gt = torch.cat(gt_list, dim=0).to(self.device, non_blocking=True).float().contiguous()

                # mask/feature 可视化仅对含首个 chunk 的 group 收集
                group_has_first = any(gi == 0 for gi, _, _ in group)
                self._collect_visual_debug = save_img and group_has_first

                # 项目级 chunk hook：仅单 chunk（bs 退化）时保持原 per-chunk 语义
                if len(group) == 1:
                    self.prepare_validation_chunk(val_data, group[0][1], group[0][2], frame_names)

                # 推理（一次 forward 处理整组等尺寸 chunk）
                self.test()

                visuals = self.get_current_visuals()
                batch_results = visuals['result']
                batch_gts = visuals.get('gt')
                if batch_results.dim() == 4:
                    batch_results = batch_results.unsqueeze(0)
                if batch_gts is not None and batch_gts.dim() == 4:
                    batch_gts = batch_gts.unsqueeze(0)

                for local_b, (gi, gs, ge) in enumerate(group):
                    actual_len = ge - gs
                    res = batch_results[local_b][:actual_len].detach().cpu()
                    gtv = batch_gts[local_b][:actual_len].detach().cpu() if batch_gts is not None else None
                    chunk_cache[gi] = (res, gtv)
                    if gi == 0:
                        first_mask_pack = getattr(self, '_last_visual_masks', None)
                        first_feature_pack = getattr(self, '_last_visual_features', None)

                del self.lq, self.output
                if hasattr(self, 'gt'):
                    del self.gt
                if torch.cuda.is_available() and self.opt.get('val', {}).get('empty_cache', True):
                    torch.cuda.empty_cache()
                self._last_visual_masks = None
                self._last_visual_features = None

            # 阶段 2：逐 chunk 保存输出 + 计算逐帧指标（与原逐帧逻辑一致）
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = effective_frames if chunk_idx == num_chunks - 1 else min(
                    (chunk_idx + 1) * chunk_size, effective_frames)

                chunk_result_tensors, chunk_gt_tensors = chunk_cache[chunk_idx]

                all_result_tensors.append(chunk_result_tensors)
                if chunk_gt_tensors is not None:
                    all_gt_tensors.append(chunk_gt_tensors)

                # 收集 mask 和 feature（仅第一个 chunk）
                if chunk_idx == 0:
                    mask_pack = first_mask_pack
                    feature_pack = first_feature_pack
                else:
                    mask_pack = None
                    feature_pack = None

                # 逐帧处理当前块
                chunk_frame_names = frame_names[start_idx:end_idx]
                for local_idx, frame_name in enumerate(chunk_frame_names):
                    global_idx = start_idx + local_idx
                    result_img = tensor2img([chunk_result_tensors[local_idx]], min_max=vis_min_max)
                    gt_img = tensor2img([chunk_gt_tensors[local_idx]], min_max=vis_min_max) if chunk_gt_tensors is not None else None
                    if orig_h is not None and orig_w is not None:
                        if result_img.shape[0] >= orig_h and result_img.shape[1] >= orig_w:
                            result_img = result_img[:orig_h, :orig_w]
                        if gt_img is not None and gt_img.shape[0] >= orig_h and gt_img.shape[1] >= orig_w:
                            gt_img = gt_img[:orig_h, :orig_w]

                    if save_img:
                        img_stem = osp.splitext(frame_name)[0]
                        if suffix:
                            save_name = f'{img_stem}_{suffix}.png'
                        else:
                            save_name = f'{img_stem}_{self.opt["name"]}.png'
                        imwrite(result_img, osp.join(output_dir, save_name))

                        # mask/feature 仅对第一个 chunk 可用
                        if mask_pack and local_idx < len(next(iter(mask_pack.values()), [])):
                            for direction, tensors in mask_pack.items():
                                if tensors is None:
                                    continue
                                dir_path = osp.join(mask_dir, direction)
                                os.makedirs(dir_path, exist_ok=True)
                                mask_img = self._mask_to_uint8(tensors[local_idx])
                                if orig_h is not None and orig_w is not None and mask_img.ndim >= 2:
                                    if mask_img.shape[0] >= orig_h and mask_img.shape[1] >= orig_w:
                                        mask_img = mask_img[:orig_h, :orig_w]
                                mask_name = f'{img_stem}_{direction}.png'
                                imwrite(mask_img, osp.join(dir_path, mask_name))

                        if feature_pack and local_idx < len(next(iter(feature_pack.values()), [])):
                            for name, tensors in feature_pack.items():
                                if tensors is None:
                                    continue
                                dir_path = osp.join(feature_dir, name)
                                os.makedirs(dir_path, exist_ok=True)
                                feature_img = self._feature_to_uint8(tensors[local_idx])
                                if orig_h is not None and orig_w is not None and feature_img.ndim >= 2:
                                    if feature_img.shape[0] >= orig_h and feature_img.shape[1] >= orig_w:
                                        feature_img = feature_img[:orig_h, :orig_w]
                                feat_name = f'{img_stem}_{name}.png'
                                imwrite(feature_img, osp.join(dir_path, feat_name))

                    if with_metrics:
                        # 无参考指标列表（不需要 GT）
                        no_ref_metrics = {'calculate_niqe', 'calculate_brisque', 'calculate_clipiqa', 'calculate_clipiqa_plus'}
                        # 数据集级别指标（需要单独处理，不在帧级别计算）
                        dataset_level_metrics = {'calculate_fid', 'fid'}

                        for metric_name, metric_opt in metrics_cfg.items():
                            metric_type = metric_opt.get('type', metric_name)

                            # 跳过数据集级别指标
                            if metric_type in dataset_level_metrics or metric_name in dataset_level_metrics:
                                continue

                            is_no_ref = metric_type in no_ref_metrics

                            # 无参考指标只需要 img，有参考指标需要 img 和 img2(gt)
                            if is_no_ref:
                                metric_data = {'img': result_img}
                                value = calculate_metric(metric_data, metric_opt)
                                if not math.isnan(value):
                                    aggregated_metrics[metric_name].append(value)
                                    seq_metrics[metric_name].append(value)
                            elif gt_img is not None:
                                metric_data = {'img': result_img, 'img2': gt_img}
                                value = calculate_metric(metric_data, metric_opt)
                                if not math.isnan(value):
                                    aggregated_metrics[metric_name].append(value)
                                    seq_metrics[metric_name].append(value)

            # 输出当前序列的评价指标
            if with_metrics and any(seq_metrics.values()):
                seq_metric_parts = []
                for metric_name, values in seq_metrics.items():
                    if values:
                        avg_val = float(sum(values) / len(values))
                        seq_metric_parts.append(f'{metric_name}: {avg_val:.4f}')
                if seq_metric_parts:
                    logger.info(f'  -> {folder} metrics: {", ".join(seq_metric_parts)}')

            # Fix C: record which checkpoint produced this clip's outputs.
            if save_img and is_test_mode:
                write_fingerprint(output_dir, ckpt_fingerprint)

        if with_metrics and any(aggregated_metrics.values()):
            log_lines = [f'Validation {dataset_name}']
            for metric_name, values in aggregated_metrics.items():
                if not values:
                    continue
                avg_val = float(sum(values) / len(values))
                self._update_best_metric_result(dataset_name, metric_name, avg_val, current_iter)
                best_info = self.best_metric_results[dataset_name][metric_name]
                line = (f'\t# {metric_name}: {avg_val:.4f}\tBest: '
                        f'{best_info["val"]:.4f} @ {best_info["iter"]} iter')
                log_lines.append(line)
                if tb_logger:
                    tb_logger.add_scalar(f'metrics/{metric_name}', avg_val, current_iter)
            logger.info('\n'.join(log_lines))

            # Save Best Model 检查
            if getattr(self, 'save_best_enabled', False):
                target = self.save_best_metric
                if target in aggregated_metrics and aggregated_metrics[target]:
                    best_info = self.best_metric_results[dataset_name].get(target, {})
                    if best_info.get('iter') == current_iter:
                        avg_val = float(sum(aggregated_metrics[target]) / len(aggregated_metrics[target]))
                        self._save_best_model(current_iter, target, avg_val)

        self._collect_visual_debug = False

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def _manage_flow_freeze(self, current_iter):
        if self.fix_flow_iters <= 0:
            return

        if current_iter == 1 and not self._flow_frozen:
            self._set_flow_requires_grad(False)
            self._flow_frozen = True
        elif self._flow_frozen and current_iter > self.fix_flow_iters:
            self._set_flow_requires_grad(True)
            self._flow_frozen = False

    def _set_flow_requires_grad(self, flag):
        for name, param in self.net_g.named_parameters():
            if any(flow_key in name for flow_key in self.flow_modules):
                param.requires_grad = flag

    # ==================== Save Best Model ====================

    def _save_best_model(self, current_iter, metric_name, metric_value):
        """保存最佳模型为 net_g_best.pth（覆盖式）"""
        save_path = osp.join(self.opt['path']['models'], 'net_g_best.pth')
        if hasattr(self, 'ema_decay') and self.ema_decay > 0:
            save_dict = {
                'params': self.get_bare_model(self.net_g).state_dict(),
                'params_ema': self.get_bare_model(self.net_g_ema).state_dict(),
                'best_iter': current_iter,
                'best_metric': metric_name,
                'best_value': metric_value,
            }
        else:
            save_dict = {
                'params': self.get_bare_model(self.net_g).state_dict(),
                'best_iter': current_iter,
                'best_metric': metric_name,
                'best_value': metric_value,
            }
        torch.save(save_dict, save_path)
        logger = get_root_logger()
        logger.info(f'Best model saved: net_g_best.pth ({metric_name}={metric_value:.4f} @ iter {current_iter})')
