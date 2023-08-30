import os, sys
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger


class NeRFSystem(LightningModule):
    def __init__(self, hparams_1):
        super(NeRFSystem, self).__init__()
        self.hparams_1 = hparams_1

        self.loss = loss_dict[hparams_1.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams_1.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams_1.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams_1.chunk],
                            self.hparams_1.N_samples,
                            self.hparams_1.use_disp,
                            self.hparams_1.perturb,
                            self.hparams_1.noise_std,
                            self.hparams_1.N_importance,
                            self.hparams_1.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams_1.dataset_name]
        kwargs = {'root_dir': self.hparams_1.root_dir,
                  'img_wh': tuple(self.hparams_1.img_wh)}
        if self.hparams_1.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams_1.spheric_poses
            kwargs['val_num'] = self.hparams_1.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams_1, self.models)
        scheduler = get_scheduler(self.hparams_1, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams_1.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams_1.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }


if __name__ == '__main__':
    hparams_1 = get_opts()
    system = NeRFSystem(hparams_1)
    # checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams_1.exp_name}',
    #                                                             '{epoch:d}'),
    #                                       monitor='val/loss',
    #                                       mode='min',
    #                                       save_top_k=5,)
    checkpoint_callback = ModelCheckpoint(
                                                                                dirpath=os.path.join(f'ckpts/{hparams_1.exp_name}'),
                                                                                filename='{epoch:d}',
                                                                                monitor='val/loss',
                                                                                mode='min',
                                                                                save_top_k=5,
                                                                                )


    logger = WandbLogger()
    

    # trainer = Trainer(max_epochs=hparams_1.num_epochs,
    #                   callbacks=checkpoint_callback,
    #                   resume_from_checkpoint=hparams_1.ckpt_path,
    #                   logger=logger,
    #                   early_stop_callback=None,
    #                   weights_summary=None,
    #                   progress_bar_refresh_rate=1,
    #                   gpus=hparams_1.num_gpus,
    #                   distributed_backend='ddp' if hparams_1.num_gpus>1 else None,
    #                   num_sanity_val_steps=1,
    #                   benchmark=True,
    #                   profiler=hparams_1.num_gpus==1)

    trainer = Trainer(
                        max_epochs=hparams_1.num_epochs,
                        callbacks=[checkpoint_callback],
                        resume_from_checkpoint=hparams_1.ckpt_path,
                        logger=logger,
                        enable_checkpointing=True,
                        enable_progress_bar=True,
                        devices=hparams_1.num_gpus,
                        strategy='ddp' if hparams_1.num_gpus>1 else None,
                        num_sanity_val_steps=1,
                        benchmark=True,
                        profiler=hparams_1.num_gpus==1
                    )

    trainer.fit(system)
