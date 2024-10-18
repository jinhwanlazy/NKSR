# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import gc
import random
from typing import Optional

import numpy as np
import torch
from fvdb import JaggedTensor
from nksr import NKSRNetwork, SparseFeatureHierarchy
from nksr.configs import load_checkpoint_from_url
from nksr.fields import KernelField, LayerField, NeuralField
from nksr.utils import jagged_cat
from pycg import exp, vis
from pycg.isometry import ScaledIsometry

from dataset.base import DatasetSpec as DS
from dataset.base import list_collate
from models.base_model import BaseModel


class Model(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.network = NKSRNetwork(self.hparams)
        if self.hparams.url:
            ckpt_data = load_checkpoint_from_url(self.hparams.url)
            self.network.load_state_dict(ckpt_data['state_dict'])

    @exp.mem_profile(every=1)
    def forward(self, batch, out: dict):
        input_xyz = JaggedTensor(batch[DS.INPUT_PC])

        if self.hparams.feature == 'normal':
            assert DS.TARGET_NORMAL in batch.keys(), "normal must be provided in this config!"
            feat = JaggedTensor(batch[DS.TARGET_NORMAL])
        elif self.hparams.feature == 'sensor':
            assert DS.INPUT_SENSOR_POS in batch.keys(), "sensor must be provided in this config!"
            view_dir = JaggedTensor(batch[DS.INPUT_SENSOR_POS]) - input_xyz
            view_dir.jdata = view_dir.jdata / \
                (torch.linalg.norm(view_dir.jdata, dim=-1, keepdim=True) + 1e-6)
            feat = view_dir
        else:
            feat = None
        out['feat'] = feat

        enc_svh = SparseFeatureHierarchy(
            voxel_size=self.hparams.voxel_size,
            depth=self.hparams.tree_depth,
            device=self.device
        )
        enc_svh.build_point_splatting(input_xyz)

        # Compute density by computing points per voxel.
        if self.hparams.runtime_density:
            q_xyz = torch.div(input_xyz.jdata, self.hparams.voxel_size).floor().int()
            q_xyz = torch.unique(torch.cat([q_xyz, input_xyz.jidx[:, None]], dim=-1), dim=0)
            density = input_xyz.jdata.size(0) / q_xyz.size(0)
            exp.logger.info(f"Density {density}, # pts = {input_xyz.jdata.size(0)}")

        if self.hparams.runtime_visualize:
            for batch_idx in range(input_xyz.joffsets.size(0)):
                vis.show_3d([vis.pointcloud(
                    input_xyz[batch_idx].jdata,
                    normal=feat[batch_idx].jdata if feat is not None else None)],
                    enc_svh.get_visualization(batch_idx))

        feat = self.network.encoder(input_xyz, feat, enc_svh, 0)
        feat, dec_svh, udf_svh = self.network.unet(
            feat, enc_svh,
            adaptive_depth=self.hparams.adaptive_depth,
            gt_decoder_svh=out.get('gt_svh', None)
        )

        if all([dec_svh.grids[d].total_voxels == 0 for d in range(self.hparams.adaptive_depth)]):
            if self.trainer.training or self.trainer.validating:
                # In case training data is corrupted (pd & gt not aligned)...
                exp.logger.warning("Empty grid detected during training/validation.")
                return None

        out.update({'enc_svh': enc_svh, 'dec_svh': dec_svh, 'dec_tmp_svh': udf_svh})

        if self.hparams.geometry == 'kernel':
            output_field = KernelField(
                svh=dec_svh,
                interpolator=self.network.interpolators,
                features=feat.basis_features,
                approx_kernel_grad=False
            )
            if self.hparams.solver_verbose:
                output_field.solver_config['verbose'] = True

            normal_xyz = jagged_cat([dec_svh.get_voxel_centers(d) for d in range(self.hparams.adaptive_depth)])
            normal_value = jagged_cat([feat.normal_features[d] for d in range(self.hparams.adaptive_depth)])

            normal_weight = self.hparams.solver.normal_weight * (self.hparams.voxel_size ** 2) / \
                (normal_xyz.joffsets[1:] - normal_xyz.joffsets[:-1])
            output_field.solve(
                pos_xyz=input_xyz,
                normal_xyz=normal_xyz,
                normal_value=-normal_value,
                pos_weight=self.hparams.solver.pos_weight / (input_xyz.joffsets[1:] - input_xyz.joffsets[:-1]),
                normal_weight=normal_weight,
                reg_weight=1.0
            )

        elif self.hparams.geometry == 'neural':
            output_field = NeuralField(
                svh=dec_svh,
                decoder=self.network.sdf_decoder,
                features=feat.basis_features
            )

        else:
            raise NotImplementedError

        if self.hparams.udf.enabled:
            mask_field = NeuralField(
                svh=udf_svh,
                decoder=self.network.udf_decoder,
                features=feat.udf_features
            )
            mask_field.set_level_set(2 * self.hparams.voxel_size)
        else:
            mask_field = LayerField(dec_svh, self.hparams.adaptive_depth)
        output_field.set_mask_field(mask_field)

        out.update({
            'structure_features': feat.structure_features,
            'normal_features': feat.normal_features,
            'basis_features': feat.basis_features,
            'field': output_field
        })
        return out

    def transform_field_visualize(self, field: JaggedTensor):
        spatial_config = self.hparams.supervision.spatial
        if spatial_config.gt_type == "binary":
            return field.jagged_like(torch.tanh(field.jdata))
        else:
            if spatial_config.pd_transform:
                from models.loss import KitchenSinkMetricLoss
                return KitchenSinkMetricLoss.transform_field(self.hparams, field)
            else:
                return field

    def compute_gt_svh(self, batch, out):
        if 'gt_svh' in out.keys():
            return out['gt_svh']

        if DS.GT_GEOMETRY in batch.keys():
            assert len(batch[DS.GT_GEOMETRY]) == 1, "Does not support batching now."
            ref_geometry = batch[DS.GT_GEOMETRY][0]
            ref_xyz, ref_normal, _ = ref_geometry.torch_attr()
        else:
            ref_xyz = JaggedTensor(batch[DS.GT_DENSE_PC])
            ref_normal = JaggedTensor(batch[DS.GT_DENSE_NORMAL])

        gt_svh = SparseFeatureHierarchy(
            voxel_size=self.hparams.voxel_size,
            depth=self.hparams.tree_depth,
            device=self.device
        )

        if self.hparams.adaptive_policy.method == "normal":
            gt_svh.build_adaptive_normal_variation(
                ref_xyz, ref_normal,
                tau=self.hparams.adaptive_policy.tau,
                adaptive_depth=self.hparams.adaptive_depth
            )
        else:
            # Not recommended, removed
            raise NotImplementedError

        out['gt_svh'] = gt_svh
        return gt_svh

    @exp.mem_profile(every=1)
    def compute_loss(self, batch, out, compute_metric: bool):
        loss_dict = exp.TorchLossMeter()
        metric_dict = exp.TorchLossMeter()

        from models.loss import (GTSurfaceLoss, ShapeNetIoUMetric, SpatialLoss,
                                 StructureLoss, UDFLoss)

        SpatialLoss.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)
        GTSurfaceLoss.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)

        self.compute_gt_svh(batch, out)
        StructureLoss.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)

        UDFLoss.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)
        ShapeNetIoUMetric.apply(self.hparams, loss_dict, metric_dict, batch, out, compute_metric)

        return loss_dict, metric_dict

    def log_visualizations(self, batch, out, batch_idx):
        if self.trainer.logger is None:
            return
        with torch.no_grad():
            field = out['field']
            if field is None:
                return

            if not self.hparams.no_mesh_vis:
                mesh_res = field.extract_dual_mesh()
                mesh = vis.mesh(mesh_res.v, mesh_res.f)
                self.log_geometry("pd_mesh", mesh)

    def should_use_pd_structure(self, is_val):
        # In case this returns True:
        #   - The tree generation would completely rely on prediction, so does the supervision signal.
        prob = (self.trainer.global_step - self.hparams.structure_schedule.start_step) / \
               (self.hparams.structure_schedule.end_step - self.hparams.structure_schedule.start_step)
        prob = min(max(prob, 0.0), 1.0)
        if not is_val:
            self.log("pd_struct_prob", prob, prog_bar=True, on_step=True, on_epoch=False)
        return random.random() < prob

    # @exp.mem_profile(every=1)
    def train_val_step(self, batch, batch_idx, is_val):
        if batch_idx % 100 == 0:
            gc.collect()

        out = {'idx': batch_idx}
        if not self.should_use_pd_structure(is_val):
            self.compute_gt_svh(batch, out)

        with exp.pt_profile_named("forward"):
            out = self(batch, out)

        # OOM Guard.
        if out is None:
            return None

        with exp.pt_profile_named("loss"):
            loss_dict, metric_dict = self.compute_loss(batch, out, compute_metric=is_val)

        if not is_val:
            self.log_dict_prefix('train_loss', loss_dict)
            if batch_idx % 200 == 0:
                self.log_visualizations(batch, out, batch_idx)
        else:
            self.log_dict_prefix('val_metric', metric_dict)
            self.log_dict_prefix('val_loss', loss_dict)

        loss_sum = loss_dict.get_sum()
        if is_val and torch.any(torch.isnan(loss_sum)):
            exp.logger.warning("Get nan val loss during validation. Setting to 0.")
            loss_sum = 0
        self.log('val_loss' if is_val else 'train_loss/sum', loss_sum)

        return loss_sum

    def test_step(self, batch, batch_idx):
        test_transform = None
        if self.hparams.test_transform is not None:
            test_transform = ScaledIsometry.from_matrix(np.array(self.hparams.test_transform))

        out = {'idx': batch_idx}
        self.transform_batch_input(batch, test_transform)

        if self.hparams.test_use_gt_structure:
            self.compute_gt_svh(batch, out)

        out = self(batch, out)

        # loss_dict, metric_dict = self.compute_loss(batch, out, compute_metric=True)
        # self.log_dict(loss_dict)
        # self.log_dict(metric_dict)

        self.transform_batch_input(batch, test_transform.inv() if test_transform is not None else None)

        field = out['field']
        batch_size = field.svh.grids[-1].grid_count
        
        for b in range(batch_size):
            exp.logger.info(f"Now visualizing data {b + 1} of {batch_size}...")
            self._test_metric_and_visualize(b, batch, field, test_transform)

    def _test_metric_and_visualize(self, batch_idx: int, batch, field, test_transform):
        mesh_res = field.extract_dual_mesh(grid_upsample=self.hparams.test_n_upsample, batch_idx=batch_idx)
        mesh = vis.mesh(mesh_res.v, mesh_res.f)

        if test_transform is not None:
            mesh = test_transform.inv() @ mesh

        if DS.GT_GEOMETRY in batch.keys():
            ref_geometry = batch[DS.GT_GEOMETRY][batch_idx]
            ref_xyz, ref_normal, _ = ref_geometry.torch_attr()
        else:
            ref_geometry = None
            ref_xyz, ref_normal = batch[DS.GT_DENSE_PC][batch_idx], batch[DS.GT_DENSE_NORMAL][batch_idx]

        if self.hparams.test_print_metrics:
            from metrics import MeshEvaluator

            evaluator = MeshEvaluator(
                n_points=int(5e6) if ref_geometry is not None else int(5e5),
                metric_names=MeshEvaluator.ESSENTIAL_METRICS)
            onet_samples = None
            if DS.GT_ONET_SAMPLE in batch:
                onet_samples = [
                    batch[DS.GT_ONET_SAMPLE][0][batch_idx].cpu().numpy(),
                    batch[DS.GT_ONET_SAMPLE][1][batch_idx].cpu().numpy()
                ]
            eval_dict = evaluator.eval_mesh(mesh, ref_xyz, ref_normal, onet_samples=onet_samples)
            self.log_dict(eval_dict)
            exp.logger.info("Metric: " + ", ".join([f"{k} = {v:.4f}" for k, v in eval_dict.items()]))

        input_pc = batch[DS.INPUT_PC][batch_idx]
        self.log('source', batch[DS.SHAPE_NAME][batch_idx])

        if self.record_folder is not None:
            # Record also input for comparison.
            self.test_log_data({
                'input': vis.pointcloud(input_pc, normal=out['feat']),
                'mesh': mesh
            })

        if self.hparams.visualize:
            exp.logger.info(f"Visualizing data {batch[DS.SHAPE_NAME][batch_idx]}...")
            scenes = vis.show_3d(
                [vis.pointcloud(input_pc), mesh],
                [vis.pointcloud(ref_xyz, normal=ref_normal)],
                point_size=1, use_new_api=False, show=not self.overfit_logger.working,
                viewport_shading='NORMAL', cam_path=f"../cameras/{self.get_dataset_short_name()}.bin"
            )
            self.overfit_logger.log_overfit_visuals({'scene': scenes[0]})

    @classmethod
    def transform_batch_input(cls, batch, transform: Optional[ScaledIsometry]):
        if transform is None:
            return
        batch_size = len(batch[DS.INPUT_PC])
        for b in range(batch_size):
            batch[DS.INPUT_PC][b] = transform @ batch[DS.INPUT_PC][b]
            if DS.TARGET_NORMAL in batch:
                batch[DS.TARGET_NORMAL][b] = transform.rotation @ batch[DS.TARGET_NORMAL][b]
            if DS.INPUT_SENSOR_POS in batch:
                batch[DS.INPUT_SENSOR_POS][b] = transform @ batch[DS.INPUT_SENSOR_POS][b]

    def get_dataset_spec(self):
        all_specs = [DS.SHAPE_NAME, DS.INPUT_PC,
                     DS.GT_DENSE_PC, DS.GT_DENSE_NORMAL, DS.GT_ONET_SAMPLE,
                     DS.GT_GEOMETRY]
        if self.hparams.feature == 'normal':
            all_specs.append(DS.TARGET_NORMAL)
        elif self.hparams.feature == 'sensor':
            all_specs.append(DS.INPUT_SENSOR_POS)
        return all_specs

    def get_collate_fn(self):
        return list_collate

    def get_hparams_metrics(self):
        return [('val_loss', True)]
