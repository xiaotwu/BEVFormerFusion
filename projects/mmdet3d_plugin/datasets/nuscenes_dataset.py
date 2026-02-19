import copy
import random
from collections.abc import KeysView, ValuesView, ItemsView
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from projects.mmdet3d_plugin.models.utils.visual import save_tensor  # if unused, safe to remove
from .nuscnes_eval import NuScenesEval_custom


def _materialize_mapping_views(x):
    """Convert dict view objects (dict_keys / dict_values / dict_items) into
    regular Python containers so they can be serialized and deep-copied safely.
    """
    if isinstance(x, (KeysView, ValuesView, ItemsView)):
        return list(x)
    if isinstance(x, dict):
        return {k: _materialize_mapping_views(v) for k, v in x.items()}
    return x


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset for BEVFormer.

    This dataset extends the default NuScenesDataset by:
      * Adding camera intrinsics / extrinsics.
      * Adding CAN bus and temporal queuing support (queue_length, union2one).
      * Adding robust debug utilities and custom nuScenes evaluation wrapper.
    """

    def __init__(self,
                 queue_length=4,
                 bev_size=(100, 100),
                 overlap_test=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

        # ---- Fix eval_detection_configs if it contains dict views ----
        if hasattr(self, 'eval_detection_configs') and self.eval_detection_configs is not None:
            self.eval_detection_configs = _materialize_mapping_views(
                self.eval_detection_configs
            )

            cfg = self.eval_detection_configs
            # Some versions store class_names as dict_keys instead of list
            if hasattr(cfg, 'class_names') and not isinstance(cfg.class_names, list):
                try:
                    cfg.class_names = list(cfg.class_names)
                except TypeError:
                    pass

        # Ensure .flag exists (needed by GroupSampler)
        if not hasattr(self, 'flag'):
            self.flag = np.ones(len(self), dtype=np.uint8)

        # Some helper attributes that sometimes end up as dict views
        if isinstance(getattr(self, 'tokens', None), type({}.keys())):
            self.tokens = list(self.tokens)
        if isinstance(getattr(self, 'scene_to_samples', None), dict):
            self.scene_list = list(self.scene_to_samples.keys())

        #import inspect
        #print("[DBG] Dataset class:", type(self))
        #print("[DBG] Dataset file:", inspect.getfile(type(self)))

    # ------------------------------------------------------------------
    #   TRAIN PIPELINE WITH TEMPORAL QUEUE
    # ------------------------------------------------------------------
    def prepare_train_data(self, index):
        """Prepare training data with temporal queue for BEVFormer."""
        queue = []

        index_list = list(range(index - self.queue_length, index))
        random.shuffle(index_list)
        # Use all but the earliest, then append current index
        index_list = sorted(index_list[1:])
        index_list.append(index)

        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)

            if self.filter_empty_gt and \
                    (example is None or
                     ~(example['gt_labels_3d']._data != -1).any()):
                return None

            queue.append(example)

        return self.union2one(queue)

    def union2one(self, queue):
        """Merge a temporal queue into a single training sample.

        The last frame will carry the stacked images and all img_metas
        for the temporal sequence.
        """
        imgs_list = [each['img'].data for each in queue]

        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None

        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data

            if metas_map[i]['scene_token'] != prev_scene_token:
                # New scene: reset
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                # Same scene: encode relative motion into can_bus
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        # Stack images over time, keep metas as a dict of timesteps
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        return queue[-1]

    # ------------------------------------------------------------------
    #   DATA INFO / CAN BUS / CAMERA MATRICES
    # ------------------------------------------------------------------
    def get_data_info(self, index):
        """Get per-sample info dict used by the pipeline."""
        info = self.data_infos[index]

        prev_idx = info.get('prev', None)
        next_idx = info.get('next', None)

        # Scene token and frame index fallbacks
        scene_token = info.get('scene_token', info.get('token', f'NO_SCENE_{index}'))
        frame_idx = info.get('frame_idx', info.get('sample_idx', index))

        # ---- CAN BUS (safe default) ----
        can_bus = info.get('can_bus', None)
        if can_bus is None:
            # 18-dim vector as usually used in BEVFormer
            can_bus = np.zeros(18, dtype=np.float32)

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=prev_idx,
            next_idx=next_idx,
            scene_token=scene_token,
            can_bus=can_bus,
            frame_idx=frame_idx,
            timestamp=info['timestamp'] / 1e6,
        )

        # -------------- CAMERAS --------------
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []

            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])

                # lidar -> camera
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                lidar2img_rt = (viewpad @ lidar2cam_rt.T)

                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )

        # -------------- ANNOTATIONS --------------
        if not self.test_mode:
            annos = self.get_ann_info(index)

            # Ensure integer labels for Hungarian matcher
            if 'gt_labels_3d' in annos:
                annos['gt_labels_3d'] = np.asarray(annos['gt_labels_3d'],
                                                   dtype=np.int64)
            elif 'gt_labels' in annos:
                annos['gt_labels'] = np.asarray(annos['gt_labels'],
                                                dtype=np.int64)
                annos.setdefault('gt_labels_3d', annos['gt_labels'])

            if annos.get('gt_labels_3d', None) is None:
                annos['gt_labels_3d'] = np.empty((0,), dtype=np.int64)

            input_dict['ann_info'] = annos

        # -------------- ENRICH CAN BUS --------------
        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']

        can_bus[:3] = translation
        can_bus[3:7] = rotation

        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi  # rad
        can_bus[-1] = patch_angle               # deg

        # -------------- img_metas FOR DEBUG / TSA --------------
        img_metas = dict(
            token=info.get('token', None),
            scene_token=scene_token,
            timestamp=info.get('timestamp', None),
            frame_idx=frame_idx,
            sample_idx=info.get('sample_idx', None),
            prev=prev_idx,
            next=next_idx,
            prev_bev_exists=(
                prev_idx is not None and prev_idx != -1 and prev_idx != ""
            ),
        )

        sample_id = img_metas['token']
        if sample_id is None:
            ts = img_metas['timestamp']
            sample_id = str(ts) if ts is not None else None
        if sample_id is None and self.modality['use_camera']:
            fn_list = input_dict.get('img_filename', [])
            if len(fn_list) > 0:
                sample_id = str(fn_list[0]).split('/')[-1]
        img_metas['sample_id'] = sample_id

        input_dict['img_metas'] = img_metas

        return input_dict

    # ------------------------------------------------------------------
    #   DEBUG UTILITIES
    # ------------------------------------------------------------------

    def _debug_print_sample(self, data, idx, split="train"):
        print("[DBG] data keys:", list(data.keys()))
        try:
            print(f"\n[DBG:{split}] idx={idx}")

            def unwrap(x):
                # mmcv DataContainer -> get the actual payload
                if hasattr(x, 'data'):
                    return x.data
                return x

            gt = unwrap(data.get('gt_bboxes_3d', None))

            # DataContainer sometimes stores a list (per-sample) or list-of-lists
            if isinstance(gt, (list, tuple)):
                # often length == 1 here because __getitem__ returns one sample
                gt = gt[0] if len(gt) > 0 else None

            if gt is None:
                print("[DBG] no gt_bboxes_3d")
                return

            print("[DBG] gt_bboxes_3d type:", type(gt))
            print("[DBG] gt_bboxes_3d origin:", getattr(gt, "origin", None))

            t = gt.tensor if hasattr(gt, "tensor") else gt
            print("[DBG] gt tensor type:", type(t))

            if torch.is_tensor(t):
                print("[DBG] gt tensor shape:", tuple(t.shape))
                if t.numel() > 0:
                    print("[DBG] gt first 7:", t[0, :7].detach().cpu().numpy())
            else:
                # fallback if it’s numpy or something else
                import numpy as np
                arr = np.array(t)
                print("[DBG] gt array shape:", arr.shape)
                if arr.size > 0:
                    print("[DBG] gt first 7:", arr.reshape(-1, arr.shape[-1])[0, :7])

        except Exception as e:
            print("[DBG] debug print failed:", repr(e))



    # ------------------------------------------------------------------
    #   __getitem__ WITH SAFE DEBUG
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        """Get item from infos according to the given index."""
        if self.test_mode:
            data = self.prepare_test_data(idx)

            # One-time debug for test mode
            if not hasattr(self, "_dbg_once"):
                self._dbg_once = True
                if data is not None:
                    self._debug_print_sample(data, idx, split="test")
            return data

        # Train mode
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            # One-time debug for train mode
            if not hasattr(self, "_dbg_once"):
                self._dbg_once = True
                self._debug_print_sample(data, idx, split="train")

            return data

    # ------------------------------------------------------------------
    #   CUSTOM NUSCENES EVALUATION
    # ------------------------------------------------------------------
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol."""
        from nuscenes import NuScenes

        self.nusc = NuScenes(version=self.version,
                             dataroot=self.data_root,
                             verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail[f'{metric_prefix}/{name}_{k}'] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = val

        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Write nuScenes JSON and sanitize it so no None boxes exist."""
        out = super().format_results(results, jsonfile_prefix=jsonfile_prefix, **kwargs)

        # ---- locate json path from super() return ----
        try:
            if isinstance(out, tuple):
                result_files, _ = out
            else:
                result_files = out

            json_path = None
            if isinstance(result_files, dict):
                if isinstance(result_files.get('pts_bbox'), str) and result_files['pts_bbox'].endswith('.json'):
                    json_path = result_files['pts_bbox']
                else:
                    for v in result_files.values():
                        if isinstance(v, str) and v.endswith('.json'):
                            json_path = v
                            break
            elif isinstance(result_files, str) and result_files.endswith('.json'):
                json_path = result_files

            if json_path is None:
                print("[PKG] Could not locate results JSON path from format_results return:", result_files)
                return out

            # ---- SANITIZE: remove any None / malformed detections ----
            data = mmcv.load(json_path)
            res = data.get('results', {})

            removed = 0
            fixed_tokens = 0

            for tok, dets in list(res.items()):
                # dets should be a list; if it isn't, force it
                if dets is None:
                    res[tok] = []
                    fixed_tokens += 1
                    continue
                if not isinstance(dets, (list, tuple)):
                    res[tok] = []
                    fixed_tokens += 1
                    continue

                clean = []
                for d in dets:
                    # drop literal None
                    if d is None:
                        removed += 1
                        continue
                    # must be dict
                    if not isinstance(d, dict):
                        removed += 1
                        continue

                    # minimal nuScenes detection fields that MUST exist
                    # (if any missing, skip to avoid loader creating None box)
                    if ('translation' not in d) or ('size' not in d) or ('rotation' not in d) or ('detection_name' not in d):
                        removed += 1
                        continue

                    # translation/size/rotation must be correct lengths
                    try:
                        if d['translation'] is None or len(d['translation']) != 3:
                            removed += 1
                            continue
                        if d['size'] is None or len(d['size']) != 3:
                            removed += 1
                            continue
                        if d['rotation'] is None or len(d['rotation']) != 4:
                            removed += 1
                            continue
                    except Exception:
                        removed += 1
                        continue

                    clean.append(d)

                if len(clean) != len(dets):
                    res[tok] = clean
                    fixed_tokens += 1

            data['results'] = res
            mmcv.dump(data, json_path)

            print(f"[PKG] sanitize done: removed={removed} fixed_tokens={fixed_tokens} json={json_path}")

            # ---- keep your existing debug preview ----
            printed = 0
            for tok, dets in res.items():
                print(f"[PKG] sample_token={tok}  #preds={len(dets)}")
                for j, d in enumerate(dets[:3]):
                    print(
                        "   ", j,
                        d.get('detection_name', '?'),
                        "trans=",
                        [round(x, 2) for x in d.get('translation', [])],
                        "size=",
                        [round(x, 2) for x in d.get('size', [])],
                        "score=",
                        round(float(d.get('detection_score', 0.0)), 3),
                    )
                printed += 1
                if printed >= 3:
                    break

        except Exception as e:
            print("[PKG] format_results sanitize/debug failed:", repr(e))

        return out

