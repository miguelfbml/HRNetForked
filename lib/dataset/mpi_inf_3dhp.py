# ------------------------------------------------------------------------------
# MPI-INF-3DHP dataset loader for HRNet
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import logging
import os
from collections import OrderedDict

import numpy as np

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class MPIINF3DHPDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 17
        self.flip_pairs = [[2, 5], [3, 6], [4, 7], [8, 11], [9, 12], [10, 13]]
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 15, 16)
        self.lower_body_ids = (8, 9, 10, 11, 12, 13, 14)

        self.aspect_ratio = self.image_size[0] * 1.0 / self.image_size[1]

        self.train_annotation_file = cfg.DATASET.TRAIN_ANNOTATION_FILE
        self.test_annotation_file = cfg.DATASET.TEST_ANNOTATION_FILE
        self.train_image_root = cfg.DATASET.TRAIN_IMAGE_ROOT if cfg.DATASET.TRAIN_IMAGE_ROOT else root
        self.test_image_root = (
            cfg.DATASET.TEST_IMAGE_ROOT
            if cfg.DATASET.TEST_IMAGE_ROOT
            else os.path.join(self.train_image_root, 'mpi_inf_3dhp_test_set')
        )

        self.train_frame_stride = max(1, int(os.environ.get('MPI3DHP_TRAIN_FRAME_STRIDE', '1')))
        self.test_frame_stride = max(1, int(os.environ.get('MPI3DHP_TEST_FRAME_STRIDE', '1')))
        self.max_train_samples = max(0, int(os.environ.get('MPI3DHP_MAX_TRAIN_SAMPLES', '0')))
        self.max_test_samples = max(0, int(os.environ.get('MPI3DHP_MAX_TEST_SAMPLES', '0')))

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info(
            '=> mpi_inf_3dhp sampling: train_stride=%d test_stride=%d max_train=%d max_test=%d',
            self.train_frame_stride,
            self.test_frame_stride,
            self.max_train_samples,
            self.max_test_samples,
        )
        logger.info('=> load {} samples'.format(len(self.db)))

    def _resolve_file(self, path_value):
        if not path_value:
            return ''
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(os.getcwd(), path_value)

    def _compute_center_scale(self, joints_xy):
        x_min = np.min(joints_xy[:, 0])
        y_min = np.min(joints_xy[:, 1])
        x_max = np.max(joints_xy[:, 0])
        y_max = np.max(joints_xy[:, 1])

        w = x_max - x_min
        h = y_max - y_min
        if w < 2 or h < 2:
            return None, None

        center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)

        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array([w / self.pixel_std, h / self.pixel_std], dtype=np.float32)
        scale = scale * 1.25

        return center, scale

    def _build_train_db(self):
        train_file = self._resolve_file(self.train_annotation_file)
        if not os.path.exists(train_file):
            raise FileNotFoundError('Training annotation file not found: {}'.format(train_file))

        annotations = np.load(train_file, allow_pickle=True)['data'].item()
        db = []

        for seq_name, seq_data in annotations.items():
            if not isinstance(seq_data, list) or len(seq_data) < 1:
                continue

            parts = seq_name.split(' ')
            if len(parts) != 2:
                continue
            subject, sequence = parts
            camera_dict = seq_data[0]

            camera_keys = sorted(
                [k for k in camera_dict.keys() if isinstance(camera_dict[k], dict) and 'data_2d' in camera_dict[k]],
                key=lambda x: int(x) if str(x).isdigit() else str(x)
            )

            for cam_key in camera_keys:
                cam_data = camera_dict[cam_key]
                poses_2d = cam_data['data_2d']

                image_folder = os.path.join(self.train_image_root, subject, sequence, 'imageFrames', 'video_{}'.format(cam_key))
                if not os.path.exists(image_folder):
                    continue

                image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
                image_files.extend(glob.glob(os.path.join(image_folder, '*.JPG')))
                image_files.sort()

                if not image_files:
                    continue

                max_frames = min(len(image_files), len(poses_2d))
                for frame_idx in range(0, max_frames, self.train_frame_stride):
                    joints_xy = poses_2d[frame_idx].astype(np.float32)
                    if joints_xy.shape[0] != self.num_joints:
                        continue

                    center, scale = self._compute_center_scale(joints_xy)
                    if center is None:
                        continue

                    joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
                    joints_3d_vis = np.ones((self.num_joints, 3), dtype=np.float32)
                    joints_3d[:, 0:2] = joints_xy[:, 0:2]

                    db.append(
                        {
                            'image': image_files[frame_idx],
                            'center': center,
                            'scale': scale,
                            'joints_3d': joints_3d,
                            'joints_3d_vis': joints_3d_vis,
                            'filename': '{}_{}_cam{}_frame{:06d}'.format(subject, sequence, cam_key, frame_idx),
                            'imgnum': frame_idx,
                            'score': 1.0,
                        }
                    )

                    if self.max_train_samples > 0 and len(db) >= self.max_train_samples:
                        logger.info('=> reached MPI3DHP_MAX_TRAIN_SAMPLES=%d', self.max_train_samples)
                        return db

        return db

    def _build_test_db(self):
        test_file = self._resolve_file(self.test_annotation_file)
        if not os.path.exists(test_file):
            logger.warning('Test annotation file not found: %s', test_file)
            return []

        annotations = np.load(test_file, allow_pickle=True)['data'].item()
        db = []

        for seq_name, seq_data in annotations.items():
            if not isinstance(seq_data, dict) or 'data_2d' not in seq_data:
                continue

            image_folder = os.path.join(self.test_image_root, seq_name, 'imageSequence')
            if not os.path.exists(image_folder):
                continue

            image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
            image_files.extend(glob.glob(os.path.join(image_folder, '*.png')))
            image_files.sort()
            if not image_files:
                continue

            poses_2d = seq_data['data_2d']
            max_frames = min(len(image_files), len(poses_2d))

            for frame_idx in range(0, max_frames, self.test_frame_stride):
                joints_xy = poses_2d[frame_idx].astype(np.float32)
                if joints_xy.shape[0] != self.num_joints:
                    continue

                center, scale = self._compute_center_scale(joints_xy)
                if center is None:
                    continue

                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
                joints_3d_vis = np.ones((self.num_joints, 3), dtype=np.float32)
                joints_3d[:, 0:2] = joints_xy[:, 0:2]

                db.append(
                    {
                        'image': image_files[frame_idx],
                        'center': center,
                        'scale': scale,
                        'joints_3d': joints_3d,
                        'joints_3d_vis': joints_3d_vis,
                        'filename': '{}_frame{:06d}'.format(seq_name, frame_idx),
                        'imgnum': frame_idx,
                        'score': 1.0,
                    }
                )

                if self.max_test_samples > 0 and len(db) >= self.max_test_samples:
                    logger.info('=> reached MPI3DHP_MAX_TEST_SAMPLES=%d', self.max_test_samples)
                    return db

        return db

    def _get_db(self):
        if self.is_train:
            return self._build_train_db()
        return self._build_test_db()

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        if len(self.db) == 0:
            name_value = OrderedDict([('PCK@0.05', 0.0), ('MPJPE2D', 0.0)])
            return name_value, name_value['PCK@0.05']

        n = min(len(preds), len(self.db))
        total_dist = 0.0
        total_joints = 0
        pck_hits = 0

        for i in range(n):
            gt = self.db[i]['joints_3d'][:, :2]
            pred = preds[i, :, :2]

            d = np.linalg.norm(pred - gt, axis=1)
            total_dist += np.sum(d)
            total_joints += len(d)

            x_min = np.min(gt[:, 0])
            x_max = np.max(gt[:, 0])
            y_min = np.min(gt[:, 1])
            y_max = np.max(gt[:, 1])
            scale_ref = max(x_max - x_min, y_max - y_min)
            thr = max(1.0, 0.05 * scale_ref)

            pck_hits += np.sum(d <= thr)

        mpjpe_2d = float(total_dist / max(1, total_joints))
        pck_005 = float(pck_hits / max(1, total_joints))

        name_value = OrderedDict([
            ('PCK@0.05', pck_005),
            ('MPJPE2D', mpjpe_2d),
        ])

        return name_value, name_value['PCK@0.05']
