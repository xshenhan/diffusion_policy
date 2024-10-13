from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()
from scipy.spatial.transform import Rotation as R

class FrankaIsaacSimEpisodicDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()
        replay_buffer = self._convert_dataset_to_replay(
            store=zarr.MemoryStore(), 
            shape_meta=shape_meta, 
            dataset_path=dataset_path, )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        this_normalizer = get_range_normalizer_from_stat(stat)
        if self.use_legacy_normalizer:
            this_normalizer = normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer
        
        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data

    def _convert_dataset_to_replay(self, store, shape_meta, dataset_path, n_workers=None, max_inflight_tasks=None):
        if n_workers is None:
            n_workers = multiprocessing.cpu_count()
        if max_inflight_tasks is None:
            max_inflight_tasks = n_workers * 5

        # parse shape_meta
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            value_type = attr.get('type', 'low_dim')
            if value_type == 'rgb':
                rgb_keys.append(key)
            elif value_type == 'low_dim':
                lowdim_keys.append(key)
        
        root = zarr.group(store)
        data_group = root.require_group('data', overwrite=True)
        meta_group = root.require_group('meta', overwrite=True)

        episode_ends = list()
        prev_end = 0
        for path in Path(dataset_path).glob('*.hdf5'):
            with h5py.File(path) as file:
                episode_length = file['action'].shape[0]
                episode_end = prev_end + episode_length
                prev_end = episode_end
                episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)
        # save lowdim data
        datas = {key: [] for key in lowdim_keys}
        actions = []
        for path in Path(dataset_path).glob('*.hdf5'):
            with h5py.File(path) as file:
                for key in lowdim_keys:
                    data_key = f"observations/{key}"
                    datas[key].append(file[data_key][()].astype(np.float32))
                actions.append(file['action'][:].astype(np.float32))
        for key in lowdim_keys:
            datas[key] = np.concatenate(datas[key], axis=0)
            assert datas[key].shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=datas[key],
                shape=datas[key].shape,
                chunks=datas[key].shape,
                compressor=None,
                dtype=datas[key].dtype
            )
        actions = np.concatenate(actions, axis=0)
        assert actions.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
        _ = data_group.array(
            name='action',
            data=actions,
            shape=actions.shape,
            chunks=actions.shape,
            compressor=None,
            dtype=actions.dtype
        )
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        episodes = [h5py.File(file) for file in Path(dataset_path).glob('*.hdf5')]
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = f"observations/images"
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(episodes)):
                        demo = episodes[episode_idx]
                        hdf5_arr = demo[data_key][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))
                
        replay_buffer = ReplayBuffer(root)
        return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )


class FrankaIsaacSimRealRobotDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            relative_control=True,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()
        replay_buffer = self._convert_dataset_to_replay(
            store=zarr.MemoryStore(), 
            shape_meta=shape_meta,
            relative_control=relative_control,
            dataset_path=dataset_path, )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        this_normalizer = get_range_normalizer_from_stat(stat)
        if self.use_legacy_normalizer:
            this_normalizer = normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer
        
        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('euler'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data
    
    
    def _convert_dataset_to_replay(self, store, shape_meta, dataset_path, relative_control=True, n_workers=None, max_inflight_tasks=None):
        if n_workers is None:
            n_workers = multiprocessing.cpu_count()
        if max_inflight_tasks is None:
            max_inflight_tasks = n_workers * 5

        # parse shape_meta
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            value_type = attr.get('type', 'low_dim')
            if value_type == 'rgb':
                rgb_keys.append(key)
            elif value_type == 'low_dim':
                lowdim_keys.append(key)
        
        root = zarr.group(store)
        data_group = root.require_group('data', overwrite=True)
        meta_group = root.require_group('meta', overwrite=True)

        episode_ends = list()
        prev_end = 0
        for path in Path(dataset_path).glob('*.hdf5'):
            with h5py.File(path) as file:
                episode_length = file['action'].shape[0]
                episode_end = prev_end + episode_length - 1
                prev_end = episode_end
                episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)
        # save lowdim data
        actions = []
        obses = []
        for path in Path(dataset_path).glob('*.hdf5'):
            with h5py.File(path) as file:
                ee_pose = file["observations/ee_pose"][()].astype(np.float32)
                qpos = file["observations/qpos"][()].astype(np.float32)
                ee_translation = ee_pose[:, :3, 3]
                ee_euler = R.from_matrix(ee_pose[:, :3, :3]).as_euler('xyz', degrees=False)
                raw_traj = np.concatenate([qpos, ee_translation, ee_euler], axis=-1)
                obs = raw_traj[:-1]
                action = raw_traj[1:, -8:]
                if relative_control:
                    action[:, -6:] = action[:, -6:] - obs[:, -6:]
                actions.append(action.astype(np.float32))
                obses.append(obs.astype(np.float32))

        actions = np.concatenate(actions, axis=0)
        obses = np.concatenate(obses, axis=0)
        qposes = obses[:, :9]
        poses = obses[:, 9:12]
        eulers = obses[:, 12:15]
        assert actions.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
        _ = data_group.array(
            name='action',
            data=actions,
            shape=actions.shape,
            chunks=actions.shape,
            compressor=None,
            dtype=actions.dtype
        )
        _ = data_group.array(
            name='qpos',
            data=qposes,
            shape=qposes.shape,
            chunks=qposes.shape,
            compressor=None,
            dtype=qposes.dtype
        )
        _ = data_group.array(
            name='pos',
            data=poses,
            shape=poses.shape,
            chunks=poses.shape,
            compressor=None,
            dtype=poses.dtype
        )
        _ = data_group.array(
            name='euler',
            data=eulers,
            shape=eulers.shape,
            chunks=eulers.shape,
            compressor=None,
            dtype=eulers.dtype
        )
        
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        episodes = [h5py.File(file) for file in Path(dataset_path).glob('*.hdf5')]
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = f"observations/images"
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(episodes)):
                        demo = episodes[episode_idx]
                        hdf5_arr = demo[data_key][key]
                        for hdf5_idx in range(hdf5_arr.shape[0] - 1):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))
                
        replay_buffer = ReplayBuffer(root)
        return replay_buffer
