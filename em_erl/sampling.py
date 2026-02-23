from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from .io import read_vol


def _resolve_dataset_name_h5(fid, dataset):
    if dataset is not None:
        return dataset
    keys = list(fid.keys())
    if not keys:
        raise ValueError("HDF5 file has no datasets")
    return keys[0]


def _resolve_dataset_name_zarr(group, dataset):
    if dataset is not None:
        return dataset
    keys = list(group.keys())
    if not keys:
        raise ValueError("Zarr group has no arrays")
    return keys[0]


class VolumeSource:
    """Random-access 3D volume interface in zyx order."""

    shape: tuple[int, ...]
    dtype: np.dtype

    def read_slab(self, z0: int, z1: int) -> np.ndarray:
        raise NotImplementedError

    def sample_points(self, points_zyx: np.ndarray) -> np.ndarray:
        if points_zyx.size == 0:
            return np.zeros(0, dtype=self.dtype)
        points_zyx = np.asarray(points_zyx)
        z0 = int(points_zyx[:, 0].min())
        z1 = int(points_zyx[:, 0].max()) + 1
        slab = self.read_slab(z0, z1)
        return slab[
            points_zyx[:, 0] - z0,
            points_zyx[:, 1],
            points_zyx[:, 2],
        ]

    def read_full(self) -> np.ndarray:
        return self.read_slab(0, int(self.shape[0]))

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class ArrayVolumeSource(VolumeSource):
    def __init__(self, array):
        self._array = np.asarray(array)
        if self._array.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {self._array.shape}")
        self.shape = self._array.shape
        self.dtype = self._array.dtype

    def read_slab(self, z0, z1):
        return self._array[z0:z1]


class H5VolumeSource(VolumeSource):
    def __init__(self, path, dataset=None):
        import h5py

        self._h5py = h5py
        self._fid = h5py.File(path, "r")
        self._dataset_name = _resolve_dataset_name_h5(self._fid, dataset)
        self._ds = self._fid[self._dataset_name]
        if self._ds.ndim != 3:
            self._fid.close()
            raise ValueError(f"Expected 3D HDF5 dataset, got shape {self._ds.shape}")
        self.shape = tuple(self._ds.shape)
        self.dtype = self._ds.dtype

    def read_slab(self, z0, z1):
        return np.asarray(self._ds[z0:z1])

    def close(self):
        self._fid.close()


class ZarrVolumeSource(VolumeSource):
    def __init__(self, path, dataset=None):
        import zarr

        self._group = zarr.open_group(path)
        self._dataset_name = _resolve_dataset_name_zarr(self._group, dataset)
        self._ds = self._group[self._dataset_name]
        if self._ds.ndim != 3:
            raise ValueError(f"Expected 3D Zarr array, got shape {self._ds.shape}")
        self.shape = tuple(self._ds.shape)
        self.dtype = self._ds.dtype

    def read_slab(self, z0, z1):
        return np.asarray(self._ds[z0:z1])


def open_volume_source(volume, dataset=None) -> VolumeSource:
    if isinstance(volume, VolumeSource):
        return volume
    if not isinstance(volume, str):
        return ArrayVolumeSource(volume)

    if volume.endswith(".h5"):
        return H5VolumeSource(volume, dataset=dataset)
    if volume.endswith(".zip"):
        return ZarrVolumeSource(volume, dataset=dataset)

    # Fallback for formats that are only supported via read_vol (npy, tif, cloudvolume, etc.).
    return ArrayVolumeSource(read_vol(volume, dataset))


def iter_z_chunks(total_z: int, chunk_num: int):
    if chunk_num <= 1:
        yield 0, total_z
        return
    for chunk_id in range(chunk_num):
        z0 = (chunk_id * total_z) // chunk_num
        z1 = ((chunk_id + 1) * total_z) // chunk_num
        if z1 > z0:
            yield z0, z1


def sample_segment_lut_from_sources(
    segment_source: VolumeSource,
    node_position_zyx,
    mask_source: VolumeSource | None = None,
    chunk_num: int = 1,
    data_type=np.uint32,
):
    node_position = np.asarray(node_position_zyx, dtype=np.int64)
    if node_position.ndim != 2 or node_position.shape[1] != 3:
        raise ValueError("node_position must have shape [N, 3]")
    if len(node_position) == 0:
        empty_mask = None if mask_source is None else np.zeros(0, dtype=segment_source.dtype)
        return np.zeros(0, dtype=data_type), empty_mask

    node_lut = np.zeros(len(node_position), dtype=data_type)
    order = np.argsort(node_position[:, 0], kind="mergesort")
    z_sorted = node_position[order, 0]
    mask_segment_chunks = [] if mask_source is not None else None

    total_z = int(segment_source.shape[0])
    for z0, z1 in iter_z_chunks(total_z, chunk_num):
        seg_slab = segment_source.read_slab(z0, z1)

        lo = int(np.searchsorted(z_sorted, z0, side="left"))
        hi = int(np.searchsorted(z_sorted, z1, side="left"))
        if hi > lo:
            idx = order[lo:hi]
            pts = node_position[idx]
            node_lut[idx] = seg_slab[
                pts[:, 0] - z0,
                pts[:, 1],
                pts[:, 2],
            ]

        if mask_source is not None:
            mask_slab = mask_source.read_slab(z0, z1)
            if mask_slab.shape != seg_slab.shape:
                raise ValueError(
                    f"Mask slab shape {mask_slab.shape} does not match segmentation slab shape {seg_slab.shape}"
                )
            if np.any(mask_slab):
                vals = seg_slab[mask_slab > 0]
                if vals.size > 0:
                    mask_segment_chunks.append(vals)

    if mask_source is None:
        return node_lut, None

    if mask_segment_chunks:
        mask_segment_id = np.concatenate(mask_segment_chunks)
        used_segment_ids = np.unique(node_lut)
        mask_segment_id = mask_segment_id[np.isin(mask_segment_id, used_segment_ids)]
    else:
        mask_segment_id = np.zeros(0, dtype=segment_source.dtype)
    return node_lut, mask_segment_id


def sample_segment_lut(
    segment,
    node_position_zyx,
    mask=None,
    chunk_num: int = 1,
    data_type=np.uint32,
    segment_dataset=None,
    mask_dataset=None,
):
    seg_ctx = open_volume_source(segment, dataset=segment_dataset)
    mask_ctx = nullcontext(None) if mask is None else open_volume_source(mask, dataset=mask_dataset)
    with seg_ctx as seg_source, mask_ctx as mask_source:
        return sample_segment_lut_from_sources(
            seg_source,
            node_position_zyx=node_position_zyx,
            mask_source=mask_source,
            chunk_num=chunk_num,
            data_type=data_type,
        )
