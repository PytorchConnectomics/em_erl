import os
import pickle
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def mkdir(foldername, opt=""):
    """Create a directory or a parent directory."""
    folder = Path(foldername)
    if opt == "parent":
        folder = folder.parent
    if str(folder) in ("", "."):
        return
    if "all" in opt or "parent" in opt:
        folder.mkdir(parents=True, exist_ok=True)
    elif not folder.exists():
        folder.mkdir()


def _as_path_str(filename):
    return str(filename)


def _suffix_lower(filename):
    return Path(filename).suffix.lower()


def _is_cloudvolume_path(filename):
    return filename.startswith("precomputed") or filename.startswith("gs:")


def read_vol(filename, dataset=None):
    """Read a volume/array object from supported storage formats."""
    filename = _as_path_str(filename)
    suffix = _suffix_lower(filename)

    if suffix == ".npy":
        return np.load(filename)
    if suffix == ".pkl":
        return read_pkl(filename)
    if suffix in {".tif", ".tiff"}:
        return read_image(filename, data_type="nd")
    if suffix in {".h5", ".hdf5"}:
        return read_h5(filename, dataset)
    if suffix in {".zip", ".zarr"}:
        return read_zarr(filename, dataset)
    if _is_cloudvolume_path(filename):
        return _read_cloudvolume(filename)
    raise ValueError(f"Can't read the file type of {filename}")


def _read_cloudvolume(filename):
    import cloudvolume

    chunk_id = 0
    if not filename.startswith("precomputed"):
        filename = f"precomputed://{filename}"
    if "@" in filename:
        filename, chunk_id_str = filename.rsplit("@", 1)
        chunk_id = int(chunk_id_str)
    out = np.squeeze(cloudvolume.CloudVolume(filename, mip=chunk_id, cache=False)[:])
    # CloudVolume returns xyz; convert to zyx.
    return out.transpose(range(out.ndim)[::-1])


def rgb_to_seg(image):
    """Pack RGB channels into a single uint32 label image."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image.shape}")
    rgb = image[..., :3].astype(np.uint32, copy=False)
    return rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)


def _normalize_ratio(ratio, ndim):
    if ratio is None:
        return None
    if np.isscalar(ratio):
        return [ratio] * ndim
    ratio = list(ratio)
    if len(ratio) != ndim:
        raise AssertionError(
            f"ratio dim {len(ratio)} is not equal to image dim {ndim}"
        )
    return ratio


def read_image(
    filename,
    image_type="image",
    ratio=None,
    resize_order=None,
    data_type="2d",
    crop=None,
):
    """Read 2D images or N-D volumes from image files."""
    from scipy.ndimage import zoom

    if data_type == "2d":
        image = imageio.imread(filename)
        if image_type == "seg":
            image = rgb_to_seg(image)
        if ratio is not None:
            ratio_2d = _normalize_ratio(ratio, 2)
            if ratio_2d[0] != 1 or ratio_2d[1] != 1:
                if resize_order is None:
                    resize_order = 0 if image_type == "seg" else 1
                if image.ndim == 2:
                    image = zoom(image, ratio_2d, order=resize_order)
                else:
                    image = zoom(image, ratio_2d + [1], order=resize_order)
        if crop is not None:
            image = image[crop[0] : crop[1], crop[2] : crop[3]]
        return image

    image = imageio.volread(filename)
    if ratio is not None:
        ratio_nd = _normalize_ratio(ratio, image.ndim)
        image = zoom(image, ratio_nd, order=resize_order)
    if crop is not None:
        obj = tuple(slice(crop[x * 2], crop[x * 2 + 1]) for x in range(image.ndim))
        image = image[obj]
    return image


def _resolve_zarr_dataset(group, dataset=None):
    if dataset is not None:
        return dataset
    keys = list(group.keys())
    if not keys:
        raise ValueError("No dataset found in zarr group")
    return keys[0]


def read_zarr(filename, dataset=None, chunk_id=0, chunk_num=1):
    """Read an array from a zarr group."""
    del chunk_id, chunk_num  # kept for API compatibility with historical callers
    import zarr

    fid = zarr.open_group(filename)
    dataset = _resolve_zarr_dataset(fid, dataset)
    return np.asarray(fid[dataset])


def _normalize_dataset_arg(dataset):
    if dataset is None:
        return None, False
    if isinstance(dataset, (list, tuple)):
        return list(dataset), True
    return [dataset], False


def read_h5(filename, dataset=None):
    """Read one or more datasets from an HDF5 file."""
    import h5py

    req_datasets, explicit_list = _normalize_dataset_arg(dataset)
    with h5py.File(filename, "r") as fid:
        if req_datasets is None:
            req_datasets = list(fid.keys())
            explicit_list = len(req_datasets) != 1
        if not req_datasets:
            raise ValueError(f"No datasets found in {filename}")
        out = [np.asarray(fid[d]) for d in req_datasets]
    if explicit_list:
        return out
    return out[0]


def read_pkl(filename):
    """Read one or more python objects from a pickle file."""
    data = []
    with open(filename, "rb") as fid:
        while True:
            try:
                data.append(pickle.load(fid))
            except EOFError:
                break
    return data[0] if len(data) == 1 else data


def write_pkl(filename, content):
    """Write a python object or list of objects to a pickle file."""
    with open(filename, "wb") as f:
        if isinstance(content, list):
            for val in content:
                pickle.dump(val, f)
        else:
            pickle.dump(content, f)


def _default_h5_dataset_names(num_items):
    if num_items <= 0:
        return []
    digits = max(1, int(np.floor(np.log10(num_items))) + 1)
    return [f"key{i:0{digits}d}" for i in range(num_items)]


def write_h5(filename, data, dataset="main"):
    """Write one array or multiple arrays to an HDF5 file with gzip compression."""
    import h5py

    mkdir(filename, "parent")
    with h5py.File(filename, "w") as fid:
        if isinstance(data, list):
            arrays = [np.asarray(x) for x in data]
            if isinstance(dataset, (list, tuple)):
                dataset_names = list(dataset)
            else:
                dataset_names = _default_h5_dataset_names(len(arrays))
            if len(dataset_names) != len(arrays):
                raise ValueError("dataset names and data length must match")
            for name, arr in zip(dataset_names, arrays):
                ds = fid.create_dataset(
                    name,
                    arr.shape,
                    compression="gzip",
                    dtype=arr.dtype,
                )
                ds[...] = arr
            return

        arr = np.asarray(data)
        ds = fid.create_dataset(
            dataset,
            arr.shape,
            compression="gzip",
            dtype=arr.dtype,
        )
        ds[...] = arr
