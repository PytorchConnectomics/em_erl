import os, sys
import numpy as np
import imageio 
import pickle

def mkdir(foldername, opt=""):
    """
    Create a directory.

    Args:
        fn (str): The path of the directory to create.
        opt (str, optional): The options for creating the directory. Defaults to "".

    Returns:
        None

    Raises:
        None
    """
    if opt == "parent":  # until the last /
        foldername = os.path.dirname(foldername)
    if not os.path.exists(foldername):
        if "all" in opt or "parent" in opt:
            os.makedirs(foldername)
        else:
            os.mkdir(foldername)


def read_vol(filename, dataset=None):
    """
    Read data from various file formats.

    Args:
        filename (str): The path to the file.
        dataset (str, optional): The name of the dataset within the file. Defaults to None.

    Returns:
        numpy.ndarray: The read data.

    Raises:
        Exception: If the file format is not supported or if the file cannot be read.
    """
    if filename[-3:] == "npy":
        out = np.load(filename)
    elif filename[-3:] == "pkl":
        out = read_pkl(filename)
    elif filename[-3:] in ["tif", "iff"]:
        out = read_image(filename, data_type="nd")
    elif filename[-2:] == "h5":
        out = read_h5(filename, dataset)
    elif filename[-3:] == "zip":
        out = read_zarr(filename, dataset)
    elif len(filename) > 11 and (filename[:11] == "precomputed" or filename[:3] == "gs:"):
        import cloudvolume
        if filename[:11] != "precomputed":
            filename = f'precomputed://{filename}'
        if '@' in filename:
            filename, chunk_id = filename.split('@')
            chunk_id = int(chunk_id)
        # download cloudvolume data in xyz format 
        out = np.squeeze(cloudvolume.CloudVolume(filename, mip=chunk_id, cache=False)[:])
        # transpose it to zyx
        out = out.transpose(range(out.ndim)[::-1])
    else:
        raise f"Can't read the file type of {filename}"
    return out

def read_image(filename, image_type="image", ratio=None, resize_order=None, data_type="2d", crop=None):
    """
    Read an image from a file.

    Args:
        filename (str): The path to the image file.
        image_type (str, optional): The type of image to read. Defaults to "image".
        ratio (int or list, optional): The scaling ratio for the image. Defaults to None.
        resize_order (int, optional): The order of interpolation for scaling. Defaults to 1.
        data_type (str, optional): The type of image data to read. Defaults to "2d".

    Returns:
        numpy.ndarray: The image data.

    Raises:
        AssertionError: If the ratio dimensions do not match the image dimensions.
    """
    if data_type == "2d":
        # assume the image of the size M x N x C
        image = imageio.imread(filename)
        if image_type == "seg":
            image = rgb_to_seg(image)
        if ratio is not None:
            if str(ratio).isnumeric():
                ratio = [ratio, ratio]
            if ratio[0] != 1:
                if resize_order is None:
                    resize_order = 0 if image_type == "seg" else 1
                if image.ndim == 2:
                    image = zoom(image, ratio, order=resize_order)
                else:
                    # do not zoom the color channel
                    image = zoom(image, ratio + [1], order=resize_order)
        if crop is not None:
            image = image[crop[0]: crop[1], crop[2]: crop[3]]
    else:
        # read in nd volume
        image = imageio.volread(filename)
        if ratio is not None:
            assert (
                str(ratio).isnumeric() or len(ratio) == image.ndim
            ), f"ratio's dim {len(ratio)} is not equal to image's dim {image.ndim}"
            image = zoom(image, ratio, order=resize_order)
        if crop is not None:
            obj = tuple(slice(crop[x*2], crop[x*2+1]) for x in range(image.ndim))
            image = image[obj]
    return image


def read_zarr(filename, dataset=None, chunk_id=0, chunk_num=1):
    """
    Read data from a Zarr file.

    Args:
        filename (str): The path to the Zarr file.
        dataset (str, optional): The name of the dataset within the file. Defaults to None.
        chunk_id (int, optional): The ID of the chunk to read. Defaults to 0.
        chunk_num (int, optional): The total number of chunks. Defaults to 1.

    Returns:
        numpy.ndarray: The read data.

    Raises:
        Exception: If the dataset is not found or if there is an error reading the file.
    """
    import zarr

    fid = zarr.open_group(filename)
    if dataset is None:
        dataset = fid.info_items()[-1][1]
        if "," in dataset:
            dataset = dataset[: dataset.find(",")]
    
    return np.array(fid[dataset])
    
def read_h5(filename, dataset=None):
    """
    Read data from an HDF5 file.

    Args:
        filename (str): The path to the HDF5 file.
        dataset (str or list, optional): The name or names of the dataset(s) to read. Defaults to None.

    Returns:
        numpy.ndarray or list: The data from the HDF5 file.

    """
    import h5py
    fid = h5py.File(filename, "r")
    if dataset is None:
        dataset = fid.keys() if sys.version[0] == "2" else list(fid)
    else:
        if not isinstance(dataset, list):
            dataset = list(dataset)

    out = [None] * len(dataset)
    for di, d in enumerate(dataset):
        out[di] = np.array(fid[d])

    return out[0] if len(out) == 1 else out

def read_pkl(filename):
    """
    The function `read_pkl` reads a pickle file and returns a list of the objects stored in the file.

    :param filename: The filename parameter is a string that represents the name of the file you want to
    read. It should include the file extension, such as ".pkl" for a pickle file
    :return: a list of objects that were read from the pickle file.
    """
    data = []
    with open(filename, "rb") as fid:
        while True:
            try:
                data.append(pickle.load(fid))
            except EOFError:
                break
    if len(data) == 1:
        return data[0]
    return data


def write_pkl(filename, content):
    """
    Write content to a pickle file.

    Args:
        filename (str): The path to the pickle file.
        content (object or list): The content to write. If a list, each element will be pickled separately.

    Returns:
        None

    """
    with open(filename, "wb") as f:
        if isinstance(content, (list,)):
            for val in content:
                pickle.dump(val, f)
        else:
            pickle.dump(content, f)

def write_h5(filename, data, dataset="main"):
    """
    Write data to an HDF5 file.

    Args:
        filename (str): The path to the HDF5 file.
        data (numpy.ndarray or list): The data to write.
        dataset (str or list, optional): The name or names of the dataset(s) to create. Defaults to "main".

    Returns:
        None

    Raises:
        None
    """
    fid = h5py.File(filename, "w")
    if isinstance(data, (list,)):
        if not isinstance(dataset, (list,)):
            num_digit = int(np.floor(np.log10(len(data)))) + 1
            dataset = [('key%0'+str(num_digit)+'d')%x for x in range(len(data))]
        for i, dd in enumerate(dataset):
            ds = fid.create_dataset(
                dd,
                data[i].shape,
                compression="gzip",
                dtype=data[i].dtype,
            )
            ds[:] = data[i]
    else:
        ds = fid.create_dataset(
            dataset, data.shape, compression="gzip", dtype=data.dtype
        )
        ds[:] = data
    fid.close()
