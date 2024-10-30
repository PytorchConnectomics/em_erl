import numpy as np

def vol_to_skel(labels, scale=1.5, const=500, obj_ids=None, dust_size = 100, res = (32,32,30), num_thread = 1):
    import kimimaro
    # res: xyz
    if obj_ids is None:
        obj_ids = np.unique(labels)
        obj_ids = obj_ids[obj_ids>0]
    obj_ids = list(obj_ids)
    skels = kimimaro.skeletonize(
      labels,
      teasar_params={
        'scale': scale,
        'const': const, # physical units
        'pdrf_exponent': 4,
        'pdrf_scale': 100000,
        'soma_detection_threshold': 1100, # physical units
        'soma_acceptance_threshold': 3500, # physical units
        'soma_invalidation_scale': 1.0,
        'soma_invalidation_const': 300, # physical units
        'max_paths': 50, # default  None
      },
      object_ids= obj_ids, # process only the specified labels
      # object_ids=[ ... ], # process only the specified labels
      # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
      # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
      dust_threshold = dust_size, # skip connected components with fewer than this many voxels
      anisotropy = res, # default True
      fix_branching=True, # default True
      fix_borders=True, # default True
      progress=True, # default False, show progress bar
      parallel = num_thread, # <= 0 all cpu, 1 single process, 2+ multiprocess
      parallel_chunk_size = 100, # how many skeletons to process before updating progress bar
    )
    return skels

def cable_length(vertices, edges, res = [1,1,1]):
    """
    Returns cable length of connected skeleton vertices in the same
    metric that this volume uses (typically nanometers).
    """
    if len(edges) == 0:
        return 0
    v1 = vertices[edges[:,0]]
    v2 = vertices[edges[:,1]]

    delta = ((v2 - v1) * res) ** 2
    dist = np.sum(delta, axis=1)
    dist = np.sqrt(dist)
    return np.sum(dist)

def skel_to_length(skels, res=[1,1,1]):
    lid = np.fromiter(skels.keys(), dtype=int)
    l0 = np.array([cable_length(skels[x].vertices, skels[x].edges, res) for x in lid])
    return np.vstack([lid,l0]).T
