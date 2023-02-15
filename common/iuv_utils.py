# SM: code was taken basically from re-timing project
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def load_and_process_iuv(iuv_path,width=480) :
    """Read IUV file and convert to network inputs."""
    iuv_map = Image.open(iuv_path).convert('RGBA')
    iuv_map = transforms.ToTensor()(iuv_map)
    uv_map, mask, pids = iuv2input(iuv_map,width=width)
    return uv_map, mask, pids


def iuv2input(iuv, n_textures = 49,width=480):
    """Create network inputs from IUV.
    Parameters:
        iuv - - a tensor of shape [4, H, W], where the channels are: body part ID, U, V, person ID.
        index - - index of iuv

    Returns:
        uv (tensor) - - a UV map for a single layer, ready to pass to grid sampler (values in range [-1,1])
        mask (tensor) - - the corresponding mask
        person_id (tensor) - - the person IDs

    grid sampler indexes into texture map of size tile_width x tile_width*n_textures
    """
    # Extract body part and person IDs.
    part_id = (iuv[0] * 255 / 10).round()
    part_id[part_id > 24] = 24
    part_id_mask = (part_id > 0).float()
    person_id = (255 - 255 * iuv[-1]).round()  # person ID is saved as 255 - person_id
    person_id *= part_id_mask  # background id is 0
    maxId = n_textures // 24
    person_id[person_id>maxId] = maxId

    # Convert body part ID to texture map ID.
    # Essentially, each of the 24 body parts for each person, plus the background have their own texture 'tile'
    # The tiles are concatenated horizontally to create the texture map.
    tex_id = part_id + part_id_mask * 24 * (person_id - 1)

    uv = iuv[1:3]
    # Convert the per-body-part UVs to UVs that correspond to the full texture map.
    uv[0] += tex_id

    # Get the mask.
    bg_mask = (tex_id == 0).float()
    mask = 1.0 - bg_mask
    mask = mask * 2 - 1  # make 1 the foreground and -1 the background mask
    mask = mask2trimap(mask,width = width)

    # Composite background UV behind person UV.
    h, w = iuv.shape[1:]
    bg_uv = get_background_uv(w, h)
    uv = bg_mask * bg_uv + (1 - bg_mask) * uv

    # Map to [-1, 1] range.
    uv[0] /= n_textures
    uv = uv * 2 - 1
    uv = torch.clamp(uv, -1, 1)

    return uv, mask, person_id

def mask2trimap(mask,trimap_width = 10,width = 480):
    """Convert binary mask to trimap with values in [-1, 0, 1]."""
    fg_mask = (mask > 0).float()
    bg_mask = (mask < 0).float()
    trimap_width = trimap_width
    trimap_width *= bg_mask.shape[-1] / width
    trimap_width = int(trimap_width)
    bg_mask = cv2.erode(bg_mask.numpy(), kernel=np.ones((trimap_width, trimap_width)), iterations=1)
    bg_mask = torch.from_numpy(bg_mask)
    mask = fg_mask - bg_mask
    return mask

def get_background_uv(w, h):
    """Return background layer UVs at 'index' (output range [0, 1])."""
    ramp_u = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
    ramp_v = torch.linspace(0, 1, steps=h).unsqueeze(-1).repeat(1, w)
    ramp = torch.stack([ramp_u, ramp_v], 0)

    return ramp
