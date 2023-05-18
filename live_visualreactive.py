import os
from typing import List, Union, Optional, Tuple, Type
import click
import time

import dnnlib
from torch_utils import gen_utils

import numpy as np

import cv2
import imutils
import PIL.Image
import scipy
import torch
from torchvision import transforms

import legacy

from network_features import VGG16FeaturesNVIDIA


# ----------------------------------------------------------------------------


def parse_height(s: str = None) -> Union[int, Type[None]]:
    """Parse height argument."""
    if s is not None:
        if s == 'max':
            return s
        else:
            return int(s)
    return None


# ----------------------------------------------------------------------------


# TODO: Analyze latent space/variant to the proposed PCA https://openreview.net/pdf?id=SlzEll3EsKv
# TODO: Add hand tracking/normalization here: https://github.com/caillonantoine/hand_osc/blob/master/detect.py

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename: can be URL, local file, or the name of the model in torch_utils.gen_utils.resume_specs', required=True)
@click.option('--device', help='Device to use for image generation; using the CPU is slower than the GPU', type=click.Choice(['cpu', 'cuda']), default='cuda', show_default=True)
@click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
# Synthesis options (feed a list of seeds or give the projected w to synthesize)
@click.option('--seed', type=click.INT, help='Random seed to use for static synthesized image', default=0, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--new-center', type=gen_utils.parse_new_center, help='New center for the W latent space; a seed (int) or a path to a projected dlatent (.npy/.npz)', default=None)
@click.option('--mirror', is_flag=True, help='Mirror the synthesized image')
@click.option('--demo-height', type=int, help='Height of the demo window', default=360, show_default=True)
@click.option('--demo-width', type=int, help='Width of the demo window', default=None, show_default=True)
@click.option('--only-synth', is_flag=True, help='Only synthesize the image and save it to disk')
@click.option('--layer', type=str, help='Layer to use for the feature extractor', default='conv4_1', show_default=True)
# How to set the fake dlatent
@click.option('--v0', is_flag=True, help='Average the features of VGG and use a static dlatent to do style-mixing')
@click.option('--v1', is_flag=True, help='Separate the input image into regions for coarse, middle, and fine layers for style-mixing')
# TODO: intermediate layers?
# Video options
@click.option('--display-height', type=parse_height, help="Height of the display window; if 'max', will use G.img_resolution", default=None, show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
@click.option('--fps', type=click.IntRange(min=1), help='Save the video with this framerate.', default=30, show_default=True)
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file with `ffmpeg-python` (same resolution, lower file size)')
# Extra parameters
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'videos'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='live_visual-reactive', show_default=True)
@click.option('--verbose', is_flag=True, help='Print FPS of the live interpolation ever second')
def live_visual_reactive(
        ctx,
        network_pkl: str,
        device: Optional[str],
        cfg: str,
        seed: int,
        truncation_psi: float,
        class_idx: int,
        noise_mode: str,
        new_center: Union[int, str],
        mirror: bool,
        demo_height: int,
        demo_width: int,
        only_synth: bool,
        layer: str,
        v0: bool,
        v1: bool,
        display_height: Optional[int],
        anchor_latent_space: bool,
        fps: int,
        compress: bool,
        outdir: str,
        description: str,
        verbose: Optional[bool]):
    """Live Visual-Reactive interpolation. A camera/webcamera is needed to be accessed by OpenCV."""
    # Set device; GPU is recommended
    device = torch.device('cuda') if torch.cuda.is_available() and device == 'cuda' else torch.device('cpu')
    # Load the feature extractor; here, VGG16
    print('Loading VGG16 and its features...')
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    vgg16_features = VGG16FeaturesNVIDIA(vgg16).requires_grad_(False).to(device)
    del vgg16

    # If model name exists in the gen_utils.resume_specs dictionary, use it instead of the full url
    try:
        network_pkl = gen_utils.resume_specs[cfg][network_pkl]
    except KeyError:
        # Otherwise, it's a local file or an url
        pass

    print('Loading Generator...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False).to(device)  # type: ignore

    # Stabilize/anchor the latent space
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    # Warm up the Generator
    ws = G.mapping(z=torch.randn(1, 512, device=device), c=None, truncation_psi=1.0)
    _ = G.synthesis(ws[:1])

    # Label, in case it's a class-conditional model
    class_idx = gen_utils.parse_class(G, class_idx, ctx)
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Recenter the latent space, if specified
    if new_center is None:
        w_avg = G.mapping.w_avg
    else:
        new_center, new_center_value = new_center
        # We get the new center using the int (a seed) or recovered dlatent (an np.ndarray)
        if isinstance(new_center_value, int):
            w_avg = gen_utils.get_w_from_seed(G, device, new_center_value,
                                              truncation_psi=1.0)  # We want the pure dlatent
        elif isinstance(new_center_value, np.ndarray):
            w_avg = torch.from_numpy(new_center_value).to(device)
        else:
            ctx.fail('Error: New center has strange format! Only an int (seed) or a file (.npy/.npz) are accepted!')

    # Set up the video capture dimensions
    height = demo_height
    width = int(4.0/3*demo_height) if demo_width is None else demo_width
    sheight = int(height)
    swidth = sheight

    # Set display size. If none specified or user selects 'max', will use image resolution of the network
    display_height = G.img_resolution if (display_height is None or display_height == 'max') else display_height

    # Fix a dlatent to do style mixing
    static_w = G.mapping(torch.randn(1, G.z_dim, device=device, generator=torch.Generator(device=device).manual_seed(seed)), c=None)

    cam = cv2.VideoCapture(0)
    idx = 0

    start_time = time.time()
    x = 1  # displays the frame rate every 1 second if verbose is True
    counter = 0
    recording_flag = False

    # Preprocess each image for VGG16
    preprocess = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

    while cam.isOpened():
        # read frame
        idx += 1
        ret_val, img = cam.read()
        img = imutils.resize(img, height=height)
        if mirror:
            img = cv2.flip(img, 1)
        img = np.array(img).transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).float().to(device)

        frame = preprocess(img).to(device)
        fake_z = vgg16_features.get_layers_features(frame, layers=[layer])[0]
        
        # v0 
        if v0:
            fake_z = fake_z.view(1, 512, -1).mean(2)

            # Perform EMA with previous fake_z
            if counter == 0:
                prev_fake_z = fake_z
            # Do EMA
            fake_z = 0.2 * prev_fake_z + 0.8 * fake_z
            prev_fake_z = fake_z

            fake_w = gen_utils.z_to_dlatent(G, fake_z, label, truncation_psi)

            # Do style mixing
            fake_w[:, 4:] = static_w[:, 4:]

        # v1
        elif v1:
            _n, _c, h, w = fake_z.shape
            
            # Separate into coarse/middle/fine according to areas
            coarse_fake_z = fake_z[:, :, :h//2, :]
            middle_fake_z = fake_z[:, :, h//2:, :w//2]
            fine_fake_z = fake_z[:, :, h//2:, w//2:]

            # Convert them to the expected shape (each region will be their own latent)
            coarse_fake_z = coarse_fake_z.reshape(1, G.z_dim, -1).mean(2)
            middle_fake_z = middle_fake_z.reshape(1, G.z_dim, -1).mean(2)
            fine_fake_z = fine_fake_z.reshape(1, G.z_dim, -1).mean(2)

            # Get the respective dlatents
            coarse_fake_w = gen_utils.z_to_dlatent(G, coarse_fake_z, label, 1.0)
            middle_fake_w = gen_utils.z_to_dlatent(G, middle_fake_z, label, 1.0)
            fine_fake_w = gen_utils.z_to_dlatent(G, fine_fake_z, label, 1.0)
            fake_w = torch.cat([coarse_fake_w[:, :4], middle_fake_w[:, 4:8], fine_fake_w[:, 8:]], dim=1)  # [1, G.num_ws, G.z_dim]

            # Perform EMA with previous fake_w
            if counter == 0:
                prev_fake_w = fake_w
            # Do EMA
            fake_w = 0.4 * prev_fake_w + 0.6 * fake_w
            prev_fake_w = fake_w

        # Set images to expected data type
        img = img.clamp(0, 255).data[0].cpu().numpy()
        img = img.transpose(1, 2, 0).astype('uint8')

        simg = gen_utils.w_to_img(G, fake_w, noise_mode, w_avg, truncation_psi)[0]
        simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)
        
        # display
        if not only_synth:
            display_width = int(4/3*display_height)
            # Resize input image from the camera
            img = cv2.resize(img, (display_width, display_height))
            # Resize accordingly the synthesized image
            simg = cv2.resize(simg, (display_height, display_height), interpolation=cv2.INTER_CUBIC)
            img = np.concatenate((img, simg), axis=1)
            cv2.imshow('Visuorreactive Demo', img)
        else:
            # Resize the synthesized image to the desired display height/width
            simg = cv2.resize(simg, (display_height, display_height))
            cv2.imshow('Visuorreactive Demo - Only Synth Image', simg)

        counter += 1

        # FPS counter
        if (time.time() - start_time) > x and verbose:
            print(f"FPS: {counter / (time.time() - start_time):0.2f}")
            counter = 0
            start_time = time.time()

        # ESC to quit; SPACE to start recording
        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key == 32:
            # Transition from not recording to recording
            if not recording_flag:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                w = width + swidth if not only_synth else G.img_resolution
                out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, height))
                recording_flag = True
            else:
                recording_flag = False
                out.release()

        if recording_flag:
            out.write(img)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    live_visual_reactive()