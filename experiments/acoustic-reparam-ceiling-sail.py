#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
from tqdm import trange

import drjit as dr
import mitsuba as mi

from utils import shoebox_scene, estimate_max_depth

mi.set_variant('cuda_ad_acoustic')
mi.set_log_level(mi.LogLevel.Warn)

sess_seed   = np.random.randint(0, 2**30)
sess_seed_g = np.random.randint(0, 2**30)
print(f"session seeds are: sess_seed={sess_seed}; sess_seed_g={sess_seed_g}")

#############################################################################
# Scene Construction
#############################################################################

mic_poses = np.array([[8., 1., 2.],
                    #   [5., 2., 5.],
                      [2., 3., 8.]])

sail_size    = np.array([8., 2., 3.]) / 2.
sail_poses   = np.array([[ 6., 16., 2.5],
                         [ 6., 16., 7.5],
                         [16., 16., 2.5],
                         [16., 16., 7.5]])

absorption = [0.6, 0.8] * mic_poses.shape[0]
# scattering = [0.2, 0.8]

# absorption_mdim, scattering_mdim = np.meshgrid(absorption, scattering)
# absorption,      scattering      = absorption_mdim.flatten(), scattering_mdim.flatten()

config = {
    "box_dim":     [22., 18., 10.],
    "mic_pos":     mic_poses[0],
    "speaker_pos": [18.,  2., 5.],
    "speaker_radius": 1.0, #0.1,

    "absorption": [(i + 1, a) for i, a in enumerate(absorption)],
    "scattering": 0.4, #[(i + 1, s) for i, s in enumerate(scattering)],

    "wav_bins": len(absorption), # x
    "time_bins": 150,             # y
    "max_time":  0.15,

    # "integrator": "prb_acoustic",
    "integrator": "prb_reparam_acoustic",
    "max_depth": 25,
    "spp": 2**22,
}

config["max_depth"] = max([config["max_depth"], estimate_max_depth(config["box_dim"], config["max_time"], 2.5)])
print(f"max_depth = {config['max_depth']}")

scene_dict      = shoebox_scene(**config)
sail_poses_dict = {}

tf        = mi.ScalarTransform4f
box_dim   = np.array(config['box_dim']) / 2.

scene_dict["integrator"]["skip_direct"] = True
if "reparam" in config["integrator"]:
    scene_dict["integrator"]["reparam_rays"] = 16

del scene_dict["sensor"]["microphoneA"]
for i, m in enumerate(mic_poses):
    scene_dict["sensor"][f"microphone_{i}"] = {
        "type": "microphone",
        "to_world": tf.translate(m - box_dim),
    }

sail_vertex_base = mi.Transform4f.scale(sail_size).rotate([1., 0., 0.], angle=90.) @ mi.Point3f(
    [-1., -1.,  1.,  1.],
    [-1.,  1., -1.,  1.],
    [ 0.,  0.,  0.,  0.]
)

for i, s in enumerate(sail_poses):
    sail_poses_dict[f"sail_{i}.vertex_positions"] = mi.Point3f(s - box_dim)
    scene_dict[f"sail_{i}"] = {
        "type": "ply",
        "filename": "/home/daniel/Studium/masterarbeit/src/data/scenes/rectangle_mesh.ply",
        "bsdf": {
            "type": "acousticbsdf",
            "scattering": { "type": "spectrum", "value": 0.1 },
            "absorption": { "type": "spectrum", "value": 0.1 },
        },
        "to_world": tf.translate(s - box_dim).scale(sail_size).rotate([1., 0., 0.], angle=90.)
    }

scene = mi.load_dict(scene_dict)
params = mi.traverse(scene)

#############################################################################
# Optimization Setup
#############################################################################

opt = mi.ad.Adam(lr=0.002)
for key in sail_poses_dict:
    opt[key] = mi.Point2f(0.0, 0.0)
    # opt[key] = mi.Point2f(*(np.random.rand(2) * 0.2 - 0.10))

def apply_transform(params_to_update):
    for key in sail_poses_dict:
        opt[key] = dr.clamp(opt[key], -1., 1.)

        sin_a, cos_a = dr.sincos(opt[key].x * dr.pi / 6.)
        sin_b, cos_b = dr.sincos(opt[key].y * dr.pi / 6.)

        # Rz  = mi.Matrix3f([cos_b, sin_b, 0.], [-sin_b, cos_b, 0],     [0,  0,      1])
        # Rx  = mi.Matrix3f([1.,    0.,    0.], [0.,     cos_a, sin_a], [0., -sin_a, cos_a])
        Rzx  = mi.Matrix3f([cos_b, sin_b, 0.], [-sin_b * cos_a, cos_b * cos_a, sin_a], [sin_b * sin_a, -cos_b * sin_a, cos_a])

        params_to_update[key] = dr.ravel((Rzx @ sail_vertex_base) + sail_poses_dict[key])

    params_to_update.update()

#############################################################################
# Setup Loss
#############################################################################

def loss(hist, ref=None, channels=2, cams=2, cfg=config):
    assert ref is None
    assert channels * cams == cfg["wav_bins"]

    idx = dr.arange(mi.UInt32, channels * cfg["time_bins"])
    y = idx // channels
    x = idx - y * channels
    idx = y * 2 * channels + x

    vals_A = dr.gather(mi.Float, hist[:, :, 0].array, idx)
    vals_B = dr.gather(mi.Float, hist[:, :, 0].array, idx + 2)
    return dr.mean(dr.sqr(vals_A - vals_B))

#############################################################################
# Main Optimization Loop
#############################################################################

errors, losses, grads = [], [], []

iters = 400

losses = []
for i in trange(iters) if iters > 1 else range(iters):
    apply_transform(params)
    img = mi.render(scene, params, seed=sess_seed+i, seed_grad=sess_seed_g+i)

    l = loss(img)
    losses.append(l)
    dr.backward(-1. * l)

    angles, sail_grads = [], []
    for key in sail_poses_dict:
        angles.append(opt[key])
        sail_grads.append(dr.grad(opt[key]))

    errors.append(angles)
    grads.append(sail_grads)
    losses.append(l)

    opt.step()

#############################################################################
# Output
#############################################################################

for key in sail_poses_dict:
    print(opt[key])

V      = np.array(errors)[:, :, 0]
G      = np.array(grads)[:, :, 0]
losses = np.array(losses)[:, 0]

exi = 20
np.save(f"ceiling-sail_{exi:02d}.npy", V)
np.save(f"ceiling-sail_{exi:02d}_grads.npy", G)
# np.save(f"ceiling-sail_{exi:02d}_losses_optloop.npy", losses)

print(V.shape, G.shape)

if True:
    losses = []

    for i in trange(V.shape[0]):
        opt["sail_0.vertex_positions"] = mi.Point2f(V[i, 0])
        opt["sail_1.vertex_positions"] = mi.Point2f(V[i, 1])
        opt["sail_2.vertex_positions"] = mi.Point2f(V[i, 2])
        opt["sail_3.vertex_positions"] = mi.Point2f(V[i, 3])
        apply_transform(params)

        img = mi.render(scene, seed=sess_seed)[:, :, 0].numpy()
        losses.append(np.mean(np.square(img[:, :2] - img[:, 2:])))

    np.save(f"ceiling-sail_{exi:02d}_losses.npy", np.array(losses))
