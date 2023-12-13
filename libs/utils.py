import numpy as np
import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi


def dict_replace(d, key, val):
    new = d.copy()
    new[key] = val
    return new


def plot_hist(image: mi.TensorXf,
              figsize                = (6, 4),
              time:       np.ndarray = None,
              max_time:   float      = None,
              time_bins:  int        = None,
              absorption: list       = None,
              log:        bool       = False,
              abs:        bool       = False,
              ylim                   = None,
              **_):

    if log:
        image = dr.log(dr.abs(image))
    elif abs:
        image = dr.abs(image)

    if time is None:
        time = np.linspace(0., max_time, time_bins, endpoint=False)

    plt.figure(figsize=figsize)

    try:
        label = list(map(lambda x: f"$\\alpha={x[1]:.2f}$", absorption))
        plt.plot(time, image.numpy(), label=label)
        plt.legend()
    except TypeError:
        plt.plot(time, image.numpy())

    plt.ylim(ylim)
    plt.xlabel('time [s]')
    plt.ylabel('Energy [a.U.]')
    plt.show()


def bitmap(image, convert=True):
    bim = mi.Bitmap(image)
    if convert:
        bim = bim.convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True)
    return bim


def plot_img(image, convert=True, figsize=(16, 9)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(bitmap(image, convert=convert))
    plt.show()


def drjit_turn_off_optimizations(apply=True):
    if apply:
        dr.set_flag(dr.JitFlag.LoopRecord,       False)
        dr.set_flag(dr.JitFlag.VCallRecord,      False)
        dr.set_flag(dr.JitFlag.VCallDeduplicate, False)
        dr.set_flag(dr.JitFlag.ConstProp,        False)
        dr.set_flag(dr.JitFlag.ValueNumbering,   False)
        dr.set_flag(dr.JitFlag.VCallOptimize,    False)
        dr.set_flag(dr.JitFlag.LoopOptimize,     False)


def write_rectangle_mesh(path=None):
    rect = mi.Mesh('rectangle',
                   vertex_count=4,
                   face_count=2,
                   has_vertex_normals=False,
                   has_vertex_texcoords=False)
    rect_params = mi.traverse(rect)
    rect_params['faces'] = dr.ravel(mi.Vector3u([2, 1],
                                                [1, 2],
                                                [0, 3]))
    rect_params['vertex_positions'] = dr.ravel(mi.Point3f([-1., -1., 1., 1.],
                                                          [-1., 1., -1., 1.],
                                                          0.))
    rect_params.update()
    path = '/home/daniel/Studium/masterarbeit/src/data/scenes/rectangle_mesh.ply' if path is None else path
    rect.write_ply(path)


def write_cube_mesh(path=None):
    cube = mi.Mesh('cube',
                   vertex_count=8,
                   face_count=12,
                   has_vertex_normals=False,
                   has_vertex_texcoords=False)

    cube_params = mi.traverse(cube)
    cube_params['faces']            = dr.ravel(mi.Vector3u([0, 3, 0, 6, 5, 3, 6, 5, 4, 1, 2, 7],
                                                           [1, 2, 2, 4, 3, 5, 5, 6, 1, 4, 3, 6],
                                                           [2, 1, 4, 2, 1, 7, 4, 7, 0, 5, 6, 3]))
    cube_params['vertex_positions'] = dr.ravel(mi.Point3f([-1., -1., -1., -1.,  1.,  1.,  1., 1.],
                                                          [-1., -1.,  1.,  1., -1., -1.,  1., 1.],
                                                          [-1.,  1., -1.,  1., -1.,  1., -1., 1.]))
    cube_params.update()
    path = '/home/daniel/Studium/masterarbeit/src/data/scenes/cube_mesh.ply' if path is None else path
    cube.write_ply(path)


def generate_shoebox(bsdf=None, **kwargs):
    shoebox = {
        "type": "shapegroup",
        "back": {
            "type": "ply",
            "filename": "/home/daniel/Studium/masterarbeit/data/scenes/meshes/rectangle.ply",
            "bsdf": { "type": "ref", "id": "back_bsdf" },
            "to_world": mi.ScalarTransform4f.translate([0., 0., -1.]),
        },
        "front": {
            "type": "ply",
            "filename": "/home/daniel/Studium/masterarbeit/data/scenes/meshes/rectangle.ply",
            "bsdf": { "type": "ref", "id": "front_bsdf" },
            "flip_normals": True,
            "to_world": mi.ScalarTransform4f.translate([0., 0., 1.]),
        },
        "left": {
            "type": "ply",
            "filename": "/home/daniel/Studium/masterarbeit/data/scenes/meshes/rectangle.ply",
            "bsdf": { "type": "ref", "id": "left_bsdf" },
            "to_world": mi.ScalarTransform4f.translate([-1., 0., 0.]).rotate(axis=[0., 1., 0.], angle=90),
        },
        "right": {
            "type": "ply",
            "filename": "/home/daniel/Studium/masterarbeit/data/scenes/meshes/rectangle.ply",
            "bsdf": { "type": "ref", "id": "right_bsdf" },
            "to_world": mi.ScalarTransform4f.translate([1., 0., 0.]).rotate(axis=[0., -1., 0.], angle=90),
        },
        "top": {
            "type": "ply",
            "filename": "/home/daniel/Studium/masterarbeit/data/scenes/meshes/rectangle.ply",
            "bsdf": { "type": "ref", "id": "top_bsdf" },
            "to_world": mi.ScalarTransform4f.translate([0., 1., 0.]).rotate(axis=[1., 0., 0.], angle=90),
        },
        "bottom": {
            "type": "ply",
            "filename": "/home/daniel/Studium/masterarbeit/data/scenes/meshes/rectangle.ply",
            "bsdf": { "type": "ref", "id": "bottom_bsdf" },
            "to_world": mi.ScalarTransform4f.translate([0., -1., 0.]).rotate(axis=[-1., 0., 0.], angle=90),
        },
    }

    if bsdf is not None:
        for val in shoebox.values():
            if "bsdf" in val:
                val["bsdf"] = bsdf

    for face in list(shoebox.keys()):
        if face in kwargs and not kwargs[face]:
            shoebox.pop(face)

    return shoebox


def diff_shoebox_bsdf():
    return {
        "back_bsdf":   { "type": "diffuse", "reflectance": { "type": "rgb", "value": [.1, .1, .3] } },
        "front_bsdf":  { "type": "diffuse", "reflectance": { "type": "rgb", "value": [.3, .3, .1] } },
        "left_bsdf":   { "type": "diffuse", "reflectance": { "type": "rgb", "value": [.1, .3, .1] } },
        "right_bsdf":  { "type": "diffuse", "reflectance": { "type": "rgb", "value": [.1, .3, .3] } },
        "top_bsdf":    { "type": "diffuse", "reflectance": { "type": "rgb", "value": [.3, .1, .3] } },
        "bottom_bsdf": { "type": "diffuse", "reflectance": { "type": "rgb", "value": [.3, .1, .1] } },
    }


def shoebox_scene(**kwargs):
    tf          = mi.ScalarTransform4f
    box_dim     = np.array(kwargs['box_dim']) / 2.
    mic_pos     = np.array(kwargs['mic_pos'])
    speaker_pos = np.array(kwargs['speaker_pos'])

    scene = {
        "type": "scene",

        "integrator": {
            "type": kwargs['integrator'],
            "max_depth": kwargs['max_depth'],
            "max_time": kwargs['max_time'],
        },

        "sensor": {
            "type": "batch",
            "microphoneA": {
                "type": "microphone",
                "to_world": tf.translate(mic_pos - box_dim),
            },
            "film": {
                "type": "tape",
                "wav_bins": kwargs['wav_bins'],
                "time_bins": kwargs['time_bins'],
                "rfilter": { "type": "box" },
                "count": True
            },
            "sampler": { "type": "stratified", "sample_count": kwargs['spp'] },
        },

        "acoustic_bsdf": {
            "type": "acousticbsdf",
            "scattering": { "type": "spectrum", "value": kwargs['scattering'] },
            "absorption": { "type": "spectrum", "value": kwargs['absorption'] },
        },

        "speaker": {
            "type": "ply",
            "filename": "/home/daniel/Studium/masterarbeit/data/scenes/meshes/sphere.ply",
            "to_world": tf.translate(speaker_pos - box_dim).scale(kwargs['speaker_radius']),
            "emitter": { "type": "area", "radiance": { "type": "uniform", "value": 1. } },
        },
    }

    if "connected_cube" in kwargs and kwargs["connected_cube"]:
        scene["shoebox"] = {
            "type": "ply",
            "filename": "/home/daniel/Studium/masterarbeit/data/scenes/meshes/cuberoom.ply",
            "face_normals": True,
            "flip_normals": True,
            "to_world": tf.scale(box_dim),
            "bsdf": { "type": "ref", "id": "acoustic_bsdf" },
        }
    else:
        scene["shoebox"] = generate_shoebox(bsdf={ "type": "ref", "id": "acoustic_bsdf" })

        scene["main_box"] = {
            "type": "instance",
            "geometry": { "type": "ref", "id": "shoebox" },
            "to_world": tf.scale(box_dim),
        }

    return scene


def shoebox_scene_visual(resf=2, **kwargs):
    tf          = mi.ScalarTransform4f
    box_dim     = np.array(kwargs['box_dim']) / 2.
    mic_pos     = np.array(kwargs['mic_pos'])
    speaker_pos = np.array(kwargs['speaker_pos'])

    scene = {
        "type": "scene",

        "integrator": {
            "type": "prb_reparam",
            "max_depth": 8,
            "hide_emitters": True,
        },

        "sensor": {
            "type": "batch",
            "cameraA": {
                "type": "perspective",
                "to_world": tf.look_at(
                    origin=box_dim * np.array([-3.,  0.,  6.]),
                    target=box_dim * np.array([  .5, 0., -1.]),
                    up=[0, 1, 0]
                ),
            },
            "film": {
                "type": "hdrfilm",
                "rfilter": { "type": "gaussian" },
                "width": 128 * resf,
                "height": 72 * resf,
                "sample_border": True,
            },
            "sampler": { "type": "stratified", "sample_count": 256, },
        },

        "microphoneA": {
            "type": "sphere",
            "radius": .15,
            "to_world": tf.translate(mic_pos - box_dim),
            "emitter": { "type": "area", "radiance": { "type": "rgb", "value": [0.4, 0., 0.1] } },
        },

        "speaker": {
            "type": "ply",
            "filename": "/home/daniel/Studium/masterarbeit/data/scenes/meshes/sphere.ply",
            "to_world": tf.translate(speaker_pos - box_dim).scale(kwargs['speaker_radius']),
            "emitter": { "type": "area", "radiance": { "type": "uniform", "value": 1. } },
        },

        "env_emitter": {
            "type": "constant",
            "radiance": { "type": "spectrum", "value": 1. }
        },
    }

    if "connected_cube" in kwargs and kwargs["connected_cube"]:
        scene["shoebox"] = {
            "type": "ply",
            "filename": "/home/daniel/Studium/masterarbeit/data/scenes/meshes/cuberoom.ply",
            "face_normals": True,
            "flip_normals": False,
            "to_world": tf.scale(box_dim),
            "bsdf": { "type": "diffuse", "reflectance": { "type": "rgb", "value": [0.1, 0.1, 0.7] } },
        }
    else:
        scene.update(diff_shoebox_bsdf())
        scene["shoebox"] = generate_shoebox(back=True, front=False, left=False, right=True, top=True, bottom=True)

        scene["main_box"] = {
            "type": "instance",
            "geometry": { "type": "ref", "id": "shoebox" },
            "to_world": tf.scale(box_dim),
        }

    return scene


def estimate_max_depth(box_dimensions, max_time, boost=1., speed_of_sound=343.):
    max_box_distance   = np.linalg.norm(box_dimensions) / 2.
    max_box_time       = max_box_distance / speed_of_sound
    max_depth_estimate = int(np.ceil(boost * max_time / max_box_time))
    return max_depth_estimate


def remove_direct(x, distance, fs=1000., c=343.):
    """ helper function to remove sound which hit the microphone directly """
    i = int((distance / c) * fs) + 1
    x[:i] = 0.
    return x


def mse(image, reference):
    if len(image.shape) > 2:
        image = image[:, :, 0]

    if len(reference.shape) > 2:
        reference = reference[:, :, 0]

    return dr.mean(dr.sqr(image - reference))


def mae(image, reference):
    if len(image.shape) > 2:
        image = image[:, :, 0]

    if len(reference.shape) > 2:
        reference = reference[:, :, 0]

    return dr.mean(dr.abs(image - reference))


def msl(image, reference):
    if len(image.shape) > 2:
        image = image[:, :, 0]

    if len(reference.shape) > 2:
        reference = reference[:, :, 0]

    return dr.mean(dr.sqr(dr.log(image) - dr.log(reference)))


def lhc(image, reference):
    if len(image.shape) > 2:
        image = image[:, :, 0]

    if len(reference.shape) > 2:
        reference = reference[:, :, 0]

    return dr.sum(dr.log(dr.cosh(image - reference)))
