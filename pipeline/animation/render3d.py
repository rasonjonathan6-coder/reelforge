"""Real 3D character renderer — a software rasteriser in pure Python + Pillow.

There is no GPU, no Blender and no numpy in this deployment, so this module is
a genuine 3D engine written from scratch:

* a 3D scene graph of triangles (boxes, spheres, tapered limbs),
* a perspective camera with a proper view basis and projection,
* per-face Lambert shading from a key light plus ambient,
* a skeletal rig (hips -> spine -> chest -> neck/head, shoulders -> elbows ->
  wrists, hips -> knees -> ankles) animated by forward kinematics,
* painter's algorithm depth sorting, so 3D geometry really occludes itself.

The result is a real 3D cartoon: the camera orbits a volumetric character that
turns, leans, gestures and walks inside a 3D set — not a flat drawing.

Rendering happens at a fraction of the output size (`scale`) and is upscaled,
because triangle setup costs scale with pixel count while the silhouette and
shading are resolution independent.
"""

from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from pipeline import avatars

WIDTH, HEIGHT, FPS = 1080, 1920, 30

Vec3 = tuple[float, float, float]
Mat3 = tuple[float, float, float, float, float, float, float, float, float]


# ---------------------------------------------------------------------------
# Minimal 3D math (tuples, no numpy)
# ---------------------------------------------------------------------------
def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return (
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    )


def rot_x(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return (1, 0, 0, 0, c, -s, 0, s, c)


def rot_y(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return (c, 0, s, 0, 1, 0, -s, 0, c)


def rot_z(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return (c, -s, 0, s, c, 0, 0, 0, 1)


def apply(m: Mat3, v: Vec3) -> Vec3:
    x, y, z = v
    return (m[0] * x + m[1] * y + m[2] * z,
            m[3] * x + m[4] * y + m[5] * z,
            m[6] * x + m[7] * y + m[8] * z)


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def norm(a: Vec3) -> Vec3:
    length = math.sqrt(dot(a, a)) or 1.0
    return (a[0] / length, a[1] / length, a[2] / length)


# ---------------------------------------------------------------------------
# Geometry buffer
# ---------------------------------------------------------------------------
@dataclass
class Tri:
    a: Vec3
    b: Vec3
    c: Vec3
    color: tuple[int, int, int]
    spec: float = 0.0      # specular strength (glossiness of the material)
    shin: float = 24.0     # specular exponent (higher = tighter highlight)


# Materials: (specular strength, exponent). Skin is soft, eyes are glossy,
# cloth is matte, hair is semi-gloss — that spread is what stops the model
# from reading as uniformly plastic.
MAT_SKIN = (0.28, 18.0)
MAT_CLOTH = (0.05, 8.0)
MAT_HAIR = (0.40, 32.0)
MAT_EYE = (0.85, 90.0)
MAT_METAL = (0.55, 60.0)
MAT_MATTE = (0.02, 6.0)


def _add_box(buf: list[Tri], center: Vec3, size: Vec3, rot: Mat3,
             color: tuple[int, int, int], mat: tuple[float, float] = MAT_CLOTH) -> None:
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    local = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
             (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    pts = [add(center, apply(rot, p)) for p in local]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
    for i0, i1, i2, i3 in faces:
        buf.append(Tri(pts[i0], pts[i1], pts[i2], color, *mat))
        buf.append(Tri(pts[i0], pts[i2], pts[i3], color, *mat))


def _add_sphere(buf: list[Tri], center: Vec3, radius: float,
                color: tuple[int, int, int], rings: int = 9, segments: int = 14,
                squash: Vec3 = (1.0, 1.0, 1.0),
                mat: tuple[float, float] = MAT_SKIN) -> None:
    grid: list[list[Vec3]] = []
    for i in range(rings + 1):
        theta = math.pi * i / rings
        row: list[Vec3] = []
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            p = (radius * math.sin(theta) * math.cos(phi) * squash[0],
                 radius * math.cos(theta) * squash[1],
                 radius * math.sin(theta) * math.sin(phi) * squash[2])
            row.append(add(center, p))
        grid.append(row)
    for i in range(rings):
        for j in range(segments):
            j2 = (j + 1) % segments
            a, b = grid[i][j], grid[i][j2]
            c, d = grid[i + 1][j2], grid[i + 1][j]
            if i != 0:
                buf.append(Tri(a, b, c, color, *mat))
            if i != rings - 1:
                buf.append(Tri(a, c, d, color, *mat))


def _add_limb(buf: list[Tri], p0: Vec3, p1: Vec3, r0: float, r1: float,
              color: tuple[int, int, int], sides: int = 8,
              mat: tuple[float, float] = MAT_SKIN) -> None:
    """Tapered prism from `p0` to `p1` — arms, legs, necks, fingers."""
    axis = sub(p1, p0)
    length = math.sqrt(dot(axis, axis))
    if length < 1e-6:
        return
    w = norm(axis)
    helper = (0.0, 0.0, 1.0) if abs(w[1]) < 0.9 else (1.0, 0.0, 0.0)
    u = norm(cross(w, helper))
    v = cross(w, u)
    ring0: list[Vec3] = []
    ring1: list[Vec3] = []
    for k in range(sides):
        ang = 2 * math.pi * k / sides
        cu, cv = math.cos(ang), math.sin(ang)
        ring0.append(add(p0, (u[0] * cu * r0 + v[0] * cv * r0,
                              u[1] * cu * r0 + v[1] * cv * r0,
                              u[2] * cu * r0 + v[2] * cv * r0)))
        ring1.append(add(p1, (u[0] * cu * r1 + v[0] * cv * r1,
                              u[1] * cu * r1 + v[1] * cv * r1,
                              u[2] * cu * r1 + v[2] * cv * r1)))
    for k in range(sides):
        k2 = (k + 1) % sides
        buf.append(Tri(ring0[k], ring0[k2], ring1[k2], color, *mat))
        buf.append(Tri(ring0[k], ring1[k2], ring1[k], color, *mat))
    for k in range(1, sides - 1):
        buf.append(Tri(ring1[0], ring1[k], ring1[k + 1], color, *mat))
        buf.append(Tri(ring0[0], ring0[k + 1], ring0[k], color, *mat))


def _add_quad(buf: list[Tri], p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3,
              color: tuple[int, int, int],
              mat: tuple[float, float] = MAT_MATTE) -> None:
    buf.append(Tri(p0, p1, p2, color, *mat))
    buf.append(Tri(p0, p2, p3, color, *mat))


def _add_disc(buf: list[Tri], center: Vec3, radius: float, normal: Vec3,
              color: tuple[int, int, int], sides: int = 10,
              mat: tuple[float, float] = MAT_MATTE) -> None:
    """Flat disc facing `normal` — pupils, irises, catchlights."""
    n = norm(normal)
    helper = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (1.0, 0.0, 0.0)
    u = norm(cross(n, helper))
    v = cross(n, u)
    ring = [add(center, add(scale(u, radius * math.cos(2 * math.pi * k / sides)),
                            scale(v, radius * math.sin(2 * math.pi * k / sides))))
            for k in range(sides)]
    for k in range(sides):
        buf.append(Tri(center, ring[k], ring[(k + 1) % sides], color, *mat))


def _add_poly(buf: list[Tri], points: list[Vec3], color: tuple[int, int, int],
              mat: tuple[float, float] = MAT_MATTE) -> None:
    """Fan-triangulate a convex polygon (eyelid and lip shapes)."""
    for k in range(1, len(points) - 1):
        buf.append(Tri(points[0], points[k], points[k + 1], color, *mat))


# ---------------------------------------------------------------------------
# Rig: forward kinematics
# ---------------------------------------------------------------------------
# Body proportions in metres, feet on the floor at y=0.
HIP_H, TORSO, NECK, HEAD_L = 0.95, 0.42, 0.10, 0.20
SHOULDER_W, HIP_W = 0.20, 0.11
UPPER_ARM, FOREARM = 0.30, 0.28
THIGH, SHIN = 0.45, 0.45
HEAD_R = 0.155


@dataclass
class Pose:
    """Joint angles (radians) and facial state for one frame."""

    root_y: float = 0.0
    lean: float = 0.0
    turn: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    l_shoulder: float = 0.0
    l_elbow: float = 0.0
    r_shoulder: float = 0.0
    r_elbow: float = 0.0
    l_arm_out: float = 0.10
    r_arm_out: float = 0.10
    l_hip: float = 0.0
    l_knee: float = 0.0
    r_hip: float = 0.0
    r_knee: float = 0.0
    sit: bool = False
    mouth: float = 0.0
    mouth_wide: float = 1.0
    blink: float = 0.0
    brow: float = 0.0
    smile: float = 0.0        # -1 frown .. +1 smile
    squint: float = 0.0       # lower-lid raise (joy / anger)
    jaw_side: float = 0.0     # small lateral jaw shift while speaking


@dataclass
class Skeleton:
    """World-space joint positions for one frame."""

    hip: Vec3
    chest: Vec3
    neck: Vec3
    head: Vec3
    head_rot: Mat3
    l_shoulder: Vec3
    l_elbow: Vec3
    l_wrist: Vec3
    r_shoulder: Vec3
    r_elbow: Vec3
    r_wrist: Vec3
    l_hip: Vec3
    l_knee: Vec3
    l_ankle: Vec3
    r_hip: Vec3
    r_knee: Vec3
    r_ankle: Vec3
    torso_rot: Mat3


def solve(pose: Pose, origin: Vec3 = (0.0, 0.0, 0.0)) -> Skeleton:
    """Forward kinematics: joint angles -> world-space joint positions."""
    body = mat_mul(rot_y(pose.turn), rot_x(pose.lean))
    hip_h = HIP_H * (0.62 if pose.sit else 1.0)
    hip = add(origin, (0.0, hip_h + pose.root_y, 0.0))

    up = apply(body, (0.0, 1.0, 0.0))
    right = apply(body, (1.0, 0.0, 0.0))
    chest = add(hip, scale(up, TORSO))
    neck = add(chest, scale(up, NECK))

    head_rot = mat_mul(body, mat_mul(rot_y(pose.head_yaw),
                                     mat_mul(rot_x(pose.head_pitch), rot_z(pose.head_roll))))
    head = add(neck, apply(head_rot, (0.0, HEAD_L, 0.0)))

    l_shoulder = add(chest, scale(right, -SHOULDER_W))
    r_shoulder = add(chest, scale(right, SHOULDER_W))

    def arm(shoulder: Vec3, swing: float, elbow: float, out: float):
        upper = mat_mul(body, mat_mul(rot_z(out), rot_x(swing)))
        d1 = apply(upper, (0.0, -1.0, 0.0))
        elbow_pos = add(shoulder, scale(d1, UPPER_ARM))
        lower = mat_mul(body, mat_mul(rot_z(out), rot_x(swing + elbow)))
        d2 = apply(lower, (0.0, -1.0, 0.0))
        return elbow_pos, add(elbow_pos, scale(d2, FOREARM))

    l_elbow, l_wrist = arm(l_shoulder, pose.l_shoulder, pose.l_elbow, pose.l_arm_out)
    r_elbow, r_wrist = arm(r_shoulder, pose.r_shoulder, pose.r_elbow, pose.r_arm_out)

    l_hip = add(hip, scale(right, -HIP_W))
    r_hip = add(hip, scale(right, HIP_W))
    thigh_scale = 0.62 if pose.sit else 1.0

    def leg(hip_pos: Vec3, hip_ang: float, knee_ang: float):
        base = body if not pose.sit else rot_y(pose.turn)
        t = mat_mul(base, rot_x(hip_ang))
        knee_pos = add(hip_pos, scale(apply(t, (0.0, -1.0, 0.0)), THIGH * thigh_scale))
        s = mat_mul(base, rot_x(hip_ang + knee_ang))
        return knee_pos, add(knee_pos, scale(apply(s, (0.0, -1.0, 0.0)), SHIN * thigh_scale))

    l_knee, l_ankle = leg(l_hip, pose.l_hip, pose.l_knee)
    r_knee, r_ankle = leg(r_hip, pose.r_hip, pose.r_knee)

    return Skeleton(hip, chest, neck, head, head_rot, l_shoulder, l_elbow, l_wrist,
                    r_shoulder, r_elbow, r_wrist, l_hip, l_knee, l_ankle,
                    r_hip, r_knee, r_ankle, body)


# ---------------------------------------------------------------------------
# Motion: turn the scene's action/emotion/visemes into joint angles
# ---------------------------------------------------------------------------
def pose_at(action: str, emotion: str, t: float, talking: bool, mouth: float,
            mouth_wide: float, walk_phase: float = 0.0) -> Pose:
    """One frame of character motion, driven by the scene's own direction."""
    p = Pose()
    bob = 0.012 * math.sin(2 * math.pi * 0.9 * t)
    p.root_y = bob
    p.turn = 0.12 * math.sin(2 * math.pi * 0.18 * t)

    if action in ("marche", "court"):
        speed = 2.1 if action == "court" else 1.5
        phase = 2 * math.pi * speed * t
        p.l_hip = 0.55 * math.sin(phase)
        p.r_hip = -0.55 * math.sin(phase)
        p.l_knee = -0.65 * max(0.0, math.sin(phase + 1.2)) - 0.12
        p.r_knee = -0.65 * max(0.0, math.sin(phase + 1.2 + math.pi)) - 0.12
        p.l_shoulder = -0.55 * math.sin(phase)
        p.r_shoulder = 0.55 * math.sin(phase)
        p.l_elbow = -0.25 - 0.15 * max(0.0, math.sin(phase))
        p.r_elbow = -0.25 - 0.15 * max(0.0, -math.sin(phase))
        p.l_arm_out = 0.14
        p.r_arm_out = 0.14
        # Counter-rotate the torso against the legs: that is what sells a walk.
        p.turn += 0.10 * math.sin(phase)
        p.root_y = 0.02 * abs(math.sin(phase))
        p.lean = 0.06
        p.head_pitch = 0.04
    elif action in ("s'assied", "assied", "assis"):
        p.sit = True
        p.l_hip = -1.45
        p.r_hip = -1.45
        p.l_knee = 1.35
        p.r_knee = 1.35
        p.l_shoulder = -0.18
        p.r_shoulder = -0.18
        p.l_elbow = -0.75
        p.r_elbow = -0.75
        p.lean = 0.05
    else:
        p.lean = 0.03
        gesture = math.sin(2 * math.pi * 0.45 * t)
        if talking:
            p.l_shoulder = -0.28 + 0.22 * gesture
            p.r_shoulder = -0.28 - 0.22 * gesture
            p.l_elbow = -0.85 - 0.20 * gesture
            p.r_elbow = -0.85 + 0.20 * gesture
            p.l_arm_out = 0.30 + 0.10 * gesture
            p.r_arm_out = 0.30 - 0.10 * gesture
            p.head_pitch = 0.05 * math.sin(2 * math.pi * 1.3 * t)
            p.head_yaw = 0.06 * math.sin(2 * math.pi * 0.5 * t)
        else:
            p.l_shoulder = -0.05
            p.r_shoulder = -0.05
            p.l_elbow = -0.22
            p.r_elbow = -0.22
            p.head_yaw = 0.05 * math.sin(2 * math.pi * 0.3 * t)

    if action in ("regarde", "prend"):
        p.head_yaw += 0.22
        p.head_pitch += 0.05
    if action in ("mange",):
        p.r_shoulder = -1.15 + 0.25 * math.sin(2 * math.pi * 1.6 * t)
        p.r_elbow = -1.5

    # Emotion reads through the head, the brow, the lids and the mouth shape.
    p.brow = {"joie": -0.35, "colere": 0.55, "surprise": -0.6,
              "tristesse": 0.35}.get(emotion, 0.0)
    p.smile = {"joie": 0.9, "surprise": 0.3, "colere": -0.6,
               "tristesse": -0.5}.get(emotion, 0.1)
    p.squint = {"joie": 0.7, "colere": 0.5}.get(emotion, 0.0)
    if emotion == "joie":
        p.head_roll = 0.06 * math.sin(2 * math.pi * 0.6 * t)
    elif emotion == "tristesse":
        p.head_pitch = 0.16
    elif emotion == "colere":
        p.lean = 0.10
        p.head_pitch = -0.06
    elif emotion == "surprise":
        p.head_pitch = -0.05
        p.brow = -0.6

    # Idle micro-motion: weight shift and a small lateral jaw drift while
    # talking. Without it a still character reads as a mannequin.
    p.lean += 0.012 * math.sin(2 * math.pi * 0.23 * t + 0.7)
    p.root_y += 0.004 * math.sin(2 * math.pi * 0.35 * t + 1.9)
    if talking:
        p.jaw_side = 0.004 * math.sin(2 * math.pi * 1.7 * t)

    p.mouth = mouth
    p.mouth_wide = mouth_wide
    p.blink = 0.0
    return p


# ---------------------------------------------------------------------------
# Character mesh
# ---------------------------------------------------------------------------
def _character_mesh(skel: Skeleton, pal: dict, pose: Pose,
                    buf: list[Tri]) -> None:
    skin, hair, outfit = pal["skin"], pal["hair"], pal["outfit"]
    trouser = tuple(int(c * 0.55) for c in outfit)
    shirt = tuple(int(c * 1.12) if max(outfit) < 220 else outfit for c in outfit)
    shoe = (44, 40, 48)
    body = skel.torso_rot
    body_right = apply(body, (1.0, 0.0, 0.0))

    # Torso: tapered chest and pelvis blocks (so the waist narrows), a collar
    # and a belt. Real proportions read as a person, not a stack of boxes.
    for frac, w, h, d, color in (
            (0.18, 0.44, 0.24, 0.27, outfit),
            (0.52, 0.40, 0.22, 0.25, outfit),
            (0.80, 0.35, 0.18, 0.23, shirt)):
        _add_box(buf, add(skel.hip, scale(sub(skel.chest, skel.hip), frac)),
                 (w, h, d), body, color)
    _add_box(buf, add(skel.hip, (0.0, 0.04, 0.0)), (0.38, 0.30, 0.25), body, trouser)
    _add_box(buf, add(skel.hip, (0.0, 0.10, 0.0)), (0.40, 0.06, 0.27), body, (52, 44, 40))
    # Neck + shoulders: trapezius slabs stop the head floating off the collar.
    _add_limb(buf, skel.neck, add(skel.neck, apply(body, (0.0, 0.10, 0.0))),
              0.070, 0.068, skin, 8, MAT_SKIN)
    for side in (-1, 1):
        _add_box(buf, add(skel.chest, add(scale(body_right, side * 0.14),
                                          apply(body, (0.0, -0.01, 0.0)))),
                 (0.16, 0.12, 0.24), body, outfit, MAT_CLOTH)
    head_r = skel.head_rot
    _add_sphere(buf, skel.head, HEAD_R, skin, rings=12, segments=20,
                squash=(0.92, 1.08, 0.96), mat=MAT_SKIN)
    _add_box(buf, add(skel.head, apply(head_r, (0.0, -0.13, 0.03))),
             (0.17, 0.11, 0.16), head_r, skin, MAT_SKIN)          # jaw / chin
    _add_sphere(buf, add(skel.head, apply(head_r, (0.0, -0.05, 0.14))),
                HEAD_R * 0.55, skin, rings=5, segments=9,
                squash=(0.9, 1.0, 0.6), mat=MAT_SKIN)             # muzzle
    for side in (-1, 1):
        _add_sphere(buf, add(skel.head, apply(head_r, (side * HEAD_R * 0.96,
                                                      0.0, 0.0))),
                    HEAD_R * 0.30, skin, rings=4, segments=8,
                    squash=(0.5, 1.1, 0.8), mat=MAT_SKIN)         # ear
    # Nose: a small wedge on the face.
    nose_c = add(skel.head, apply(head_r, (0.0, -0.01, HEAD_R * 0.88)))
    _add_limb(buf, add(nose_c, apply(head_r, (0.0, 0.03, -0.02))),
              add(nose_c, apply(head_r, (0.0, -0.03, 0.05))),
              0.016, 0.026, skin, 5, MAT_SKIN)

    # Hair: a shell over the crown, a fringe and side locks.
    _add_sphere(buf, add(skel.head, apply(head_r, (0.0, 0.012, -0.012))),
                HEAD_R * 1.08, hair, rings=11, segments=20,
                squash=(0.99, 1.03, 1.0), mat=MAT_HAIR)
    _add_box(buf, add(skel.head, apply(head_r, (0.0, 0.10, -0.02))),
             (0.31, 0.15, 0.31), head_r, hair, MAT_HAIR)
    for index in range(3):
        side = -1 if index % 2 else 1
        _add_box(buf, add(skel.head, apply(head_r,
                                          (side * 0.15, 0.10 - index * 0.05, 0.0))),
                 (0.10, 0.10 + index * 0.03, 0.25), head_r, hair, MAT_HAIR)
    _add_limb(buf, add(skel.head, apply(head_r, (-0.12, 0.05, -0.02))),
              add(skel.head, apply(head_r, (-0.15, -0.14, 0.02))),
              0.075, 0.05, hair, 6, MAT_HAIR)
    _add_limb(buf, add(skel.head, apply(head_r, (0.12, 0.05, -0.02))),
              add(skel.head, apply(head_r, (0.15, -0.14, 0.02))),
              0.075, 0.05, hair, 6, MAT_HAIR)
    # Flyaway strands: thin tapered wisps around the hairline. They break the
    # perfectly smooth sphere silhouette, which is what makes CG hair read as
    # hair rather than a helmet.
    strand_tint = tuple(min(255, int(c * 1.18)) for c in hair)
    for k in range(16):
        ang = k / 16.0 * 2 * math.pi
        lateral, depth = math.cos(ang), math.sin(ang)
        root = add(skel.head, apply(head_r, (lateral * HEAD_R * 0.92,
                                             0.10, depth * HEAD_R * 0.92)))
        tip_len = 0.05 + 0.035 * ((k * 7) % 5) / 4.0
        tip = add(root, apply(head_r, (lateral * 0.02, -tip_len, depth * 0.02)))
        color = strand_tint if k % 2 else hair
        _add_limb(buf, root, tip, 0.020, 0.004, color, 4, MAT_HAIR)

    # Arms: shoulder cap, sleeve, forearm and a real hand with fingers.
    for shoulder, elbow, wrist, side in (
            (skel.l_shoulder, skel.l_elbow, skel.l_wrist, -1),
            (skel.r_shoulder, skel.r_elbow, skel.r_wrist, 1)):
        _add_sphere(buf, shoulder, 0.105, outfit, rings=6, segments=10, mat=MAT_CLOTH)
        _add_limb(buf, shoulder, elbow, 0.095, 0.075, outfit, 8, MAT_CLOTH)
        _add_limb(buf, elbow, wrist, 0.072, 0.058, skin, 8, MAT_SKIN)
        hand_axis = norm(sub(wrist, elbow))
        palm_c = add(wrist, scale(hand_axis, 0.06))
        _add_box(buf, palm_c, (0.085, 0.10, 0.045),
                 rot_y(pose.turn), skin, MAT_SKIN)
        finger_base = add(palm_c, scale(hand_axis, 0.05))
        for f in range(4):
            off = (f - 1.5) * 0.022
            _add_limb(buf, add(finger_base, (off, 0.0, 0.0)),
                      add(finger_base, (off, -0.055, 0.005)),
                      0.011, 0.008, skin, 4, MAT_SKIN)
        thumb_base = add(palm_c, (side * 0.045, -0.01, 0.0))
        _add_limb(buf, thumb_base, add(thumb_base, (side * 0.02, -0.045, 0.01)),
                  0.012, 0.009, skin, 4, MAT_SKIN)

    # Legs: thigh, knee, shin and a shoe with a sole.
    for hip, knee, ankle in ((skel.l_hip, skel.l_knee, skel.l_ankle),
                             (skel.r_hip, skel.r_knee, skel.r_ankle)):
        _add_limb(buf, hip, knee, 0.128, 0.088, trouser, 9, MAT_CLOTH)
        _add_limb(buf, knee, ankle, 0.086, 0.062, trouser, 9, MAT_CLOTH)
        _add_box(buf, add(ankle, (0.0, 0.045, 0.03)), (0.13, 0.09, 0.26),
                 rot_y(pose.turn), shoe, MAT_CLOTH)
        _add_box(buf, add(ankle, (0.0, 0.012, 0.05)), (0.14, 0.03, 0.30),
                 rot_y(pose.turn), (28, 26, 32), MAT_MATTE)


def _face_features(skel: Skeleton, pal: dict, pose: Pose, buf: list[Tri]) -> None:
    """Eyes, brows and mouth as camera-facing layers on the head.

    Flat facial features are the standard stylised-3D shortcut and keep the
    face readable at any angle without a texture pipeline. The eyes are layered
    (socket shadow, sclera, iris, pupil, catchlight) with an upper lid and
    lashes that close on a blink; the mouth gets lips, teeth and a tongue.
    """
    hair, eye = pal["hair"], pal["eye"]
    head_rot = skel.head_rot
    forward = apply(head_rot, (0.0, 0.0, 1.0))
    right = apply(head_rot, (1.0, 0.0, 0.0))
    up = apply(head_rot, (0.0, 1.0, 0.0))
    skin = pal["skin"]
    shadow = tuple(int(c * 0.78) for c in skin)
    lash = tuple(int(c * 0.45) for c in hair)
    pupil = tuple(int(c * 0.42) for c in eye)

    def quad(center: Vec3, half_r: float, half_u: float, color, depth=0.0,
             mat=MAT_MATTE):
        c = add(center, scale(forward, depth))
        p0 = add(c, add(scale(right, -half_r), scale(up, -half_u)))
        p1 = add(c, add(scale(right, half_r), scale(up, -half_u)))
        p2 = add(c, add(scale(right, half_r), scale(up, half_u)))
        p3 = add(c, add(scale(right, -half_r), scale(up, half_u)))
        _add_quad(buf, p0, p1, p2, p3, color, mat)

    eye_y, eye_x = 0.035, 0.055
    openness = max(0.06, 1.0 - pose.blink)
    lower_lid = pose.squint * 0.012
    sclera = (222, 220, 226)
    for side in (-1, 1):
        center = add(skel.head, add(scale(right, side * eye_x), scale(up, eye_y)))
        # Socket shadow, then the sclera, iris, pupil and catchlight layers.
        quad(center, 0.052, 0.042 * openness, shadow, depth=HEAD_R * 0.885)
        quad(center, 0.040, 0.032 * openness, sclera, depth=HEAD_R * 0.90)
        quad(add(center, scale(forward, HEAD_R * 0.92)), 0.021, 0.024 * openness,
             eye, mat=MAT_EYE)
        quad(add(center, scale(forward, HEAD_R * 0.935)), 0.011, 0.013 * openness,
             pupil, mat=MAT_EYE)
        quad(add(center, add(scale(forward, HEAD_R * 0.95),
                             add(scale(right, side * 0.008), scale(up, 0.010)))),
             0.006, 0.007, (255, 255, 255), mat=MAT_EYE)
        # Upper lid + lash line: they travel down as the eye closes.
        lid_u = 0.032 * openness
        lid_c = add(center, add(scale(forward, HEAD_R * 0.905), scale(up, lid_u)))
        quad(lid_c, 0.044, 0.012, skin, mat=MAT_SKIN)
        lash_c = add(center, add(scale(forward, HEAD_R * 0.915), scale(up, lid_u)))
        quad(lash_c, 0.043, 0.005, lash)
        # Lower lid (raises on a squint) and an outer-corner lash.
        quad(add(center, add(scale(forward, HEAD_R * 0.905),
                             scale(up, -0.030 + lower_lid))),
             0.038, 0.006, tuple(int(c * 0.9) for c in skin), mat=MAT_SKIN)
        corner = add(center, add(scale(right, side * 0.042),
                                 scale(up, 0.014)))
        quad(corner, 0.010, 0.004, lash)
        # Brow: an angled bar (inner end lower for a frown, higher for a smile).
        tilt = pose.brow * 0.012 - pose.smile * 0.006
        brow_c = add(center, add(scale(up, 0.048 + pose.brow * 0.012),
                                 scale(right, side * 0.004)))
        _add_poly(buf, [
            add(brow_c, add(scale(right, -0.040), scale(up, -0.009 + tilt))),
            add(brow_c, add(scale(right, 0.040), scale(up, -0.007 - tilt))),
            add(brow_c, add(scale(right, 0.040), scale(up, 0.006 - tilt))),
            add(brow_c, add(scale(right, -0.040), scale(up, 0.008 + tilt))),
        ], tuple(int(c * 0.85) for c in hair))

    # Mouth: lips curve with the smile, open with the viseme, with teeth and
    # a tongue inside so an open mouth is not a black hole.
    mouth_c = add(skel.head, add(scale(up, -0.062), scale(forward, HEAD_R * 0.90)))
    mouth_c = add(mouth_c, scale(right, pose.jaw_side))
    open_h = 0.008 + 0.045 * pose.mouth
    wide = 0.042 * pose.mouth_wide
    smile = pose.smile
    lip = (150, 62, 72)
    _add_poly(buf, [
        add(mouth_c, add(scale(right, -wide), scale(up, smile * 0.014))),
        add(mouth_c, add(scale(right, wide), scale(up, smile * 0.014))),
        add(mouth_c, add(scale(right, wide * 0.9), scale(up, -0.014 + smile * 0.004))),
        add(mouth_c, add(scale(right, -wide * 0.9), scale(up, -0.014 + smile * 0.004))),
    ], lip)
    if open_h > 0.014:
        cavity = add(mouth_c, scale(forward, 0.004))
        quad(cavity, wide * 0.80, open_h, (74, 30, 36))
        quad(add(cavity, add(scale(forward, 0.002), scale(up, open_h * 0.5))),
             wide * 0.66, open_h * 0.30, (245, 245, 240))
        quad(add(cavity, add(scale(forward, 0.002), scale(up, -open_h * 0.55))),
             wide * 0.40, open_h * 0.25, (176, 78, 88))


# ---------------------------------------------------------------------------
# 3D set
# ---------------------------------------------------------------------------
def _env_palette(env: str):
    from .providers import _env_palette as palette  # lazy: avoids a cycle

    return palette(env)


def _build_set(env: str, buf: list[Tri], size: float = 14.0) -> None:
    """Floor, back/side walls and props — a room the camera sits in.

    Props are spread at several depths and heights so the perspective read is
    unmistakable and the shallow depth of field has something to fall off onto.
    """
    top, mid, accent = _env_palette(env)
    floor = tuple(min(255, int(c * 1.02)) for c in mid)
    wall = tuple(int(c * 0.95) for c in accent)
    wall2 = tuple(int(c * 0.78) for c in accent)
    dark = tuple(int(c * 0.55) for c in accent)
    light = tuple(min(255, int(c * 1.25)) for c in accent)

    _add_quad(buf, (-size, 0, -size), (size, 0, -size),
              (size, 0, size), (-size, 0, size), floor)
    # A ceiling closes the box: without it the walls stop mid-frame and the
    # room reads as a stage flat instead of an interior.
    _add_quad(buf, (-size, size, -size), (size, size, -size),
              (size, size, size), (-size, size, size),
              tuple(int(c * 0.85) for c in mid))
    _add_quad(buf, (-size, 0, -size), (size, 0, -size),
              (size, size, -size), (-size, size, -size), wall)
    _add_quad(buf, (-size, 0, -size), (-size, 0, size),
              (-size, size, size), (-size, size, -size), wall2)
    _add_quad(buf, (size, 0, -size), (size, 0, size),
              (size, size, size), (size, size, -size), wall2)
    # A rug gives the floor a focal plane and a colour anchor.
    _add_quad(buf, (-1.1, 0.004, 0.9), (1.1, 0.004, 0.9),
              (1.1, 0.004, -0.9), (-1.1, 0.004, -0.9),
              tuple(min(255, int(c * 0.85)) for c in accent), MAT_MATTE)
    # Wall art breaks the flat back wall.
    _add_box(buf, (-0.2, 1.5, -2.6), (0.9, 0.7, 0.05), rot_y(0.0), light)
    _add_box(buf, (1.6, 1.4, -2.6), (0.6, 0.5, 0.05), rot_y(0.0), dark)

    # Props: a desk, a cabinet, a lamp and a low bench at different depths.
    _add_box(buf, (1.45, 0.35, -0.9), (1.1, 0.70, 0.7), rot_y(0.25), accent)
    _add_box(buf, (1.45, 0.71, -0.9), (1.2, 0.05, 0.78), rot_y(0.25), light)
    _add_box(buf, (-1.55, 0.45, -1.1), (0.55, 0.90, 0.55), rot_y(-0.2), dark)
    _add_box(buf, (-1.55, 0.95, -1.1), (0.75, 0.10, 0.75), rot_y(-0.2), light)
    _add_box(buf, (2.1, 0.12, 1.4), (0.9, 0.24, 0.6), rot_y(0.4), dark)
    _add_box(buf, (-2.2, 0.55, 0.6), (0.10, 1.10, 0.10), rot_y(0.0), dark)
    _add_sphere(buf, (-2.2, 1.20, 0.6), 0.22, light, rings=6, segments=12,
                squash=(1.0, 0.7, 1.0), mat=MAT_MATTE)


def _ground_shadow(skel: Skeleton, buf: list[Tri]) -> None:
    """Flat disc under the character so it sits on the floor instead of floating."""
    center = (skel.hip[0], 0.01, skel.hip[2])
    _add_sphere(buf, center, 0.30, (26, 26, 34), rings=2, segments=10,
                squash=(1.0, 0.02, 0.7))


# ---------------------------------------------------------------------------
# Camera + rasteriser
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    eye: Vec3 = (0.0, 1.16, 3.60)
    target: Vec3 = (0.0, 0.98, 0.0)
    fov: float = 0.72


def _camera_basis(cam: Camera):
    forward = norm(sub(cam.target, cam.eye))
    right = norm(cross(forward, (0.0, 1.0, 0.0)))
    up = cross(right, forward)
    return forward, right, up


def _camera_for(camera: str, t: float) -> Camera:
    """Scene camera move, in 3D (the camera really travels).

    Every shot gets a slow dolly or arc so no frame is a still image; the
    movement is deliberately small (centimetres) — that is what a real
    operator does, and it reads as "filmed" rather than "rendered".
    """
    cam = Camera()
    drift = 0.05 * math.sin(2 * math.pi * t / 6.0)
    if camera == "tracking":
        cam.eye = (0.70 * math.sin(2 * math.pi * t / 4.0), 1.18, 3.30)
        cam.target = (0.18 * math.sin(2 * math.pi * t / 4.0), 0.98, 0.0)
    elif camera == "close_up":
        cam.eye = (drift, 1.34, 2.05 - 0.10 * math.sin(2 * math.pi * t / 5.0))
        cam.target = (0.0, 1.24, 0.0)
        cam.fov = 0.62
    elif camera == "over_the_shoulder":
        cam.eye = (0.90 + drift, 1.28, 2.40)
        cam.target = (0.0, 1.10, 0.0)
    elif camera == "wide":
        cam.eye = (0.0, 1.25, 4.60 + 0.15 * math.sin(2 * math.pi * t / 7.0))
        cam.target = (0.0, 0.92, 0.0)
    elif camera == "zoom_in":
        cam.eye = (drift, 1.18, 3.95 - 1.0 * t)
        cam.target = (0.0, 1.02, 0.0)
    elif camera == "zoom_out":
        cam.eye = (drift, 1.18, 2.75 + 1.0 * t)
        cam.target = (0.0, 1.02, 0.0)
    elif camera == "dolly":
        cam.eye = (0.25, 1.22, 3.6 - 0.9 * t)
        cam.target = (0.0, 1.05, 0.0)
    elif camera == "orbit":
        angle = 0.55 * math.sin(2 * math.pi * t / 8.0)
        cam.eye = (3.3 * math.sin(angle), 1.25, 3.3 * math.cos(angle))
        cam.target = (0.0, 1.02, 0.0)
    else:  # medium and anything unknown
        cam.eye = (drift, 1.16 + 0.02 * math.sin(2 * math.pi * t / 5.0), 3.60)
        cam.target = (0.0, 0.98, 0.0)
    return cam


def _head_screen_y(scene, t: float, cam: Camera, height: int) -> float | None:
    """Where the head lands vertically (0 = top, 1 = bottom) — focus target."""
    pose = pose_at(scene.character_action, scene.emotion, t, False, 0.0, 1.0)
    skel = solve(pose)
    forward, _, up = _camera_basis(cam)
    d = sub(skel.head, cam.eye)
    zc = dot(d, forward)
    if zc <= 0.25:
        return None
    f = 0.5 * height / math.tan(cam.fov / 2)
    return (height / 2.0 - f * dot(d, up) / zc) / height


# ---------------------------------------------------------------------------
# Render settings: every knob that shapes the look, with named presets
# ---------------------------------------------------------------------------
@dataclass
class RenderSettings:
    """All the quality/look parameters of the software 3D engine.

    Presets give a one-word starting point; every field stays individually
    overridable, and `from_env` lets a deployment tune them without code.
    """

    # Geometry / sampling
    scale: float = 0.85         # supersampling factor (1.0 = no downscale)
    fps: int = 30
    # Lighting
    ambient: float = 0.30
    key_intensity: float = 1.00
    fill_intensity: float = 0.62
    rim_intensity: float = 0.70
    specular: float = 1.00       # multiplier on per-material specular
    # Post-processing
    bloom: bool = True
    bloom_threshold: int = 196
    bloom_strength: float = 0.30
    bloom_radius: float = 2.5
    dof: bool = True
    dof_strength: float = 1.00
    dof_focus_y: float = 0.55
    grade: bool = True
    saturation: float = 1.08
    warm: float = 1.06
    cool: float = 0.97
    vignette: bool = True
    vignette_strength: float = 0.32
    grain: bool = True
    grain_amount: float = 0.035
    sharpen: float = 0.0         # unsharp-mask amount (lens micro-contrast)
    chroma: float = 0.0          # lateral chromatic aberration in pixels
    tone_map: bool = True        # filmic highlight roll-off instead of clipping
    # Motion
    camera_shake: float = 0.0    # handheld wobble amplitude in metres
    motion_blur: bool = False    # average two sub-frames (cheap motion blur)

    @classmethod
    def preset(cls, name: str) -> "RenderSettings":
        presets = {
            "draft": cls(scale=0.5, bloom=False, dof=False, grain=False,
                         vignette=False, fps=24),
            "standard": cls(),
            "cinematic": cls(scale=1.15, bloom_strength=0.34, bloom_radius=3.0,
                             saturation=1.12, vignette_strength=0.38,
                             grain_amount=0.05, camera_shake=0.006,
                             motion_blur=True),
            "photoreal": cls(scale=1.25, ambient=0.24, specular=1.25,
                             bloom_strength=0.40, bloom_radius=3.4,
                             dof_strength=1.25, saturation=1.05, warm=1.08,
                             cool=0.94, vignette_strength=0.34,
                             grain_amount=0.045, camera_shake=0.004,
                             motion_blur=True, sharpen=0.55, chroma=1.2),
            "anime": cls(scale=0.85, bloom_strength=0.42, saturation=1.25,
                         ambient=0.38, specular=0.5, dof=False,
                         vignette_strength=0.20, grain=False),
        }
        return presets.get((name or "").strip().lower(), presets["standard"])

    @classmethod
    def from_env(cls) -> "RenderSettings":
        """Start from the `REELFORGE_3D_PRESET` preset, then apply overrides."""
        settings = cls.preset(os.environ.get("REELFORGE_3D_PRESET", "standard"))
        env_map = {
            "REELFORGE_3D_SCALE": ("scale", float),
            "REELFORGE_3D_AMBIENT": ("ambient", float),
            "REELFORGE_3D_SATURATION": ("saturation", float),
            "REELFORGE_3D_BLOOM_STRENGTH": ("bloom_strength", float),
            "REELFORGE_3D_VIGNETTE": ("vignette_strength", float),
            "REELFORGE_3D_GRAIN": ("grain_amount", float),
            "REELFORGE_3D_SHAKE": ("camera_shake", float),
        }
        for key, (field_name, cast) in env_map.items():
            raw = os.environ.get(key)
            if raw:
                try:
                    setattr(settings, field_name, cast(raw))
                except ValueError:
                    pass
        return settings


_DEFAULT_SETTINGS = RenderSettings()


# Three-point lighting: a warm key, a cool fill and a rim/back light. The
# colour split is what reads as "photographed" instead of "flat shaded".
KEY_DIR = norm((-0.40, 0.78, 0.62))
KEY_COLOR = (255.0, 246.0, 228.0)
FILL_DIR = norm((0.70, 0.25, 0.55))
FILL_COLOR = (150.0, 180.0, 225.0)
RIM_DIR = norm((0.25, 0.35, -0.90))
RIM_COLOR = (200.0, 225.0, 255.0)
AMBIENT = (0.24, 0.25, 0.30)


def _shade(color, normal, view_dir, spec: float = 0.0, shin: float = 24.0,
           ambient: float = 0.30, key: float = 1.0, fill: float = 0.62,
           rim: float = 0.70) -> tuple[int, int, int]:
    """Lambert diffuse from three lights + Blinn-Phong specular + ambient."""
    out = [color[i] * ambient for i in range(3)]

    for direction, light, intensity in ((KEY_DIR, KEY_COLOR, key),
                                        (FILL_DIR, FILL_COLOR, fill),
                                        (RIM_DIR, RIM_COLOR, rim)):
        lam = max(0.0, dot(normal, direction))
        if lam <= 0.0:
            continue
        lam *= intensity
        for i in range(3):
            out[i] += color[i] * lam * (light[i] / 255.0)

    if spec > 0.0:
        half = norm(add(KEY_DIR, view_dir))
        highlight = max(0.0, dot(normal, half)) ** shin
        glow = 255.0 * spec * highlight
        for i in range(3):
            out[i] += glow * (KEY_COLOR[i] / 255.0)

    for i in range(3):
        out[i] += color[i] * AMBIENT[i]
    return (min(255, int(out[0])), min(255, int(out[1])), min(255, int(out[2])))


def render_frame(cam: Camera, tris: list[Tri], width: int, height: int,
                 background: Image.Image | None = None,
                 settings: "RenderSettings | None" = None) -> Image.Image:
    """Project, shade, depth-sort and fill every triangle (painter's algorithm)."""
    forward, right, up = _camera_basis(cam)
    eye = cam.eye
    f = 0.5 * height / math.tan(cam.fov / 2)
    rs = settings or _DEFAULT_SETTINGS
    spec_scale = rs.specular
    cx, cy = width / 2.0, height / 2.0

    img = (background.copy() if background is not None
           else Image.new("RGB", (width, height), (24, 26, 38)))
    draw = ImageDraw.Draw(img)

    polys: list[tuple[float, tuple, tuple[int, int, int]]] = []
    for tri in tris:
        pts = []
        depth = 0.0
        ok = True
        for vertex in (tri.a, tri.b, tri.c):
            d = sub(vertex, eye)
            zc = dot(d, forward)
            if zc < 0.25:
                ok = False
                break
            xc = dot(d, right)
            yc = dot(d, up)
            pts.append((cx + f * xc / zc, cy - f * yc / zc))
            depth += zc
        if not ok:
            continue
        n = cross(sub(tri.b, tri.a), sub(tri.c, tri.a))
        if dot(n, forward) >= 0:  # back-facing
            continue
        centroid = ((tri.a[0] + tri.b[0] + tri.c[0]) / 3.0,
                    (tri.a[1] + tri.b[1] + tri.c[1]) / 3.0,
                    (tri.a[2] + tri.b[2] + tri.c[2]) / 3.0)
        view_dir = norm(sub(eye, centroid))
        polys.append((depth / 3.0, tuple(pts),
                      _shade(tri.color, norm(n), view_dir, tri.spec * spec_scale,
                             tri.shin, rs.ambient, rs.key_intensity,
                             rs.fill_intensity, rs.rim_intensity)))

    polys.sort(key=lambda item: item[0], reverse=True)
    for _, pts, color in polys:
        draw.polygon(pts, fill=color)

    return img


def _downsample(img: Image.Image, width: int, height: int) -> Image.Image:
    """Supersampled frames are averaged down — that is the anti-aliasing."""
    return img.resize((width, height), Image.LANCZOS)


class FilmLook:
    """Camera-and-lens post-processing, with the static masks built once.

    Bloom, shallow depth of field, colour grading, vignetting and sensor grain
    are what separate "a 3D model" from "a photograph of a 3D model". The masks
    that do not change between frames (focus and vignette) are precomputed so
    the per-frame cost stays low.
    """

    def __init__(self, width: int, height: int,
                 settings: "RenderSettings | None" = None) -> None:
        rs = settings or _DEFAULT_SETTINGS
        self.width, self.height = width, height
        self.rs = rs
        self.bloom_on, self.dof_on = rs.bloom, rs.dof
        self.grade_on = rs.grade
        self.vignette_on, self.grain_on = rs.vignette, rs.grain
        self.sharpen_on = rs.sharpen > 0.0
        self.chroma_on = rs.chroma > 0.0
        self.dof_mask = self._focus_mask(rs.dof_focus_y) if rs.dof else None
        self.vignette_mask = self._vignette_mask() if rs.vignette else None
        self.noise = self._noise() if rs.grain else None
        self._noise_index = 0

    def _chroma_shift(self, img: Image.Image) -> Image.Image:
        """Lateral chromatic aberration: split the red and blue planes."""
        dx = max(1, int(round(self.rs.chroma)))
        r, g, b = img.split()
        r = r.transform(r.size, Image.AFFINE, (1, 0, dx, 0, 1, 0))
        b = b.transform(b.size, Image.AFFINE, (1, 0, -dx, 0, 1, 0))
        return Image.merge("RGB", (r, g, b))

    def _focus_mask(self, focus_y: float) -> Image.Image:
        # Built as a 1-pixel-wide column then stretched: the mask only varies
        # vertically, so a full-resolution Python loop would be pure waste.
        column = Image.new("L", (1, self.height))
        px = column.load()
        for y in range(self.height):
            d = abs(y / self.height - focus_y) / 0.5
            px[0, y] = int(min(255, max(0.0, (d - 0.35) / 0.65) * 255))
        return column.resize((self.width, self.height), Image.NEAREST)

    def _vignette_mask(self) -> Image.Image:
        # Rendered at 1/8 scale (the falloff is smooth) and scaled up.
        small_w, small_h = max(2, self.width // 8), max(2, self.height // 8)
        mask = Image.new("L", (small_w, small_h), 255)
        px = mask.load()
        cx, cy = small_w / 2.0, small_h / 2.0
        max_d = math.hypot(cx, cy)
        for y in range(small_h):
            for x in range(small_w):
                d = math.hypot(x - cx, y - cy) / max_d
                px[x, y] = int(255 * (1.0 - 0.32 * d * d))
        return mask.resize((self.width, self.height), Image.BILINEAR)

    def _noise(self) -> list[Image.Image]:
        import random

        rnd = random.Random(7)
        small_w, small_h = max(2, self.width // 4), max(2, self.height // 4)
        tiles = []
        for _ in range(3):
            tile = Image.new("L", (small_w, small_h))
            tile.putdata([rnd.randint(0, 255) for _ in range(small_w * small_h)])
            tiles.append(tile.resize((self.width, self.height),
                                     Image.BILINEAR))
        return tiles

    def _bloom(self, img: Image.Image) -> Image.Image:
        # The glow is blurred at quarter resolution then scaled back up: a
        # Gaussian that wide is low-frequency, so full-res blurring is waste.
        small_w, small_h = max(2, self.width // 4), max(2, self.height // 4)
        small = img.resize((small_w, small_h), Image.BILINEAR)
        gray = small.convert("L")
        mask = gray.point(lambda v: 255 if v >= self.rs.bloom_threshold else 0)
        bright = Image.composite(small, Image.new("RGB", small.size, (0, 0, 0)), mask)
        blurred = bright.filter(ImageFilter.GaussianBlur(self.rs.bloom_radius))
        blurred = blurred.resize((self.width, self.height), Image.BILINEAR)
        return Image.blend(img, blurred, self.rs.bloom_strength)

    def _grade(self, img: Image.Image) -> Image.Image:
        from PIL import ImageEnhance

        r, g, b = img.split()
        rs = self.rs
        # Filmic highlight roll-off: bright values compress toward white
        # instead of hard-clipping, so lit skin keeps its modelling.
        if rs.tone_map:
            def _roll(v: int) -> int:
                x = v / 255.0
                aces = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
                y = 0.3 * x + 0.7 * aces       # blend toward the filmic curve
                return min(255, int(round(255.0 * y)))
            tone = [_roll(v) for v in range(256)]
            r, g, b = r.point(tone), g.point(tone), b.point(tone)

        def curve(channel, lo, hi):
            lut = [min(255, int((lo + (v / 255.0) * hi) * 255)) for v in range(256)]
            return channel.point(lut)

        merged = Image.merge("RGB", (curve(r, 0.02, rs.warm),
                                      curve(g, 0.025, (rs.warm + rs.cool) / 2),
                                      curve(b, 0.05, rs.cool)))
        return ImageEnhance.Color(merged).enhance(rs.saturation)

    def apply(self, img: Image.Image) -> Image.Image:
        rs = self.rs
        if self.bloom_on:
            img = self._bloom(img)
        if self.dof_on and self.dof_mask is not None:
            small_w, small_h = max(2, self.width // 4), max(2, self.height // 4)
            small = img.resize((small_w, small_h), Image.BILINEAR)
            blurred = small.filter(ImageFilter.GaussianBlur(1.5 * rs.dof_strength))
            blurred = blurred.resize((self.width, self.height), Image.BILINEAR)
            img = Image.composite(blurred, img, self.dof_mask)
        if self.grade_on:
            img = self._grade(img)
        if self.chroma_on:
            img = self._chroma_shift(img)
        if self.sharpen_on:
            img = img.filter(ImageFilter.UnsharpMask(
                radius=1.6, percent=int(rs.sharpen * 100), threshold=3))
        if self.vignette_on and self.vignette_mask is not None:
            img = Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)),
                                  self.vignette_mask)
        if self.grain_on and self.noise:
            tile = self.noise[self._noise_index % len(self.noise)]
            self._noise_index += 1
            img = Image.blend(img, Image.merge("RGB", (tile, tile, tile)),
                              rs.grain_amount)
        return img


def _background(env: str, width: int, height: int) -> Image.Image:
    """Sky/room gradient with a soft light bloom behind the subject."""
    top, mid, _ = _env_palette(env)
    img = Image.new("RGB", (width, height), top)
    draw = ImageDraw.Draw(img)
    for y in range(0, height, 4):
        frac = y / height
        color = tuple(int(top[i] + (mid[i] - top[i]) * frac) for i in range(3))
        draw.rectangle((0, y, width, y + 4), fill=color)
    # A cheap radial glow where the key light sits: it fakes environmental
    # bounce and gives the shallow DOF a bright background to dissolve into.
    glow = Image.new("L", (max(2, width // 8), max(2, height // 8)), 0)
    gpx = glow.load()
    gx, gy = glow.size[0] * 0.28, glow.size[1] * 0.24
    for y in range(glow.size[1]):
        for x in range(glow.size[0]):
            d = math.hypot((x - gx) / glow.size[0], (y - gy) / glow.size[1])
            gpx[x, y] = int(max(0.0, 1.0 - d * 2.2) * 120)
    glow = glow.resize((width, height), Image.BILINEAR)
    bright = Image.new("RGB", (width, height), (255, 240, 214))
    return Image.composite(bright, img, glow)


# ---------------------------------------------------------------------------
# Scene rendering
# ---------------------------------------------------------------------------
def _spans_lookup(spans):
    return [(text, float(start), float(end)) for text, start, end in spans]


def render_frame_for(profile, scene, spans, t: float, width: int, height: int,
                     background: Image.Image | None = None,
                     settings: "RenderSettings | None" = None,
                     static_tris: list[Tri] | None = None) -> Image.Image:
    """One full 3D frame: set + rigged character + facial features.

    `static_tris` is the prebuilt set geometry; it does not change over a scene
    so the caller builds it once and reuses it every frame.
    """
    rs = settings or _DEFAULT_SETTINGS
    timed = _spans_lookup(spans)
    talking = any(s - 0.05 <= t <= e + 0.05 for _, s, e in timed)
    word = ""
    for text, s, e in timed:
        if s - 0.02 <= t <= e + 0.02:
            word = text
            break
    shape = avatars.viseme(word) if talking else "neutral"
    mouth, wide = _mouth_for(shape, talking)
    blink = 1.0 if (t % 4.0) > 3.84 else 0.0
    pose = pose_at(scene.character_action, scene.emotion, t, talking, mouth, wide)
    pose.blink = blink

    pal = profile.palette()
    tris: list[Tri] = list(static_tris) if static_tris is not None else []
    if static_tris is None:
        _build_set(scene.environment, tris)
    skel = solve(pose)
    _ground_shadow(skel, tris)
    _character_mesh(skel, pal, pose, tris)
    _face_features(skel, pal, pose, tris)
    cam = _camera_for(scene.camera, t)
    if rs.camera_shake:
        # Handheld wobble: two incommensurate sines so it never visibly loops.
        amp = rs.camera_shake
        cam = Camera(
            eye=add(cam.eye, (amp * math.sin(t * 11.3),
                              amp * 0.6 * math.sin(t * 8.1 + 1.3), 0.0)),
            target=cam.target, fov=cam.fov)
    return render_frame(cam, tris, width, height, background, rs)


_MOUTH_SHAPES = {
    "closed": (0.05, 1.05), "a": (1.00, 0.95), "e": (0.55, 1.15),
    "i": (0.30, 1.25), "o": (0.80, 0.75), "u": (0.55, 0.65),
    "f": (0.25, 1.05), "neutral": (0.12, 0.95),
}


def _mouth_for(shape: str, talking: bool) -> tuple[float, float]:
    if not talking:
        return 0.0, 0.9
    return _MOUTH_SHAPES.get(shape, (0.3, 1.0))


def render_scene_3d(profile, scene, spans, out_path: Path, *,
                    width: int = WIDTH, height: int = HEIGHT,
                    scale: float | None = None,
                    settings: "RenderSettings | None" = None) -> dict:
    """Render one scene to H.264, frame by frame, and return its metadata.

    Frames are rendered at `scale` x the output size and averaged down (that
    is the anti-aliasing), then run through the `FilmLook` pass. Every look
    knob comes from `settings` (see `RenderSettings` / its presets).
    """
    rs = settings or _DEFAULT_SETTINGS
    if scale is not None:
        rs.scale = scale
    fps = rs.fps or FPS
    rw = max(180, int(width * rs.scale) // 2 * 2)
    rh = max(320, int(height * rs.scale) // 2 * 2)
    duration = max(0.5, float(scene.duration))
    frames = max(1, int(round(duration * fps)))
    background = _background(scene.environment, rw, rh)
    # Autofocus: put the focal plane on the character's head, wherever the
    # shot places it, so the face is always the sharpest thing in frame.
    samples = [y for y in (
        _head_screen_y(scene, i / fps, _camera_for(scene.camera, i / fps), rh)
        for i in range(0, frames, max(1, frames // 8))) if y is not None]
    if samples:
        rs.dof_focus_y = min(0.92, max(0.08, sum(samples) / len(samples)))
    look = FilmLook(width, height, rs)
    static_tris: list[Tri] = []
    _build_set(scene.environment, static_tris)

    from .providers import encoder_args

    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-",
         *encoder_args(), str(out_path)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None

    def _frame(ts: float) -> Image.Image:
        raw = render_frame_for(profile, scene, spans, ts, rw, rh, background, rs,
                               static_tris)
        return _downsample(raw, width, height)

    try:
        for index in range(frames):
            t = index / fps
            if rs.motion_blur:
                # Average two sub-frames a half-step apart: a cheap, convincing
                # motion blur without a full temporal accumulation buffer.
                a = _frame(t)
                b = _frame(t + 0.5 / fps)
                frame = Image.blend(a, b, 0.5)
            else:
                frame = _frame(t)
            proc.stdin.write(look.apply(frame).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0 or not out_path.exists():
        from .providers import AnimationFailed

        raise AnimationFailed(scene.scene_id, "local", "3d ffmpeg encode failed")
    return {"camera": scene.camera, "environment": scene.environment,
            "frames": frames, "animated": True, "engine": "software_3d",
            "render_size": f"{rw}x{rh}", "post": "film_look",
            "settings": rs.__dict__.copy()}
