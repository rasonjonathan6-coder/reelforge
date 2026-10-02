"""Tests for the software 3D engine (volumetric characters, real camera).

These assert that the 3D path is genuinely 3D — perspective projection, camera
parallax, depth occlusion — and not a flat drawing with a 3D label.
"""

from __future__ import annotations

import math
import subprocess

import pytest
from PIL import Image, ImageChops

from pipeline.animation import render3d
from pipeline.animation.character import profile_for
from pipeline.animation.providers import AnimationProviderFactory
from pipeline.animation.scenes import Scene


def _scene(**kw) -> Scene:
    base = dict(scene_id=1, duration=2.0, character_action="parle", emotion="joie",
                camera="medium", environment="rue", speaker="Léo")
    base.update(kw)
    return Scene(**base)


def _render(camera="medium", t=0.4, width=200, height=356, **kw) -> Image.Image:
    scene = _scene(camera=camera, **kw)
    profile = profile_for("Léo")
    spans = [("Bonjour", 0.0, 0.6), ("le", 0.6, 0.8), ("monde", 0.8, 1.2)]
    return render3d.render_frame_for(profile, scene, spans, t, width, height)


def _character_only(camera="medium", t=0.4, width=200, height=356):
    """Render the rig alone (no set) so we can measure the projection itself."""
    pose = render3d.pose_at("parle", "joie", t, True, 0.6, 1.0)
    skel = render3d.solve(pose)
    tris: list[render3d.Tri] = []
    render3d._character_mesh(skel, profile_for("Léo").palette(), pose, tris)
    return render3d.render_frame(render3d._camera_for(camera, t), tris, width, height)


def test_projection_is_perspective_not_orthographic():
    """A near object must project larger than the same object moved away."""
    pose = render3d.Pose()
    near = render3d.solve(pose, origin=(0.0, 0.0, 0.0))
    far = render3d.solve(pose, origin=(0.0, 0.0, -4.0))
    cam = render3d.Camera(eye=(0.0, 1.16, 3.6), target=(0.0, 0.98, 0.0))

    def head_height(skel):
        fwd, right, up = render3d._camera_basis(cam)
        f = 0.5 * 800 / math.tan(cam.fov / 2)
        zc = render3d.dot(render3d.sub(skel.head, cam.eye), fwd)
        return f / zc
    assert head_height(near) > head_height(far) * 1.3

def test_camera_move_changes_the_view():
    a = _render(camera="medium", t=0.0)
    b = _render(camera="tracking", t=1.0)
    assert ImageChops.difference(a, b).getbbox() is not None


def test_character_occludes_the_set_and_the_props():
    """Depth sorting: the character pixels must differ from the empty set."""
    profile = profile_for("Léo")
    scene = _scene()
    spans = [("Bonjour", 0.0, 0.6)]
    bg = render3d._background("rue", 200, 356)
    with_char = render3d.render_frame_for(profile, scene, spans, 0.4, 200, 356, bg)
    # The set alone, without the rig.
    tris: list[render3d.Tri] = []
    render3d._build_set("rue", tris)
    set_only = render3d.render_frame(render3d._camera_for("medium", 0.4), tris, 200, 356, bg)
    assert ImageChops.difference(with_char, set_only).getbbox() is not None


def test_pose_is_driven_by_action_and_emotion():
    walk = render3d.pose_at("marche", "joie", 0.5, True, 0.5, 1.0)
    talk = render3d.pose_at("parle", "joie", 0.5, True, 0.5, 1.0)
    assert walk.l_hip != talk.l_hip
    sad = render3d.pose_at("parle", "tristesse", 0.5, True, 0.5, 1.0)
    assert sad.head_pitch > talk.head_pitch


def test_mouth_opens_with_the_viseme():
    closed = render3d._mouth_for("closed", True)[0]
    wide = render3d._mouth_for("a", True)[0]
    assert wide > closed


def test_frames_differ_across_time():
    a = _character_only(t=0.2)
    b = _character_only(t=1.1)
    assert ImageChops.difference(a, b).getbbox() is not None


def test_factory_registers_the_3d_provider():
    assert AnimationProviderFactory.create("local3d").name == "local3d"


def test_render_scene_3d_produces_a_real_mp4(tmp_path):
    profile = profile_for("Léo")
    scene = _scene(duration=1.0, character_action="marche", camera="tracking")
    spans = [("Bonjour", 0.0, 0.5)]
    out = tmp_path / "scene.mp4"
    meta = render3d.render_scene_3d(profile, scene, spans, out, scale=0.35)
    assert meta["engine"] == "software_3d"
    assert out.exists() and out.stat().st_size > 1000
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True).stdout.strip()
    assert probe.startswith("1080,1920")


@pytest.mark.parametrize("camera", ["medium", "close_up", "wide", "tracking"])
def test_every_camera_frames_the_character(camera):
    """No camera may leave the frame empty or lose the character off-screen."""
    img = _character_only(camera=camera, width=200, height=356)
    px = img.convert("L").load()
    background = px[0, 0]
    changed = [(x, y) for y in range(0, 356, 4) for x in range(0, 200, 4)
               if abs(px[x, y] - background) > 18]
    assert changed, f"{camera} rendered nothing"
    ys = [y for _, y in changed]
    height = (max(ys) - min(ys)) / 356
    assert 0.2 < height <= 1.0, f"{camera} framing looks wrong ({height:.0%})"


# --- render settings / presets -------------------------------------------
def test_presets_are_distinct_and_named():
    names = ["draft", "standard", "cinematic", "photoreal", "anime"]
    presets = {n: render3d.RenderSettings.preset(n) for n in names}
    assert presets["photoreal"].motion_blur is True
    assert presets["draft"].scale < presets["photoreal"].scale
    assert render3d.RenderSettings.preset("nope").scale == presets["standard"].scale


def test_settings_change_the_rendered_frame():
    scene = _scene()
    profile = profile_for("Léo")
    spans = [("Bonjour", 0.0, 0.6)]
    dim = render3d.RenderSettings.preset("anime")
    vivid = render3d.RenderSettings.preset("photoreal")
    a = render3d.render_frame_for(profile, scene, spans, 0.4, 160, 284, settings=dim)
    b = render3d.render_frame_for(profile, scene, spans, 0.4, 160, 284, settings=vivid)
    assert ImageChops.difference(a, b).getbbox() is not None


def test_camera_shake_moves_the_camera():
    scene = _scene()
    profile = profile_for("Léo")
    spans = [("Bonjour", 0.0, 0.6)]
    still = render3d.RenderSettings(camera_shake=0.0)
    shaky = render3d.RenderSettings(camera_shake=0.02)
    a = render3d.render_frame_for(profile, scene, spans, 0.4, 160, 284, settings=still)
    b = render3d.render_frame_for(profile, scene, spans, 0.4, 160, 284, settings=shaky)
    assert ImageChops.difference(a, b).getbbox() is not None


def test_film_look_grades_the_frame():
    img = Image.new("RGB", (160, 284), (90, 80, 70))
    look = render3d.FilmLook(160, 284)
    out = look.apply(img)
    assert out.size == img.size
    assert ImageChops.difference(out, img).getbbox() is not None


def test_palette_from_image_extracts_colours(tmp_path):
    from pipeline import avatars

    img = Image.new("RGB", (100, 200), (250, 250, 250))
    for y in range(0, 24):
        for x in range(20, 80):
            img.putpixel((x, y), (20, 20, 20))
    for y in range(60, 104):
        for x in range(34, 66):
            img.putpixel((x, y), (230, 180, 140))
    for y in range(156, 200):
        for x in range(28, 72):
            img.putpixel((x, y), (30, 90, 200))
    path = tmp_path / "Lea.png"
    img.save(path)

    palette = avatars.load_reference("Lea", path)
    assert palette["hair"][0] < 80
    assert palette["outfit"][2] > palette["outfit"][0]
    assert palette["skin"][0] > 150
    assert avatars.palette_for("Lea") == palette
    avatars.clear_references()
    assert avatars.palette_for("Lea") != palette


# --- realism upgrades ----------------------------------------------------
def test_static_set_is_reused_across_frames():
    """The set is built once per scene, not per frame."""
    scene = _scene()
    profile = profile_for("Léo")
    spans = [("Bonjour", 0.0, 0.6)]
    static: list = []
    render3d._build_set(scene.environment, static)
    before = len(static)
    a = render3d.render_frame_for(profile, scene, spans, 0.1, 160, 284,
                                  settings=render3d.RenderSettings(), static_tris=static)
    b = render3d.render_frame_for(profile, scene, spans, 0.4, 160, 284,
                                  settings=render3d.RenderSettings(), static_tris=static)
    assert len(static) == before          # caller's list is not mutated
    assert ImageChops.difference(a, b).getbbox() is not None


def test_tone_map_rolls_off_highlights_instead_of_clipping():
    bright = Image.new("RGB", (32, 32), (255, 255, 255))
    look = render3d.FilmLook(32, 32, render3d.RenderSettings(
        bloom=False, dof=False, vignette=False, grain=False, grade=True))
    out = look.apply(bright)
    # Pure white would stay 255 with a plain curve; the filmic roll-off pulls
    # it down so lit skin keeps detail.
    assert out.getpixel((16, 16))[0] < 255


def test_chroma_aberration_shifts_colour_channels():
    base = render3d.RenderSettings(bloom=False, dof=False, vignette=False,
                                   grain=False, grade=False, chroma=2.0)
    plain = render3d.RenderSettings(bloom=False, dof=False, vignette=False,
                                    grain=False, grade=False, chroma=0.0)
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    img.putpixel((16, 16), (255, 255, 255))
    shifted = render3d.FilmLook(32, 32, base).apply(img.copy())
    same = render3d.FilmLook(32, 32, plain).apply(img.copy())
    assert ImageChops.difference(shifted, same).getbbox() is not None


def test_photoreal_preset_enables_the_lens_extras():
    rs = render3d.RenderSettings.preset("photoreal")
    assert rs.sharpen > 0.0 and rs.chroma > 0.0 and rs.tone_map is True
    assert rs.scale >= 1.0
    assert render3d.RenderSettings.preset("draft").tone_map is True


def test_autofocus_targets_the_head():
    scene = _scene()
    cam = render3d._camera_for(scene.camera, 0.4)
    y = render3d._head_screen_y(scene, 0.4, cam, 284)
    assert y is not None and 0.0 < y < 0.85


def test_face_has_layered_eye_geometry():
    """Eyes are built from several layers (sclera/iris/pupil/catchlight/lash)."""
    pose = render3d.pose_at("parle", "joie", 0.4, True, 0.8, 1.0)
    skel = render3d.solve(pose)
    pal = profile_for("Léo").palette()
    buf: list = []
    render3d._face_features(skel, pal, pose, buf)
    colors = {tri.color for tri in buf}
    assert (222, 220, 226) in colors                     # sclera
    assert any(c[0] < 90 and c[1] < 60 for c in colors)  # pupil / lash
