"""Render fixed, reproducible FB/BFB/RFB examples through the actual benchmark evaluators.

Dynamics change between independent episodes. No example is selected by success. The first evaluation context,
first dynamics instance and seed-0 checkpoint are used; grid displays goal 3, continuous diagnostics goal 1.
AntWind displays wind IDs 0, 16 and 19, always the first episode. JSON companions retain displayed trajectories,
checkpoint hashes and full evaluation scores. Model weights and official AntWind data are needed to regenerate.

MUJOCO_GL=egl uv run --extra mujoco scripts/make_dynamics_gifs.py --suite all
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageColor, ImageDraw, ImageFont

from bfb.affine_adaptation import fit_dynamics
from bfb.continuous import ContinuousFBAgent
from bfb.continuous_evaluation import GOALS, evaluate_continuous
from bfb.envs.fourrooms import START_POS, eval_goals
from bfb.envs.grid_suites import grid_suite
from bfb.envs.pointmass import contexts, generate_dataset
from bfb.evaluation import evaluate
from bfb.experiment import ROOT, atomic_json, file_sha256, load_agent

METHODS = ("fb", "bfb", "rfb")
COLORS = {"fb": "#64748b", "bfb": "#167fb8", "rfb": "#df8535"}
INK, MUTED, GREEN = "#18283e", "#596d85", "#159577"
WIDTH, HEIGHT = 1008, 614
CARD_X = (20, 348, 676)
FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in FONTS:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            FONTS[key] = ImageFont.truetype(name, size)
        except OSError:
            FONTS[key] = ImageFont.load_default(size=size)
    return FONTS[key]


def label(draw, xy, value, size=16, color=INK, bold=False):
    draw.text(xy, str(value), font=font(size, bold), fill=color)


def arrow(draw, start, end, color, width=3):
    start, end = np.asarray(start), np.asarray(end)
    delta = end - start
    if np.linalg.norm(delta) < 1:
        return
    unit = delta / np.linalg.norm(delta)
    side = np.array([-unit[1], unit[0]])
    draw.line([tuple(start), tuple(end)], fill=color, width=width)
    draw.polygon([tuple(end), tuple(end - 9 * unit + 4 * side), tuple(end - 9 * unit - 4 * side)], fill=color)


def star(draw, xy, radius=12, color=GREEN):
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        points.append((xy[0] + r * math.cos(angle), xy[1] + r * math.sin(angle)))
    draw.polygon(points, fill=color)


def canvas(title, scene, index, count, step, horizon, subtitle):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f4f7fb")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 96), fill="#112239")
    label(draw, (24, 13), "BELIEF-CONDITIONED FB  /  DYNAMICS IN MOTION", 12, "#8cb9df", True)
    label(draw, (22, 37), title, 29, "#ffffff", True)
    label(draw, (24, 107), scene["title"], 19, INK, True)
    label(draw, (24, 137), scene["detail"], 14, MUTED)
    label(draw, (24, 162), subtitle, 13, MUTED)
    for i, method in enumerate(METHODS):
        x = CARD_X[i]
        draw.rounded_rectangle((x, 194, x + 312, 538), radius=14, fill="white", outline="#dce4ee")
        draw.rounded_rectangle((x + 14, 207, x + 19, 228), radius=2, fill=COLORS[method])
        label(draw, (x + 28, 204), method.upper(), 19, COLORS[method], True)
    label(draw, (24, 553), f"EPISODE {index + 1}/{count}  ·  step {step:03d}/{horizon}", 13, MUTED, True)
    label(draw, (442, 553), "New dynamics → reset + fresh reward-free context", 13, MUTED)
    draw.rounded_rectangle((24, 582, WIDTH - 24, 588), radius=3, fill="#dce4ee")
    progress = (index + step / horizon) / count
    draw.rounded_rectangle((24, 582, 24 + (WIDTH - 48) * max(progress, 0.001), 588), radius=3, fill="#167fb8")
    return image


def grid_frame(scene, index, count, step):
    image = canvas(
        "Grid navigation: changing transition rules",
        scene,
        index,
        count,
        step,
        100,
        "Calibration + collision memory for all methods  ·  online beliefs for BFB/RFB  ·  goal ★",
    )
    draw = ImageDraw.Draw(image)
    grid = np.asarray(scene["grid"])
    for i, method in enumerate(METHODS):
        x, y, cell = CARD_X[i] + 24, 238, 24
        for row in range(11):
            for col in range(11):
                fill = "#33465e" if grid[row, col] else "#f0f4f9"
                draw.rectangle(
                    (x + col * cell, y + row * cell, x + (col + 1) * cell - 1, y + (row + 1) * cell - 1), fill=fill
                )
        point = lambda p, x=x, y=y, cell=cell: (x + (p[1] + 0.5) * cell, y + (p[0] + 0.5) * cell)
        trace = scene["agents"][method]["trajectory"]
        current = trace[min(step, len(trace) - 1)]
        path = [point(r["observation"]) for r in trace[: step + 1]]
        if len(path) > 1:
            draw.line(path, fill=COLORS[method], width=4)
        sx, sy = point(START_POS)
        draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), outline="#167fb8", width=2)
        star(draw, point(scene["goal"]), 11)
        px, py = point(current["observation"])
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=COLORS[method], outline="white", width=2)
        status = "Goal reached" if current["score"] else "Horizon reached" if step == 100 else "Navigating"
        label(draw, (CARD_X[i] + 18, 511), status, 14, GREEN if current["score"] else MUTED)
    return image


def continuous_frame(scene, index, count, step, family):
    title = (
        "Point mass: unseen actuator dynamics" if family == "pointmass" else "Inertial motion: unseen actuator dynamics"
    )
    image = canvas(
        title, scene, index, count, step, 80, scene["protocol"] + "  ·  frozen network weights  ·  goal tolerance 0.12"
    )
    draw = ImageDraw.Draw(image)
    d = scene["dynamics"]
    # World-coordinate direction of a positive first-axis command, for explanation only.
    angle = d["angle"]
    arrow(draw, (950, 143), (950 + 24 * math.cos(angle), 143 - 24 * math.sin(angle)), "#9e5bbe")
    label(draw, (895, 106), "action +x", 12, "#9e5bbe")
    for i, method in enumerate(METHODS):
        x, y, side = CARD_X[i] + 24, 238, 264
        point = lambda p, x=x, y=y, side=side: (x + (p[0] + 1) * side / 2, y + (1 - p[1]) * side / 2)
        draw.rectangle((x, y, x + side, y + side), fill="#f7f9fc", outline="#dce4ee")
        for frac in (0.25, 0.5, 0.75):
            draw.line((x + side * frac, y, x + side * frac, y + side), fill="#e2e8f0")
            draw.line((x, y + side * frac, x + side, y + side * frac), fill="#e2e8f0")
        gx, gy = point(scene["goal"])
        r = 0.12 * side / 2
        draw.ellipse((gx - r, gy - r, gx + r, gy + r), fill="#daf3e9", outline="#9ddbc5")
        star(draw, (gx, gy), 10)
        trace = scene["agents"][method]["trajectory"]
        current = trace[min(step, len(trace) - 1)]
        path = [point(r["observation"][:2]) for r in trace[: step + 1]]
        if len(path) > 1:
            draw.line(path, fill=COLORS[method], width=3)
        px, py = path[-1]
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=COLORS[method], outline="white", width=2)
        if family == "inertial":
            velocity = np.asarray(current["observation"][2:])
            arrow(draw, (px, py), (px + 24 * velocity[0], py - 24 * velocity[1]), COLORS[method], 2)
        status = "Goal reached" if current["score"] else "Horizon reached" if step == 80 else "Navigating"
        label(draw, (CARD_X[i] + 18, 511), status, 14, GREEN if current["score"] else MUTED)
    return image


def ant_frame(scene, index, count, step):
    image = canvas(
        "AntWind: changing wind direction",
        scene,
        index,
        count,
        step,
        256,
        "Online belief  ·  matched behavior penalty 1.0  ·  task: forward locomotion",
    )
    draw = ImageDraw.Draw(image)
    angle = scene["context_id"] * 2 * np.pi / 20
    arrow(draw, (950, 143), (950 + 24 * math.cos(angle), 143 - 24 * math.sin(angle)), "#9e5bbe")
    label(draw, (904, 106), "wind x/y", 12, "#9e5bbe")
    for i, method in enumerate(METHODS):
        data = scene["agents"][method]
        frame_id = min(step // 8, len(data["images"]) - 1)
        image.paste(data["images"][frame_id].resize((288, 216), Image.Resampling.LANCZOS), (CARD_X[i] + 12, 241))
        state = data["trajectory"][frame_id]
        label(draw, (CARD_X[i] + 18, 474), f"Return  {state['score']:+.0f}", 18, COLORS[method], True)
        status = "Episode ended" if state["done"] else "Horizon reached" if step == 256 else "Running"
        label(draw, (CARD_X[i] + 18, 511), status, 14, MUTED)
    return image


def recorder(index):
    trace = []

    def callback(step, observations, scores, done):
        trace.append(
            {
                "step": step,
                "observation": observations[index].tolist(),
                "score": float(scores[index]),
                "done": bool(done[index]),
            }
        )

    return trace, callback


def checkpoint(run, continuous=False):
    cfg = OmegaConf.create(json.loads((run / "final.json").read_text())["config"])
    if continuous:
        obs_dim = 105 if cfg.family == "antwind" else 4 if cfg.family == "inertial" else 2
        agent = ContinuousFBAgent(cfg.agent, obs_dim, action_dim=8 if cfg.family == "antwind" else 2, seed=cfg.seed)
        agent.load(run / "checkpoint.pkl")
    else:
        agent, cfg = load_agent(run)
    return (
        agent,
        cfg,
        {"run": run.as_posix(), "checkpoint_sha256": file_sha256(run / "checkpoint.pkl"), "training_seed": cfg.seed},
    )


def grid_scenes(args):
    descriptions = [
        (
            "iid",
            "01  Held-out rooms · same dynamics distribution",
            "Standard action meanings  ·  deterministic motion  ·  start ○ → goal ★",
        ),
        (
            "remap",
            "02  OOD · unseen action permutation",
            "Command ↑ moves →   |   ↓ moves ←   |   ← moves ↑   |   → moves ↓",
        ),
        (
            "slip",
            "03  OOD · stochastic action execution",
            "25% chance of replacing a command with a uniform random action",
        ),
        (
            "obstacles",
            "04  OOD · different obstacle structure",
            "Connected random obstacles replace the four-room layout generator",
        ),
    ]
    scenes = []
    for suite, title, detail in descriptions:
        seeds, grids, kwargs = grid_suite(suite)
        scenes.append(
            {
                "suite": suite,
                "title": title,
                "detail": detail,
                "grid": grids[0],
                "goal": eval_goals()[2],
                "layout_seed": int(seeds[0]),
                "agents": {},
            }
        )
    for method in METHODS:
        run = args.grid_fb if method == "fb" else args.grid_root / method / "num_train_layouts30_context_len100_seed0"
        agent, cfg, meta = checkpoint(run)
        for scene in scenes:
            seeds, grids, kwargs = grid_suite(scene["suite"])
            trace, callback = recorder(2)
            result = evaluate(
                agent,
                seeds,
                cfg.context_len,
                num_contexts=20,
                seed=args.seed,
                online_belief=True,
                task_kappa=200.0,
                grids_override=grids,
                env_kwargs=kwargs,
                action_calibration=True,
                guard_blocked=True,
                trajectory_callback=callback,
            )
            scene["agents"][method] = {**meta, "trajectory": trace, "evaluation": result}
            print("recorded grid", method, scene["suite"], result["return"], flush=True)
        del agent
    return scenes


def continuous_scenes(args, family):
    specs = [
        ("iid", False, "01  Interpolation · held-out dynamics within the training range"),
        ("rotation", False, "02  OOD · excluded actuator-rotation quadrant"),
        ("rotation", True, "03  Same OOD rotation · with affine system identification"),
        ("gain", False, "04  OOD · weaker actuation than any training dynamics"),
    ]
    scenes = []
    for split, adapted, title in specs:
        dynamics = contexts(family, split)[0]
        detail = f"Rotation {np.degrees(dynamics.angle):.1f}°  ·  gain {dynamics.gain:.2f}  ·  wind magnitude 0.12"
        if family == "inertial":
            detail += f"  ·  damping {dynamics.damping:.2f}"
        protocol = "Online belief + affine fit from context" if adapted else "Online belief, no action adapter"
        scenes.append(
            {
                "split": split,
                "adapted": adapted,
                "title": title,
                "detail": detail,
                "protocol": protocol,
                "dynamics": asdict(dynamics),
                "goal": GOALS[0],
                "agents": {},
            }
        )
    reference = None
    for method in METHODS:
        run = args.continuous_root / family / method / "seed0"
        agent, cfg, meta = checkpoint(run, continuous=True)
        if reference is None:
            data = generate_dataset(family, cfg.num_train_contexts, cfg.episodes, cfg.episode_length, seed=0)
            reference = fit_dynamics(*[data[k][:1] for k in ("obs", "actions", "next_obs")])[0]
            del data
        for scene in scenes:
            trace, callback = recorder(0)
            result = evaluate_continuous(
                agent,
                family,
                scene["split"],
                cfg.context_len,
                cfg.eval_contexts,
                num_dynamics=cfg.num_train_contexts,
                seed=args.seed,
                online_belief=True,
                action_reference=reference if scene["adapted"] else None,
                trajectory_callback=callback,
            )
            scene["agents"][method] = {**meta, "trajectory": trace, "evaluation": result}
            print("recorded", family, method, scene["split"], scene["adapted"], result["return"], flush=True)
        del agent
    return scenes


def style_ant(env, method):
    """Change only visual materials before renderer initialization; physics and observations stay unchanged."""
    import mujoco

    model = env.model
    texture = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "texplane")
    start = model.tex_adr[texture]
    channels = model.tex_nchannel[texture] if hasattr(model, "tex_nchannel") else 3
    length = model.tex_height[texture] * model.tex_width[texture] * channels
    buffer = model.tex_data if hasattr(model, "tex_data") else model.tex_rgb
    pixels = buffer[start : start + length].reshape(-1, channels)[:, :3]
    scale = 1.0 if np.issubdtype(pixels.dtype, np.floating) else 255.0
    bright = pixels[:, 0] > 0.4 * scale
    pixels[:] = np.where(bright[:, None], [242, 246, 251], [209, 219, 232]) / 255 * scale
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_color = model.geom_rgba[floor].copy()
    model.geom_rgba[:, :3] = np.array(ImageColor.getrgb(COLORS[method])) / 255
    model.geom_rgba[floor] = floor_color
    model.mat_reflectance[:] = 0


def ant_scenes(args):
    from bfb.antwind import evaluate_antwind, load_antwind

    data, normalization, labels, _ = load_antwind(args.data_root)
    del data
    scenes = []
    for i, context_id in enumerate((0, 16, 19)):
        split = "train" if context_id < 16 else "test"
        scenes.append(
            {
                "context_id": context_id,
                "split": split,
                "title": f"{i + 1:02d}  {'Training direction' if split == 'train' else 'OOD · excluded wind direction'} · {context_id * 18}°",
                "detail": "Wind magnitude 4.0  ·  viscosity 0.05  ·  world-coordinate wind arrow →",
                "agents": {},
            }
        )
    for method in METHODS:
        agent, cfg, meta = checkpoint(args.ant_root / method / "seed0", continuous=True)
        for split in ("train", "test"):
            selected = [s for s in scenes if s["split"] == split]
            for scene in selected:
                scene["agents"][method] = {**meta, "trajectory": [], "images": []}

            def capture(step, envs, returns, active, selected=selected, split=split, cfg=cfg, method=method):
                if step % 8:
                    return
                for scene in selected:
                    index = (scene["context_id"] - (16 if split == "test" else 0)) * cfg.eval_contexts
                    env = envs[index]
                    if step == 0:
                        style_ant(env, method)
                    # Gymnasium's free camera does not follow trackbodyid by itself. Track the torso visually.
                    viewer = env.mujoco_renderer._get_viewer("rgb_array")
                    viewer.cam.lookat[:] = env.data.qpos[:3]
                    entry = scene["agents"][method]
                    entry["images"].append(Image.fromarray(env.render()))
                    entry["trajectory"].append(
                        {
                            "step": step,
                            "observation": env.data.qpos[:3].tolist(),
                            "score": float(returns[index]),
                            "done": not bool(active[index]),
                        }
                    )

            result = evaluate_antwind(
                agent,
                args.data_root,
                normalization,
                labels,
                split,
                cfg.context_len,
                cfg.eval_contexts,
                seed=args.seed,
                online_belief=True,
                frame_callback=capture,
                render_kwargs={
                    "width": 384,
                    "height": 288,
                    "default_camera_config": {"distance": 4.2, "azimuth": 125, "elevation": -30, "trackbodyid": 1},
                },
            )
            for scene in selected:
                scene["agents"][method]["evaluation"] = result
            print("recorded AntWind", method, split, result["return"], flush=True)
        del agent
    return scenes


def encode_frames(frames, durations, path):
    # One shared palette avoids frame-to-frame color flicker; no dithering keeps text and trajectories crisp.
    samples = frames[:: max(1, len(frames) // 24)]
    strip = Image.new("RGB", (252, 154 * len(samples)))
    for i, frame in enumerate(samples):
        strip.paste(frame.resize((252, 154)), (0, i * 154))
    # Reserve one palette entry for transparent unchanged pixels in delta frames.
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    encoded = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    encoded[0].save(
        path, save_all=True, append_images=encoded[1:], duration=durations, loop=0, optimize=True, disposal=1
    )


def compress_record(path):
    """Keep human-readable local records and a small deterministic archive for Git."""
    payload = json.dumps(json.loads(path.read_text()), separators=(",", ":")).encode()
    with (
        path.with_suffix(".json.gz").open("wb") as file,
        gzip.GzipFile(fileobj=file, mode="wb", mtime=0) as compressed,
    ):
        compressed.write(payload)


def save(args, suite, scenes):
    horizon, stride = (256, 8) if suite == "antwind" else (100, 2) if suite == "grid" else (80, 2)
    frames, durations = [], []
    for index, scene in enumerate(scenes):
        for step in range(0, horizon + 1, stride):
            if suite == "grid":
                frame = grid_frame(scene, index, len(scenes), step)
            elif suite == "antwind":
                frame = ant_frame(scene, index, len(scenes), step)
            else:
                frame = continuous_frame(scene, index, len(scenes), step, suite)
            frames.append(frame)
            durations.append(650 if step == 0 else 850 if step == horizon else 80)
    path = args.out / f"{suite}.gif"
    encode_frames(frames, durations, path)
    # A small contact sheet is useful for reviewing every scene before publication.
    contact = Image.new("RGB", (WIDTH, HEIGHT * len(scenes)))
    frames_per_scene = horizon // stride + 1
    for i in range(len(scenes)):
        contact.paste(frames[i * frames_per_scene + frames_per_scene // 2], (0, HEIGHT * i))
    contact.save(args.out / f"{suite}_preview.png", optimize=True)
    for scene in scenes:
        for data in scene["agents"].values():
            data.pop("images", None)
    atomic_json(
        args.out / f"{suite}.json",
        {
            "suite": suite,
            "evaluation_seed": args.seed,
            "selection": "seed 0; first dynamics/context; grid goal 3, continuous goal 1; AntWind IDs 0,16,19",
            "dynamics_change": "between independent episodes, with reset and fresh reward-free context",
            "stride": stride,
            "frames": len(frames),
            "gif_sha256": file_sha256(path),
            "scenes": scenes,
        },
    )
    compress_record(args.out / f"{suite}.json")
    print(f"saved {path}: {len(frames)} frames, {path.stat().st_size / 1024**2:.2f} MiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "grid", "pointmass", "inertial", "antwind"], default="all")
    parser.add_argument("--grid-fb", type=Path, default=Path("runs/reproduction/fb30_seed0"))
    parser.add_argument("--grid-root", type=Path, default=Path("runs/fig5"))
    parser.add_argument("--continuous-root", type=Path, default=Path("runs/continuous"))
    parser.add_argument("--ant-root", type=Path, default=Path("runs/antwind_bc"))
    parser.add_argument("--data-root", type=Path, default=Path("data/antwind_official"))
    parser.add_argument("--out", type=Path, default=Path("figures/dynamics"))
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    suites = ["grid", "pointmass", "inertial", "antwind"] if args.suite == "all" else [args.suite]
    args.out.mkdir(parents=True, exist_ok=True)
    for suite in suites:
        if (args.out / f"{suite}.gif").exists() and not args.overwrite:
            parser.error(f"{suite}.gif already exists; use --overwrite or another --out")
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "data" / "jax_cache"))
    for suite in suites:
        scenes = (
            grid_scenes(args)
            if suite == "grid"
            else ant_scenes(args)
            if suite == "antwind"
            else continuous_scenes(args, suite)
        )
        save(args, suite, scenes)
        jax.clear_caches()


if __name__ == "__main__":
    main()
