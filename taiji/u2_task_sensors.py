"""Causal raster, acoustic and proprioceptive observations for the U2 world."""

from __future__ import annotations

import numpy as np
import torch


def render(bodies, path, command, camera, noise, occluder, inspect=False, initial=False):
    intensity = float(command[13])
    extent = 1 - 0.75 * intensity if inspect else 1.0
    center = (1 - extent) * np.asarray(command[9:11]) if inspect else np.zeros(2)
    exposure = float(command[14]) if inspect else camera["passive_exposure"]
    if initial:
        exposure = 0.0
    if inspect and exposure == 0:
        return None, None
    side = camera["side"]
    image = np.zeros((side, side, 3))
    times = np.array([entry[0] for entry in path])
    finish = times[-1]
    for temporal in range(camera["temporal_samples"]):
        when = max(
            0.0, finish - exposure + (temporal + 0.5) * exposure / camera["temporal_samples"]
        )
        upper = min(int(np.searchsorted(times, when, side="right")), len(times) - 1)
        lower = max(0, upper - 1)
        fraction = (
            0.0
            if times[upper] == times[lower]
            else (when - times[lower]) / (times[upper] - times[lower])
        )
        positions = path[lower][1] * (1 - fraction) + path[upper][1] * fraction
        order = sorted(range(len(bodies)), key=lambda i: (*positions[i], bodies[i]["radius"]))
        for sy in camera["spatial_offsets"]:
            for sx in camera["spatial_offsets"]:
                x = center[0] + extent * (2 * (np.arange(side) + sx) / side - 1)
                y = center[1] + extent * (1 - 2 * (np.arange(side) + sy) / side)
                xx, yy = np.meshgrid(x, y)
                frame = np.broadcast_to(
                    np.array(camera["background"]) / 255, (side, side, 3)
                ).copy()
                for i in order:
                    body = bodies[i]
                    mask = (xx - positions[i, 0]) ** 2 + (yy - positions[i, 1]) ** 2 <= body[
                        "radius"
                    ] ** 2
                    frame[mask] = np.array(body["color"]) / 255
                if occluder is not None:
                    x0, y0, x1, y1 = occluder
                    frame[(xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)] = (
                        np.array(camera["occluder_rgb"]) / 255
                    )
                image += frame / (camera["temporal_samples"] * len(camera["spatial_offsets"]) ** 2)
    sigma = min(
        camera["noise_cap"],
        camera["noise_base"]
        * np.sqrt(camera["noise_reference_duration"] / max(exposure, 1 / 16000)),
    )
    raw = torch.from_numpy(np.rint(np.clip(image + sigma * noise, 0, 1) * 255).astype(np.uint8))
    coordinates = np.zeros((16, 8))
    for row in range(4):
        for col in range(4):
            coordinates[row * 4 + col, :2] = center + extent * np.array(
                [2 * (col + 0.5) / 4 - 1, 1 - 2 * (row + 0.5) / 4]
            )
    coordinates[:, 3] = np.arcsinh(-exposure)
    coordinates[:, 5] = 1
    coordinates[:, 7] = 1
    return raw, torch.from_numpy(coordinates)


def acoustic(bank, ring, sounds, config, noise):
    count = len(noise)
    times = (np.arange(count) + 1) / config["rate"]
    poles = -1 / config["decay_seconds"] + 2j * np.pi * np.array(config["frequencies"])
    z = np.asarray(bank)[:, 0] + 1j * np.asarray(bank)[:, 1]
    signal = (z[:, None] * np.exp(poles[:, None] * times)).imag.sum(axis=0)
    end = times[-1]
    next_z = z * np.exp(poles * end)
    for when, frequency, amplitude in sounds:
        index = config["frequencies"].index(frequency)
        valid = times >= when
        signal[valid] += (amplitude * np.exp(poles[index] * (times[valid] - when))).imag
        next_z[index] += amplitude * np.exp(poles[index] * (end - when))
    wave = np.clip(signal + config["noise_std"] * noise, -1, 1)
    next_ring = np.concatenate([np.asarray(ring), wave])[-config["samples"] :]
    return np.stack([next_z.real, next_z.imag], axis=1).tolist(), next_ring.tolist()


def encode(raw, rgb_coordinates, config, dtype):
    observations, coordinates = {}, {}
    if "rgb" in raw:
        image = raw["rgb"].to(dtype=dtype) / 255
        observations["rgb"] = image.reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(16, 48)
        coordinates["rgb"] = rgb_coordinates.to(dtype=dtype)
    if "audio" in raw:
        audio = config["audio"]
        wave = raw["audio"].to(dtype=dtype)
        observations["audio"] = (
            torch.stft(
                wave,
                audio["fft"],
                audio["hop"],
                window=torch.hann_window(audio["fft"], dtype=dtype),
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            .abs()
            .log1p()
            .T
        )
        coord = torch.zeros(7, 8, dtype=dtype)
        coord[:, 2] = torch.linspace(-1, 1, 7, dtype=dtype)
        start = (torch.arange(7, dtype=dtype) * 32 - 256) / 16000
        coord[:, 3], coord[:, 4] = start.asinh(), (start + 64 / 16000).asinh()
        coord[:, 6:] = 1
        coordinates["audio"] = coord
    if "proprio" in raw and bool(raw["proprio_valid"].any()):
        valid = raw["proprio_valid"]
        values = torch.where(valid, raw["proprio"], 0).to(dtype=dtype)
        observations["proprio"] = torch.cat([values, valid.to(dtype=dtype)])[None]
        coordinates["proprio"] = torch.zeros(1, 8, dtype=dtype)
    return observations, coordinates
