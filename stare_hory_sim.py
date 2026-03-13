import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio import windows
from rasterio.warp import transform as rio_transform

import torch

from pytorchfire.model import WildfireModel
from pytorchfire.utils import calculate_slope


def load_stare_hory_window(
    tif_path: Path,
    center_lat: float = 48.8333,
    center_lon: float = 19.1167,
    half_size_cells: int = 100,
) -> torch.Tensor:
    """
    Load a square elevation window around Staré Hory from the Slovakia DTM.

    Parameters
    ----------
    tif_path:
        Path to `DTM_Slovakia_20m.tif`.
    center_lat, center_lon:
        Approximate geographic coordinates of Staré Hory (degrees).
    half_size_cells:
        Half window size in cells (total size = 2 * half_size_cells + 1).

    Returns
    -------
    altitude: torch.Tensor
        Elevation in meters, shape [H, W].
    cell_size: float
        Horizontal resolution in meters (assumed square pixels).
    """
    with rasterio.open(tif_path) as ds:
        # Transform geographic coordinates (WGS84) to dataset CRS (EPSG:32634)
        xs, ys = rio_transform("EPSG:4326", ds.crs, [center_lon], [center_lat])
        x, y = xs[0], ys[0]

        row, col = ds.index(x, y)
        row_off = max(row - half_size_cells, 0)
        col_off = max(col - half_size_cells, 0)
        height = min(2 * half_size_cells + 1, ds.height - row_off)
        width = min(2 * half_size_cells + 1, ds.width - col_off)

        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)
        data = ds.read(1, window=window)  # single band DEM

        # Replace NaNs with local mean (or zero if everything is NaN)
        if np.isnan(data).any():
            valid = np.isfinite(data)
            if valid.any():
                mean_val = float(data[valid].mean())
                data[~valid] = mean_val
            else:
                data[:] = 0.0

        # Cell size in meters (assume square pixels)
        cell_size = float(abs(ds.transform.a))

    altitude = torch.from_numpy(data.astype(np.float32))
    return altitude, cell_size


def build_env_from_dem(
    altitude: torch.Tensor,
    cell_size_m: float,
    wind_velocity_mps: float,
    wind_direction_deg: float | None,
) -> dict:
    """
    Construct environment tensors for WildfireModel from DEM and wind settings.
    """
    height, width = altitude.shape

    # Homogeneous fuel: single vegetation type, no density variation (Bratislava §6.4 test case)
    p_veg = torch.zeros(height, width)
    p_den = torch.zeros(height, width)

    # Wind fields
    if wind_direction_deg is None:
        wind_velocity = torch.zeros(height, width)
        wind_towards_direction = torch.zeros(height, width)
    else:
        wind_velocity = torch.full((height, width), float(wind_velocity_mps), dtype=torch.float32)
        wind_towards_direction = torch.full((height, width), float(wind_direction_deg), dtype=torch.float32)

    # Slope field from DEM
    slope = calculate_slope(altitude, torch.tensor(cell_size_m, dtype=torch.float32))

    # Initial ignition at center of window
    initial_ignition = torch.zeros(height, width, dtype=torch.bool)
    cy, cx = height // 2, width // 2
    radius = max(min(height, width) // 40, 2)
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    initial_ignition[mask] = True

    env_data = {
        "p_veg": p_veg,
        "p_den": p_den,
        "wind_velocity": wind_velocity,
        "wind_towards_direction": wind_towards_direction,
        "slope": slope,
        "initial_ignition": initial_ignition,
    }
    return env_data


def run_scenario(
    name: str,
    env_data: dict,
    params: dict,
    num_steps: int = 300,
    device: torch.device | None = None,
) -> dict:
    """
    Run a wildfire simulation scenario and return summary results.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_data_dev = {k: v.to(device) for k, v in env_data.items()}
    params_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in params.items()}

    model = WildfireModel(env_data=env_data_dev, params=params_dev).to(device)
    model.eval()
    model.reset()

    frames = []
    for _ in range(num_steps):
        burning, burned = model.state
        affected = (burning | burned).detach().cpu()
        frames.append(affected)
        model.compute()

    final = frames[-1]
    burned_fraction = final.float().mean().item()

    return {
        "name": name,
        "frames": torch.stack(frames, dim=0),  # [T, H, W] bool
        "burned_fraction": burned_fraction,
    }


def main():
    project_root = Path(__file__).resolve().parent
    tif_path = project_root / "DTM_Slovakia_20m.tif"
    if not tif_path.exists():
        raise FileNotFoundError(f"DEM not found at {tif_path}")

    altitude, cell_size = load_stare_hory_window(tif_path)

    # Bratislava group's Stare Hory test settings (from notebook markdown)
    # - "wind velocity is set to 8 m min−1"
    # Convert to m/s
    wind_velocity_mps = 8.0 / 60.0

    # Map Bratislava model parameters onto WildfireModel parameters
    # α_s ≈ a, α_w ≈ c_1 and c_2
    base_params = {
        "a": torch.tensor(0.25, dtype=torch.float32),   # slope scaling (α_s)
        "p_h": torch.tensor(0.3, dtype=torch.float32),  # baseline spread (ε_F)
        "p_continue": torch.tensor(0.3, dtype=torch.float32),
        "c_1": torch.tensor(0.02, dtype=torch.float32),  # wind magnitude (α_w)
        "c_2": torch.tensor(0.02, dtype=torch.float32),  # wind directionality (α_w)
    }

    # Wind directions from Ambroz et al. are given as meteorological
    # "from" bearings (clockwise from North). PyTorchFire expects
    # mathematical "towards" angles in degrees, starting from East and
    # going counterclockwise. We therefore:
    #   1) flip from → towards by adding 180°
    #   2) convert compass towards to math angle via θ_math = 90° - θ_compass (mod 360)
    #
    # Results for the three non-zero wind cases:
    #   SE (from 135°)  → towards 315° compass → 135° math
    #   SSW (from 202.5°) → towards 22.5° compass → 67.5° math
    #   WNW (from 292.5°) → towards 112.5° compass → 337.5° math
    scenarios = [
        ("no_wind", None),
        ("wind_135_se", 135.0),
        ("wind_202_5_ssw", 67.5),
        ("wind_292_5_wnw", 337.5),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for name, direction in scenarios:
        env = build_env_from_dem(
            altitude=altitude,
            cell_size_m=cell_size,
            wind_velocity_mps=wind_velocity_mps if direction is not None else 0.0,
            wind_direction_deg=direction,
        )
        res = run_scenario(name=name, env_data=env, params=base_params, num_steps=300, device=device)
        results.append(res)
        print(f"Scenario {name}: burned_fraction={res['burned_fraction']:.4f}")

    # Optionally save frames for later visualization
    out_dir = Path("/Users/daria_kohut/Desktop/Projects/wild_fires/cellulalar_automata/PyTorchFire/examples/project_experiments/stare_hory_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for res in results:
        npy_path = out_dir / f"{res['name']}_frames.npy"
        np.save(npy_path, res["frames"].numpy())
        print(f"Saved {res['name']} frames to {npy_path}")


if __name__ == "__main__":
    main()

