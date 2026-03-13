# Staré Hory Wildfire Simulations with PyTorchFire

This repository contains a small, reproducible set of experiments demonstrating wildfire spread over real terrain using the [PyTorchFire](https://github.com/pytorchfire/pytorchfire) library. The focus is the Staré Hory test case from central Slovakia, implemented as a raster cellular automaton on top of a national digital terrain model (DTM).

The core script is `stare_hory_sim.py`, which:

- extracts a window around Staré Hory from a Slovakia-wide DTM,
- builds PyTorchFire environment tensors (slope, wind, ignition, homogeneous fuel),
- runs several wind scenarios with identical fuel and terrain, and
- saves full time-series outputs for later visualisation in a Jupyter notebook.

For background and detailed rationale of the setup, see `stare_hory_study.md`.

## Data Source

The elevation data for Slovakia come from a national digital terrain model:

- **File used here**: `DTM_Slovakia_20m.tif`
- **Source**: `https://sonny.4lima.de`

This repository does **not** redistribute the original DTM file. You should download it yourself (or an equivalent DTM) from the source above and place it in the project root as `DTM_Slovakia_20m.tif` before running the simulations.

## Project Structure

- `stare_hory_sim.py` – main script that builds the environment and runs the Staré Hory scenarios using PyTorchFire.
- `stare_hory_study.md` – internal notes explaining the modelling assumptions, parameter translation, and how the scenarios relate to the original Staré Hory study.
- `pyproject.toml` – project metadata and dependencies for installation with `pip`.
- `.gitignore` – ignore rules for virtual environments, data, and generated outputs.

## Local Development Setup

These steps assume a Unix-like environment (macOS/Linux) with Python 3.10+ installed.

### 1. Create and activate a virtual environment

In the project root (this directory):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install the project in editable mode

With the virtual environment activated:

```bash
pip install --upgrade pip
pip install -e .
```

This will install `pytorchfire-stare-hory-experiments` along with its dependencies (including `pytorchfire`, `torch`, `numpy`, and `rasterio`). You may also want optional tools for notebooks and plotting:

```bash
pip install -e ".[dev]"
```

Note: PyTorch installation details (CPU vs GPU, specific CUDA version) may depend on your system; if needed, follow the official PyTorch installation instructions and then reinstall this project.

### 3. Prepare the DEM file

1. Download the appropriate **DTM Slovakia** GeoTIFF from `https://sonny.4lima.de` (or another authoritative source providing the same dataset).
2. Place the file in the project root and name it:

   - `DTM_Slovakia_20m.tif`

`stare_hory_sim.py` will look for this exact path and will raise an error if it cannot be found.

### 4. Run the Staré Hory simulations

With the virtual environment active and the DEM file in place:

```bash
python stare_hory_sim.py
```

This will:

- load the Staré Hory window from `DTM_Slovakia_20m.tif`,
- configure four scenarios (no wind + three wind directions),
- run 300 CA steps for each scenario, and
- save `.npy` files containing the affected-cell time series for each scenario.

By default, outputs are written to the PyTorchFire examples directory, e.g.:

- `.../PyTorchFire/examples/project_experiments/stare_hory_outputs/{scenario_name}_frames.npy`

You can adjust the output directory inside `stare_hory_sim.py` if needed.

### 5. Visualising results (optional)

If you have a Jupyter environment set up (e.g. via `pip install -e ".[dev]"`), you can create a notebook that:

- loads the saved `*_frames.npy` arrays,
- uses Matplotlib animations (`FuncAnimation`) to show the evolution of affected cells,
- compares the four wind scenarios side by side.

The internal document `stare_hory_study.md` describes one such workflow in more detail.

## Cloning and Using This Repository

Once this directory is pushed to GitHub, users can reproduce your setup as follows:

```bash
git clone https://github.com/<your-username>/pytorchfire-stare-hory-experiments.git
cd pytorchfire-stare-hory-experiments
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"
```

Then download the Slovakia DTM from `https://sonny.4lima.de`, save it as `DTM_Slovakia_20m.tif` in the project root, and run:

```bash
python stare_hory_sim.py
```

This will reproduce the Staré Hory wildfire spread simulations demonstrated in this repository.

