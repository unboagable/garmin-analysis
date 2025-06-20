# Garmin Analysis

Garmin analysis and dashboard

## Getting Started

### Installing pipx

Install pipx using Homebrew on macOS:

```bash
brew install pipx
pipx ensurepath
sudo pipx ensurepath --global
```

Install pipx via Scoop on Windows:

```bash
scoop install pipx
pipx ensurepath
```

### Installing Poetry

Install Poetry using pipx:

```bash
pipx install poetry
```

### Installing Dependencies

Lock dependencies and install them with Poetry:

```bash
poetry lock
poetry install
```

### Getting the Garmin data

[GarminDB](https://github.com/tcgoetz/GarminDB)

```bash
garmindb_cli.py --all --download --import --analyze --latest
```
copy over garmin.db file

### Running the Application

Execute the application using Poetry:

```bash
poetry run python src/preview.py
```

## Running Using Notebooks

In the `notebooks` directory, create `.ipynb` files that can utilize `requirements.txt` to install dependencies.