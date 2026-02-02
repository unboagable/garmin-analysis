# Garmin Connect Integration

This project uses [GarminDB](https://github.com/tcgoetz/GarminDB) by Tom Goetz to download and import data from Garmin Connect. GarminDB must be installed **from source** (with git submodules); it is not available as a standard PyPI package.

## Installation

1. **Clone GarminDB with submodules** (required for Garmin Connect API support):

   ```bash
   git clone --recursive https://github.com/tcgoetz/GarminDB.git ~/GarminDB
   cd ~/GarminDB && make setup
   ```

2. **Configure Garmin Connect credentials** from this project:

   ```bash
   cd /path/to/garmin-analysis
   poetry run python -m garmin_analysis.cli_garmin_sync --setup \
     --username your@email.com \
     --password yourpassword \
     --start-date 01/01/2024
   ```

3. **Sync data**:

   ```bash
   poetry run python -m garmin_analysis.cli_garmin_sync --sync --all   # full history
   poetry run python -m garmin_analysis.cli_garmin_sync --sync --latest # daily updates
   ```

4. **Generate the unified dataset**:

   ```bash
   poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs
   ```

## Troubleshooting

- **Authentication errors**: Ensure your Garmin Connect email and password are correct. Two-factor authentication may require an [application-specific password](https://support.garmin.com/en-US/?faq=tdlRCyoIQf4u2vjqOcA0w8) or disabling 2FA for the account used with GarminDB.
- **Missing databases**: Use `poetry run python -m garmin_analysis.cli_garmin_sync --find-dbs` to locate GarminDB output, then `--copy-dbs` to copy them into this project’s `db/` directory.
- **GarminDB issues**: See the [GarminDB repository](https://github.com/tcgoetz/GarminDB) and [Python Garmin Connect API](https://github.com/cyberjunky/python-garmin-connect) for upstream documentation and support.

## Credits

- [GarminDB](https://github.com/tcgoetz/GarminDB) – GPL-2.0  
- [Python Garmin Connect API](https://github.com/cyberjunky/python-garmin-connect) – used by GarminDB for authentication
