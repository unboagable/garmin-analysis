# Credits and Acknowledgments

This project stands on the shoulders of giants. We gratefully acknowledge the following projects and contributors:

## Core Dependencies

### GarminDB
- **Project**: https://github.com/tcgoetz/GarminDB
- **Author**: Tom Goetz
- **License**: GPL-2.0
- **Description**: Download and parse data from Garmin Connect or a Garmin watch into SQLite databases
- **Our Usage**: We integrate GarminDB to provide automated Garmin Connect data synchronization, eliminating manual export steps for our users

GarminDB provides the foundational infrastructure for:
- Garmin Connect authentication and API access
- FIT file downloading and parsing
- SQLite database schema for health data
- Daily monitoring, sleep, weight, and activity data extraction

### Python Data Science Stack
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **matplotlib** - Visualization library
- **seaborn** - Statistical data visualization

### Machine Learning Libraries
- **scikit-learn** - Machine learning algorithms
- **tsfresh** - Time series feature extraction
- **statsmodels** - Statistical modeling
- **prophet** - Time series forecasting

### Web Dashboard
- **Dash** by Plotly - Interactive web applications
- **Plotly** - Interactive graphing library

### Development Tools
- **pytest** - Testing framework
- **Poetry** - Dependency management
- **Jupyter** - Interactive notebooks

## Garmin Developer Community

We acknowledge the broader Garmin developer community for their work on:
- Reverse-engineering Garmin data formats
- Documenting FIT file specifications
- Creating tools and libraries for Garmin data access
- Sharing knowledge about Garmin Connect APIs

## Python Community

This project benefits from the incredible Python ecosystem, including:
- Python core developers
- PyPI package maintainers
- Open source contributors worldwide

## Inspiration

Inspired by the quantified self movement and the desire to understand and analyze personal health data beyond what commercial apps provide.

## License Compatibility

This project is licensed under the MIT License. GarminDB is an optional, externally installed dependency (GPL-2.0); this project does not vendor or distribute GarminDB. See the [README Licensing section](README.md#licensing) for details.

## Special Thanks

- **Tom Goetz** - For creating and maintaining GarminDB, making Garmin data accessible to researchers and enthusiasts
- **Garmin** - For creating excellent fitness tracking hardware
- **All open source contributors** - For making projects like this possible

---

**Last Updated**: October 18, 2025

If we've missed anyone or any project, please let us know by filing an issue or pull request.

