# Sky Quality Meter (SQM) Data Reduction Pipeline

Universal Python processing pipeline for Sky Quality Meter (`.dat`) observations. Supports SQM-LU, SQM-LU-DL-R2, and UDM software outputs.

## Features
- **Astronomical Coordinate Calculations:** Computes exact solar altitude, lunar altitude, and hours relative to local solar midnight using Astropy.
- **Statistical Cloud Detection:** Implements a rolling $\chi^2$ (chi-squared) residual fluctuation test combined with a solid overcast magnitude ceiling.
- **Celestial Foreground Filtering:** Filters out zenith Galactic plane ($\vert{}b\vert{} < 10^\circ$) and Zodiacal light ($\vert{}\beta\vert{} < 15^\circ$) foregrounds.
- **Diagnostic Plot Suite:** Generates time series, nightly median trends, multi-stage histograms, and standard Jellyfish densitograms.

## Quick Start
```bash
./venv/bin/python sqm_pipeline.py
