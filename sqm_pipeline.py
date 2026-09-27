#!/Users/johnbarentine/Documents/Sky-brightness/Sky-Quality-Meter/SQM-data-reduction/venv/bin/python
"""
sqm_pipeline.py

Universal processing pipeline for Sky Quality Meter (.dat) files.
Supports SQM-LU, SQM-LU-DL-R2, and UDM software outputs.
Includes astronomical coordinate calculations, chi-squared (chi^2) cloud detection,
Milky Way & Zodiacal light filtering, CSV reporting, and a full plot suite:
  1. sqm_timeseries_raw.png
  2. sqm_timeseries_raw_vs_filtered.png
  3. sqm_nightly_medians.png
  4. sqm_hist_filtered.png
  5. sqm_hist_filter_steps.png
  6. sqm_jellyfish_sun_moon.png (X-axis: -9h to +9h from Local Solar Midnight)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter

# Astropy for solar, lunar, galactic, and ecliptic calculations
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body
from astropy.time import Time

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = "./data"

# Site Coordinates (Spring Valley State Park)
LAT_DEG = 38.02328
LON_DEG = -114.20243
ELEV_M = 1784.0

# Chi-Squared (chi^2) Cloud Detection Settings
# Following Falco & Barentine / DSN methodology:
# chi^2 is the residual sum of squares of a polynomial fit over a 90-minute window.
SQM_CHI2_THRESHOLD = 0.009      # DSN empirical threshold for SQM sensors (chi^2 = 0.009)
CHI2_WINDOW_MINUTES = 90        # 90-minute window centered on observation
CADENCE_MINUTES = 5             # SQM measurement cadence
CHI2_WINDOW_POINTS = 19         # 90 minutes / 5 min + 1 = 19 points

# Secondary Cloud Indicator (Solid Overcast Ceiling)
OVERCAST_THRESHOLD_MARGINAL = 22.1  # Solid overcast / dark sky ceiling
OVERCAST_THRESHOLD_DEFINITE = 22.3  # Heavy overcast ceiling

# Celestial Exclusion Angles (degrees)
MILKY_WAY_MIN_GAL_LAT = 10.0     # Reject zenith galactic latitude |b| < 10 deg
ZODIACAL_MIN_ECL_LAT = 15.0      # Reject zenith ecliptic latitude |beta| < 15 deg

# Output PNG Plots
RAW_TIMESERIES_PNG = "sqm_timeseries_raw.png"
RAW_VS_FILTERED_PNG = "sqm_timeseries_raw_vs_filtered.png"
NIGHTLY_MEDIANS_PNG = "sqm_nightly_medians.png"
HIST_FILTERED_PNG = "sqm_hist_filtered.png"
HIST_STEPS_PNG = "sqm_hist_filter_steps.png"
JELLYFISH_PNG = "sqm_jellyfish_sun_moon.png"

# Output Data CSVs
OUTPUT_CLEAN_CSV = "sqm_timeseries_clean.csv"
OUTPUT_NIGHTLY_CSV = "sqm_nightly_medians.csv"
HISTOGRAM_CSV = "sqm_histogram_partial_filters.csv"

# ----------------------------------------------------------------------
# Robust File Ingestion
# ----------------------------------------------------------------------
def parse_sqm_file(filepath):
    """
    Parses any SQM .dat file regardless of header line count, trailing semicolons,
    or device schema differences (SQM-LU vs SQM-LU-DL-R2).
    """
    df = pd.read_csv(
        filepath,
        sep=';',
        comment='#',
        header=None,
        engine='python',
        skip_blank_lines=True
    )
    
    # Drop columns that are entirely empty
    df = df.dropna(how='all', axis=1)
    
    # Standard SQM base column mapping
    base_cols = ['utc_time', 'local_time', 'temp_c', 'volts_counts', 'msas', 'record_type']
    
    num_cols = len(df.columns)
    if num_cols < len(base_cols):
        cols = base_cols[:num_cols]
    else:
        extra_cols = [f'extra_{i}' for i in range(num_cols - len(base_cols))]
        cols = base_cols + extra_cols
        
    df.columns = cols
    
    # Type conversion
    df['utc_time'] = pd.to_datetime(df['utc_time'], errors='coerce')
    df['local_time'] = pd.to_datetime(df['local_time'], errors='coerce')
    
    for col in ['temp_c', 'volts_counts', 'msas']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    if 'record_type' in df.columns:
        df['record_type'] = pd.to_numeric(df['record_type'], errors='coerce')
        
    return df

def load_all_sqm_files(data_dir):
    """Recursively finds and loads all .dat files within data_dir."""
    pattern = os.path.join(data_dir, "**", "*.dat")
    files = glob.glob(pattern, recursive=True) or glob.glob(os.path.join(data_dir, "*.dat"))
    
    if not files:
        raise FileNotFoundError(f"No .dat files found in directory: {data_dir}")
        
    dfs = []
    for f in sorted(files):
        try:
            parsed_df = parse_sqm_file(f)
            if not parsed_df.empty:
                dfs.append(parsed_df)
        except Exception as e:
            print(f"Warning: Failed to parse {f}: {e}")
            
    if not dfs:
        raise ValueError("No valid SQM data could be ingested from the provided files.")
        
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('utc_time').drop_duplicates(subset=['utc_time'])
    return combined

# ----------------------------------------------------------------------
# Astronomical Coordinates & Chi-Squared Cloud Detection
# ----------------------------------------------------------------------
def compute_astro_positions(df, lat, lon, elev_m):
    """
    Computes exact solar and lunar altitudes, hours relative to local solar midnight,
    and zenith Galactic/Ecliptic coordinates for filtering celestial foregrounds.
    """
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev_m * u.m)
    times = Time(df['utc_time'].values)
    
    altaz_frame = AltAz(obstime=times, location=location)
    
    sun = get_sun(times)
    df['sun_alt'] = sun.transform_to(altaz_frame).alt.deg
    df['moon_alt'] = get_body('moon', times, location=location).transform_to(altaz_frame).alt.deg
    
    # Calculate Local Apparent Sidereal Time (LAST) and Solar Hour Angle (SHA)
    lst_hours = times.sidereal_time('apparent', longitude=location.lon).hour
    sun_ra_hours = sun.ra.hour
    solar_hour_angle = (lst_hours - sun_ra_hours) % 24.0
    
    # Hours relative to local solar midnight (-12 to +12 hours, 0 = solar midnight)
    df['hours_from_solar_midnight'] = (solar_hour_angle - 12.0 + 12.0) % 24.0 - 12.0
    
    # Zenith Galactic and Ecliptic coordinates for foreground filtering
    zenith_altaz = SkyCoord(alt=90.0*u.deg, az=0.0*u.deg, frame=altaz_frame)
    
    zenith_gal = zenith_altaz.transform_to('galactic')
    df['zenith_gal_lat'] = zenith_gal.b.deg
    
    zenith_ecl = zenith_altaz.transform_to('geocentrictrueecliptic')
    df['zenith_ecl_lat'] = zenith_ecl.lat.deg
    
    return df

def compute_rolling_chi2(msas_series, window_size=CHI2_WINDOW_POINTS, poly_deg=2):
    """
    Computes the chi-squared (chi^2) statistic defined as the residual sum of squares (RSS)
    of an edge-aware polynomial fit over a ~90 minute window centered on each observation.
    """
    n = len(msas_series)
    vals = msas_series.values
    chi2_arr = np.zeros(n)
    
    half_w = window_size // 2
    
    # Pre-construct Vandermonde matrix and pseudoinverse for full-window acceleration
    x_full = np.linspace(-1, 1, window_size)
    V_full = np.vander(x_full, poly_deg + 1)
    V_pinv = np.linalg.pinv(V_full)
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        sub = vals[start:end]
        
        valid = ~np.isnan(sub)
        if np.sum(valid) < poly_deg + 2:
            chi2_arr[i] = 0.0
            continue
            
        if len(sub) == window_size and np.all(valid):
            coeffs = V_pinv @ sub
            fit = V_full @ coeffs
            res = sub - fit
            chi2_arr[i] = np.sum(res**2)
        else:
            x = np.linspace(-1, 1, len(sub))[valid]
            y = sub[valid]
            coeffs = np.polyfit(x, y, deg=poly_deg)
            fit = np.polyval(coeffs, x)
            chi2_arr[i] = np.sum((y - fit)**2)
            
    return pd.Series(chi2_arr, index=msas_series.index)

def detect_clouds_chi2(df, chi2_threshold=SQM_CHI2_THRESHOLD, t_marginal=OVERCAST_THRESHOLD_MARGINAL, t_definite=OVERCAST_THRESHOLD_DEFINITE):
    """
    Implements the Southern Arizona Dark Sky Network (DSN) chi-squared cloud detection method:
    1. Chi-Squared Index: Identifies variable clouds via short-term brightness fluctuations (chi^2 >= 0.009).
    2. Overcast Ceiling: Flags thick/solid overcast (which produces low chi^2) when MSAS >= 22.1 mag/arcsec^2.
    """
    df['chi2'] = compute_rolling_chi2(df['msas'], window_size=CHI2_WINDOW_POINTS, poly_deg=2)
    
    # Variable cloud flag based on chi^2 statistic
    df['is_cloud_variable'] = df['chi2'] >= chi2_threshold
    
    # Solid overcast flag based on absolute magnitude ceiling
    df['is_cloud_overcast_marginal'] = (df['msas'] >= t_marginal) & (df['msas'] < t_definite)
    df['is_cloud_overcast_definite'] = df['msas'] >= t_definite
    
    # Composite cloud classifications
    df['is_cloud_marginal'] = df['is_cloud_variable'] | df['is_cloud_overcast_marginal']
    df['is_cloud_definite'] = (df['chi2'] >= 0.05) | df['is_cloud_overcast_definite']
    df['is_cloudy'] = df['is_cloud_variable'] | (df['msas'] >= t_marginal)
    
    return df

# ----------------------------------------------------------------------
# Plotting Suite
# ----------------------------------------------------------------------
def plot_raw_timeseries(df, filename):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['utc_time'], df['msas'], '.', color='gray', alpha=0.5, ms=2)
    ax.set_ylabel("Sky Brightness (mag/arcsec^2)")
    ax.set_xlabel("UTC Date & Time")
    ax.set_title("Raw SQM Sky Brightness Time Series")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_raw_vs_filtered(df, df_filtered, filename):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['utc_time'], df['msas'], '.', color='lightgray', alpha=0.4, ms=2, label='Raw / Unfiltered Data')
    
    if 'is_cloud_marginal' in df.columns:
        marginal = df[df['is_cloud_marginal']]
        definite = df[df['is_cloud_definite']]
        
        if not marginal.empty:
            ax.plot(marginal['utc_time'], marginal['msas'], '.', color='orange', alpha=0.7, ms=3, label=f'Marginal / Variable Cloud (chi^2 >= {SQM_CHI2_THRESHOLD})')
            
        if not definite.empty:
            ax.plot(definite['utc_time'], definite['msas'], '.', color='red', alpha=0.8, ms=3, label=f'Heavy Cloud / Overcast (msas >= {OVERCAST_THRESHOLD_DEFINITE})')
            
    ax.plot(df_filtered['utc_time'], df_filtered['msas'], '.', color='darkblue', alpha=0.8, ms=3, label='Clean Dark Night (Cloud-Free, MW & Zodiacal Filtered)')
    ax.set_ylabel("Sky Brightness (mag/arcsec^2)")
    ax.set_xlabel("UTC Date & Time")
    ax.set_title("SQM Time Series: Raw vs. Clean Filtered Data")
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_nightly_medians(nightly_df, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        nightly_df['night_date'], 
        nightly_df['median'], 
        yerr=nightly_df['std'], 
        fmt='o-', 
        color='navy', 
        ecolor='skyblue', 
        capsize=3, 
        label='Nightly Median +/- Std Dev'
    )
    ax.set_ylabel("Median Sky Brightness (mag/arcsec^2)")
    ax.set_xlabel("Observation Night Date")
    ax.set_title("Nightly Median Sky Brightness (Moonless Astro Night, Clean Filtered)")
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_filtered_histogram(df_filtered, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = df_filtered['msas'].dropna()
    
    if len(data) > 0:
        # Traditional count histogram (density=False)
        counts, bins, _ = ax.hist(data, bins=40, color='darkslateblue', edgecolor='black', alpha=0.7, density=False)
        mu, std = norm.fit(data)
        xmin, xmax = ax.get_xlim()
        x_fit = np.linspace(xmin, xmax, 100)
        p_fit = norm.pdf(x_fit, mu, std)
        
        # Scale Gaussian PDF to match total observation count per bin width
        bin_width = bins[1] - bins[0]
        y_fit = p_fit * len(data) * bin_width
        
        label_text = f"Gaussian Fit\nmean={mu:.2f}, std={std:.2f}"
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=label_text)
    
    ax.set_xlabel("Sky Brightness (mag/arcsec^2)")
    ax.set_ylabel("Number of Observations")
    ax.set_title("Filtered Sky Brightness Distribution (Clean Dark Night)")
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_filter_steps_histogram(df, df_filtered, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins_step = np.linspace(0, 24, 60)
    
    ax.hist(df['msas'], bins=bins_step, alpha=0.3, label='1. All Data (24h)', color='gray')
    ax.hist(df[df['is_night']]['msas'], bins=bins_step, alpha=0.5, label='2. Astro Night (Sun <= -18 deg)', color='orange')
    if 'is_cloudy' in df.columns:
        ax.hist(df[df['is_moonless_night'] & (~df['is_cloudy'])]['msas'], bins=bins_step, alpha=0.5, label=f'3. Moonless & Unclouded (chi^2 < {SQM_CHI2_THRESHOLD})', color='cyan')
    ax.hist(df_filtered['msas'], bins=bins_step, alpha=0.7, label='4. Fully Clean (MW & Zodiacal Filtered)', color='blue')
    
    ax.set_xlabel("Sky Brightness (mag/arcsec^2)")
    ax.set_ylabel("Count")
    ax.set_title("Sky Brightness Histograms Across Filtering Stages")
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_jellyfish_densitogram(df, filename):
    """
    Plots standard Jellyfish Densitogram with X-axis in hours relative to local solar midnight.
    Window tightened to [-9h, +9h].
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid = ~(df['hours_from_solar_midnight'].isna() | df['msas'].isna())
    x_val = df.loc[valid, 'hours_from_solar_midnight']
    y_val = df.loc[valid, 'msas']
    
    # Tightened window [-9h, +9h]
    x_bins = np.linspace(-9, 9, 108)
    y_bins = np.linspace(0, 24, 120)
    
    H, xedges, yedges = np.histogram2d(x_val, y_val, bins=(x_bins, y_bins))
    H_smooth = gaussian_filter(H, sigma=1.0)
    
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    mesh = ax.pcolormesh(X, Y, H_smooth.T, cmap='magma', shading='auto')
    
    ax.set_ylabel("Sky Brightness (mag/arcsec^2)")
    ax.set_xlabel("Hours relative to Local Solar Midnight (hours)")
    ax.set_title("SQM Jellyfish Densitogram (Sky Brightness vs. Time from Solar Midnight)")
    ax.set_xlim(-9, 9)
    fig.colorbar(mesh, ax=ax, label='Sample Density')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# ----------------------------------------------------------------------
# Pipeline Execution
# ----------------------------------------------------------------------
def run_pipeline():
    print("Ingesting SQM data files...")
    df_raw = load_all_sqm_files(DATA_DIR)
    n_raw_total = len(df_raw)
    
    # Step 0: Sensor Integrity Filter (drop missing timestamps, missing msas, record_type == 0, and msas <= 0)
    valid_mask = (
        df_raw['utc_time'].notna() & 
        df_raw['msas'].notna() & 
        (df_raw['msas'] > 0)
    )
    if 'record_type' in df_raw.columns:
        valid_mask = valid_mask & (df_raw['record_type'] != 0)
        
    df = df_raw[valid_mask].copy()
    n_valid_sensor = len(df)
    n_screened_sensor = n_raw_total - n_valid_sensor
    
    print("Calculating astronomical coordinates (solar, lunar, galactic, ecliptic)...")
    df = compute_astro_positions(df, LAT_DEG, LON_DEG, ELEV_M)
    
    print(f"Running DSN chi-squared cloud detection (chi^2 threshold = {SQM_CHI2_THRESHOLD})...")
    df = detect_clouds_chi2(df, chi2_threshold=SQM_CHI2_THRESHOLD, t_marginal=OVERCAST_THRESHOLD_MARGINAL, t_definite=OVERCAST_THRESHOLD_DEFINITE)
    
    # Sequential Filtering Cascade
    # Step 1: Astronomical Night Filter (Sun alt <= -18 deg)
    df['is_night'] = df['sun_alt'] <= -18.0
    step1_df = df[df['is_night']].copy()
    n_step1 = len(step1_df)
    n_screened_step1 = n_valid_sensor - n_step1
    
    # Step 2: Moonless Night Filter (Moon alt < 0 deg)
    step1_df['is_moonless'] = step1_df['moon_alt'] < 0.0
    step2_df = step1_df[step1_df['is_moonless']].copy()
    n_step2 = len(step2_df)
    n_screened_step2 = n_step1 - n_step2
    
    # Step 3: Chi-Squared Cloud Filter (Unclouded: chi^2 < 0.009 and msas < 22.1)
    step3_df = step2_df[~step2_df['is_cloudy']].copy()
    n_step3 = len(step3_df)
    n_screened_step3 = n_step2 - n_step3
    
    # Step 4: Milky Way Exclusion Filter (|b_zenith| >= 10 deg)
    step3_df['is_mw_free'] = np.abs(step3_df['zenith_gal_lat']) >= MILKY_WAY_MIN_GAL_LAT
    step4_df = step3_df[step3_df['is_mw_free']].copy()
    n_step4 = len(step4_df)
    n_screened_step4 = n_step3 - n_step4
    
    # Step 5: Zodiacal Light Exclusion Filter (|beta_zenith| >= 15 deg)
    step4_df['is_zodi_free'] = np.abs(step4_df['zenith_ecl_lat']) >= ZODIACAL_MIN_ECL_LAT
    filtered_df = step4_df[step4_df['is_zodi_free']].copy()
    n_step5 = len(filtered_df)
    n_screened_step5 = n_step4 - n_step5
    
    # Set boolean columns on full df for plotting routines
    df['is_moonless_night'] = df['is_night'] & (df['moon_alt'] < 0.0)
    df['is_milky_way_free'] = np.abs(df['zenith_gal_lat']) >= MILKY_WAY_MIN_GAL_LAT
    df['is_zodiacal_free'] = np.abs(df['zenith_ecl_lat']) >= ZODIACAL_MIN_ECL_LAT
    df['is_dark'] = (
        df['is_moonless_night'] & 
        (df['msas'] > 0) & 
        (~df['is_cloudy']) & 
        df['is_milky_way_free'] & 
        df['is_zodiacal_free']
    )
    df['night_date'] = (df['local_time'] - pd.Timedelta(hours=12)).dt.date
    filtered_df['night_date'] = (filtered_df['local_time'] - pd.Timedelta(hours=12)).dt.date
    
    # --- 1. CSV Exports ---
    print("Exporting CSV datasets...")
    filtered_df.to_csv(OUTPUT_CLEAN_CSV, index=False)
    
    nightly = filtered_df.groupby('night_date')['msas'].agg(['count', 'median', 'std', 'min', 'max']).reset_index()
    nightly.to_csv(OUTPUT_NIGHTLY_CSV, index=False)
    
    hist_all, bin_edges = np.histogram(df['msas'].dropna(), bins=50, range=(0, 25))
    hist_night, _ = np.histogram(df[df['is_night']]['msas'].dropna(), bins=bin_edges)
    hist_dark, _ = np.histogram(filtered_df['msas'].dropna(), bins=bin_edges)
    
    hist_csv_df = pd.DataFrame({
        'bin_start': bin_edges[:-1],
        'bin_end': bin_edges[1:],
        'count_all': hist_all,
        'count_astro_night': hist_night,
        'count_clean_moonless_dark': hist_dark
    })
    hist_csv_df.to_csv(HISTOGRAM_CSV, index=False)
    
    # --- 2. PNG Plotting ---
    print("Generating diagnostic plots...")
    plot_raw_timeseries(df, RAW_TIMESERIES_PNG)
    plot_raw_vs_filtered(df, filtered_df, RAW_VS_FILTERED_PNG)
    plot_nightly_medians(nightly, NIGHTLY_MEDIANS_PNG)
    plot_filtered_histogram(filtered_df, HIST_FILTERED_PNG)
    plot_filter_steps_histogram(df, filtered_df, HIST_STEPS_PNG)
    plot_jellyfish_densitogram(df, JELLYFISH_PNG)
    
    # --- Final Measurement & Filtering Cascade Summary ---
    print("\n" + "="*68)
    print("DATA FILTERING CASCADE & CHI-SQUARED CLOUD SUMMARY")
    print("="*68)
    print(f"1. Total Raw Measurements Ingested:             {n_raw_total:>8d}")
    print(f"   - Invalid / Saturated Sensor Readings Out:    {n_screened_sensor:>8d}")
    print(f"   = Valid Sensor Measurements Remaining:         {n_valid_sensor:>8d}  (100.0%)")
    print("-" * 68)
    print("2. Sequential Filtering Cascade:")
    print(f"   a. Astronomical Night Filter (Sun <= -18 deg):")
    print(f"      - Screened Out (Daytime & Twilight):       {n_screened_step1:>8d}")
    print(f"      = Remaining (Astronomical Night):          {n_step1:>8d}  ({n_step1/n_valid_sensor*100:5.1f}%)")
    print(f"   b. Lunar Altitude Filter (Moon < 0 deg):")
    print(f"      - Screened Out (Moonlit Periods):          {n_screened_step2:>8d}")
    print(f"      = Remaining (Moonless Astronomical Night): {n_step2:>8d}  ({n_step2/n_valid_sensor*100:5.1f}%)")
    print(f"   c. Chi-Squared Cloud Filter (chi^2 < {SQM_CHI2_THRESHOLD} & msas < {OVERCAST_THRESHOLD_MARGINAL}):")
    print(f"      - Screened Out (Variable & Overcast Clouds):{n_screened_step3:>8d}")
    print(f"      = Remaining (Cloud-Free Night):            {n_step3:>8d}  ({n_step3/n_valid_sensor*100:5.1f}%)")
    print(f"   d. Galactic Plane Filter (|b_zenith| >= {MILKY_WAY_MIN_GAL_LAT} deg):")
    print(f"      - Screened Out (Milky Way in Zenith):      {n_screened_step4:>8d}")
    print(f"      = Remaining (Milky Way Clear):             {n_step4:>8d}  ({n_step4/n_valid_sensor*100:5.1f}%)")
    print(f"   e. Zodiacal Light Filter (|beta_zenith| >= {ZODIACAL_MIN_ECL_LAT} deg):")
    print(f"      - Screened Out (Zodiacal Light in Zenith): {n_screened_step5:>8d}")
    print(f"      = Remaining (Zodiacal Clear):              {n_step5:>8d}  ({n_step5/n_valid_sensor*100:5.1f}%)")
    print("-" * 68)
    print(f"3. Final Clean Dark-Sky Measurements Retained:  {n_step5:>8d}  ({n_step5/n_valid_sensor*100:5.1f}%)")
    print("="*68 + "\n")
    
    print("Pipeline execution complete! Generated files:")
    print(f"  - {OUTPUT_CLEAN_CSV}")
    print(f"  - {OUTPUT_NIGHTLY_CSV}")
    print(f"  - {HISTOGRAM_CSV}")
    print(f"  - {RAW_TIMESERIES_PNG}")
    print(f"  - {RAW_VS_FILTERED_PNG}")
    print(f"  - {NIGHTLY_MEDIANS_PNG}")
    print(f"  - {HIST_FILTERED_PNG}")
    print(f"  - {HIST_STEPS_PNG}")
    print(f"  - {JELLYFISH_PNG}")

if __name__ == "__main__":
    run_pipeline()
