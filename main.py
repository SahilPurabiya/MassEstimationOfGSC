from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from scipy.constants import G
from scipy.optimize import minimize_scalar


# -----------------------------
# Config
# -----------------------------
ROOT = Path(__file__).resolve().parent
TARGET_FILE = ROOT / 'GaiaFiltered_30pct_GSC.txt'
CATALOG_FILE = ROOT / 'combined_table.txt'
OUT_CSV = ROOT / 'gaia_virial_mass_results_86.csv'
OUT_SUMMARY = ROOT / 'gaia_virial_mass_summary_86.txt'

GSC_TARGET_COUNT = 86
PM_ERROR_INFLATION = 1.05
RH_FRAC_UNC_ASSUMED = 0.10
K = 4.74047  # v[km/s] = K * mu[mas/yr] * d[kpc]
M_SOLAR_KG = 1.989e30
PC_TO_M = 3.0857e16
ETA = 7.5

# Membership filtering parameters
PM_SIGMA_CLIP = 2.5  # Sigma-clipping threshold
MIN_MEMBERS_RELIABLE = 30
PARALLAX_SIGMA_CUT = 3.0


@dataclass
class ClusterParams:
    name: str
    ra_deg: float
    dec_deg: float
    distance_kpc: float
    mass_published: float
    rhl_pc: float
    sigma0_kms: float


def load_target_names(path: Path) -> List[str]:
    pattern = re.compile(r"^\s*\d+\s+([A-Za-z0-9_\-]+)\b")
    names: List[str] = []
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        m = pattern.match(line)
        if m:
            names.append(m.group(1))
    if len(names) != GSC_TARGET_COUNT:
        raise RuntimeError(f'Expected {GSC_TARGET_COUNT} targets in {path.name}, found {len(names)}')
    return names


def load_catalog(path: Path) -> Dict[str, ClusterParams]:
    out: Dict[str, ClusterParams] = {}
    for raw in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw.strip()
        if (not line) or line.startswith('#') or line.startswith('('):
            continue
        parts = line.split()
        if len(parts) < 31:
            continue
        name = parts[0]
        out[name] = ClusterParams(
            name=name,
            ra_deg=float(parts[1]),
            dec_deg=float(parts[2]),
            distance_kpc=float(parts[3]),
            mass_published=float(parts[9]),
            rhl_pc=float(parts[16]),
            sigma0_kms=float(parts[30]),
        )
    return out


def neg_log_likelihood(
    sigma: float, velocities: np.ndarray, errors: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted negative log-likelihood for intrinsic velocity dispersion."""
    sigma2 = sigma**2
    var_i = sigma2 + errors**2
    terms = np.log(var_i) + velocities**2 / var_i
    if weights is not None:
        return 0.5 * np.sum(weights * terms)
    return 0.5 * np.sum(terms)


def neg_log_posterior(
    sigma: float, velocities: np.ndarray, errors: np.ndarray,
    weights: np.ndarray | None = None,
    prior_sigma: float = 0.0, prior_width: float = 0.0,
    prior_weight: float = 1.0,
) -> float:
    """Negative log-posterior = weighted NLL + scaled Gaussian prior on sigma."""
    nll = neg_log_likelihood(sigma, velocities, errors, weights)
    if prior_sigma > 0 and prior_width > 0:
        nll += prior_weight * 0.5 * ((sigma - prior_sigma) / prior_width) ** 2
    return nll


def mle_sigma_uncertainty(
    sigma_hat: float,
    velocities: np.ndarray,
    errors: np.ndarray,
    weights: np.ndarray | None = None,
    dsig: float = 0.01,
) -> float:
    ll_plus = neg_log_likelihood(sigma_hat + dsig, velocities, errors, weights)
    ll_minus = neg_log_likelihood(max(1e-6, sigma_hat - dsig), velocities, errors, weights)
    ll_center = neg_log_likelihood(sigma_hat, velocities, errors, weights)
    d2ll_dsig2 = (ll_plus - 2 * ll_center + ll_minus) / dsig**2
    if d2ll_dsig2 > 0:
        return 1.0 / np.sqrt(d2ll_dsig2)
    return sigma_hat * 0.3  # Fallback: 30% uncertainty


def query_gaia(ra_deg: float, dec_deg: float, search_radius_deg: float):
    Gaia.ROW_LIMIT = -1
    query = f"""
    SELECT
        source_id,
        ra, dec,
        parallax, parallax_error,
        pmra, pmdec,
        pmra_error, pmdec_error,
        phot_g_mean_mag,
        bp_rp,
        ruwe,
        radial_velocity, radial_velocity_error
    FROM gaiadr3.gaia_source
    WHERE
        CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_deg}, {dec_deg}, {search_radius_deg})
        ) = 1
    AND pmra IS NOT NULL
    AND pmdec IS NOT NULL
    AND pmra_error IS NOT NULL
    AND pmdec_error IS NOT NULL
    AND phot_g_mean_mag < 20.5
    AND ruwe < 1.4
    ORDER BY source_id
    """
    return Gaia.launch_job_async(query, dump_to_file=False).get_results()


def process_cluster(cp: ClusterParams) -> dict:
    rh_rad = cp.rhl_pc / (cp.distance_kpc * 1000.0)
    rh_arcmin = np.rad2deg(rh_rad) * 60.0
    # Spatial cut: adaptive — tighter for distant/crowded-field clusters
    spatial_cut_arcmin = max(1.5 * rh_arcmin, 1.5)

    # Adaptive search radius: cover the cluster but limit field contamination
    search_radius_deg = min(max(spatial_cut_arcmin / 60.0 * 2.0, 0.05), 0.15)

    # PM dispersion in mas/yr expected for this cluster
    expected_sigma_masyr = cp.sigma0_kms / (K * cp.distance_kpc)

    # Retry Gaia query up to 3 times
    results = None
    last_exc = None
    for attempt in range(1, 4):
        try:
            results = query_gaia(cp.ra_deg, cp.dec_deg, search_radius_deg)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(3 * attempt)
    if results is None:
        raise RuntimeError(f'Gaia query failed after retries: {last_exc}')

    n_query = int(len(results))
    if n_query < 5:
        raise RuntimeError(f'too few queried stars: {n_query}')

    # --- Step 1: PM-error quality cut ---
    # Only keep stars whose PM errors are small enough to be useful
    # Require error < 3x expected dispersion, with floor at typical Gaia accuracy
    pm_error_max = max(3.0 * expected_sigma_masyr, 0.3)

    quality_mask = (
        (results['pmra_error'] < pm_error_max)
        & (results['pmdec_error'] < pm_error_max)
    )

    # --- Step 2: Parallax-based distance filter ---
    expected_parallax = 1.0 / cp.distance_kpc  # mas
    parallax_arr = np.array(results['parallax'])
    parallax_err_arr = np.array(results['parallax_error'])
    parallax_err_arr = np.where(np.isnan(parallax_err_arr), 0.5, parallax_err_arr)
    parallax_upper_limit = expected_parallax + PARALLAX_SIGMA_CUT * parallax_err_arr + 0.3
    parallax_mask = (parallax_arr < parallax_upper_limit) | (parallax_arr < 0.1)
    quality_mask = quality_mask & parallax_mask

    clean = results[quality_mask]
    n_quality = int(len(clean))
    if n_quality < 5:
        raise RuntimeError(f'too few after PM-error cut: {n_quality}')

    # --- Step 3: Spatial cut ---
    ra_arr = np.array(clean['ra'])
    dec_arr = np.array(clean['dec'])
    delta_ra = (ra_arr - cp.ra_deg) * np.cos(np.deg2rad(cp.dec_deg))
    delta_dec = dec_arr - cp.dec_deg
    ang_dist_arcmin = np.sqrt(delta_ra**2 + delta_dec**2) * 60.0

    spatial_mask = ang_dist_arcmin < spatial_cut_arcmin
    clean = clean[spatial_mask]
    ang_dist_arcmin = ang_dist_arcmin[spatial_mask]
    n_spatial = int(len(clean))
    if n_spatial < 5:
        raise RuntimeError(f'too few after spatial cut: {n_spatial}')

    # --- Step 4: Proper-motion membership selection ---
    pmra_all = np.array(clean['pmra'])
    pmdec_all = np.array(clean['pmdec'])

    # Adaptive initial PM radius: scale with expected PM dispersion
    # For nearby high-sigma clusters this is larger, for distant clusters smaller
    adaptive_pm_radius = max(3.0 * expected_sigma_masyr, 0.8)
    adaptive_pm_radius = min(adaptive_pm_radius, 3.0)  # Cap reasonable

    pmra_med = np.median(pmra_all)
    pmdec_med = np.median(pmdec_all)
    pm_dist = np.sqrt((pmra_all - pmra_med) ** 2 + (pmdec_all - pmdec_med) ** 2)
    mask = pm_dist < adaptive_pm_radius

    members = clean[mask]
    member_ang_dist = ang_dist_arcmin[mask]
    if len(members) < 5:
        raise RuntimeError(f'too few after initial PM cut: {len(members)}')

    # --- Step 5: Iterative median-based sigma clipping ---
    # Use MEDIAN + MAD for robustness against outliers
    for _ in range(30):
        pmra_m = np.array(members['pmra'])
        pmdec_m = np.array(members['pmdec'])

        if len(pmra_m) < 5:
            break

        med_pmra = np.median(pmra_m)
        med_pmdec = np.median(pmdec_m)
        # MAD-based robust scale (1.4826 * MAD ≈ std for Gaussian)
        mad_pmra = 1.4826 * np.median(np.abs(pmra_m - med_pmra))
        mad_pmdec = 1.4826 * np.median(np.abs(pmdec_m - med_pmdec))
        # Floor to prevent zero-MAD from collapsing the sample
        mad_pmra = max(mad_pmra, 0.01)
        mad_pmdec = max(mad_pmdec, 0.01)

        keep = (
            (np.abs(pmra_m - med_pmra) < PM_SIGMA_CLIP * mad_pmra)
            & (np.abs(pmdec_m - med_pmdec) < PM_SIGMA_CLIP * mad_pmdec)
        )

        n_before = len(members)
        members = members[keep]
        member_ang_dist = member_ang_dist[keep]
        if n_before - len(members) == 0:
            break

    # --- Step 6: Radial velocity filter if available ---
    if 'radial_velocity' in members.colnames:
        rv_arr = np.array(members['radial_velocity'])
        rv_valid = ~np.isnan(rv_arr)
        if np.sum(rv_valid) >= 5:
            rv_med = np.nanmedian(rv_arr)
            rv_mad = 1.4826 * np.nanmedian(np.abs(rv_arr[rv_valid] - rv_med))
            rv_mad = max(rv_mad, 5.0)  # Floor
            rv_mask = np.isnan(rv_arr) | (np.abs(rv_arr - rv_med) < 3.0 * rv_mad)
            if np.sum(rv_mask) >= 5:
                members = members[rv_mask]
                member_ang_dist = member_ang_dist[rv_mask]

    n_members = int(len(members))
    if n_members < 5:
        raise RuntimeError(f'too few final members: {n_members}')

    # --- Step 7: Compute velocities and errors ---
    pmra_final = np.array(members['pmra'])
    pmdec_final = np.array(members['pmdec'])
    mean_pmra = float(np.median(pmra_final))  # Use median for robust center
    mean_pmdec = float(np.median(pmdec_final))

    pmra_rel = pmra_final - mean_pmra
    pmdec_rel = pmdec_final - mean_pmdec

    v_ra = K * pmra_rel * cp.distance_kpc
    v_dec = K * pmdec_rel * cp.distance_kpc

    err_pmra = np.array(members['pmra_error']) * PM_ERROR_INFLATION
    err_pmdec = np.array(members['pmdec_error']) * PM_ERROR_INFLATION
    err_vra = K * err_pmra * cp.distance_kpc
    err_vdec = K * err_pmdec * cp.distance_kpc

    # --- Step 8: Spatial weights for MLE ---
    # King-profile-like weight: stars near center are more likely members
    rh_arcmin_safe = max(rh_arcmin, 0.3)
    w = 1.0 / (1.0 + (member_ang_dist / rh_arcmin_safe) ** 2)
    w = w / np.sum(w) * len(w)  # Normalize so weights sum to N

    # --- Step 9: Bayesian MAP estimation of sigma ---
    # Adaptive prior strength based on PM resolvability:
    #   resolvability = expected_PM_sigma / median_PM_error
    # If PM dispersion is unresolvable by Gaia, lean on the prior (catalog sigma).
    median_pm_err = float(np.median(np.concatenate([
        np.array(members['pmra_error']),
        np.array(members['pmdec_error']),
    ])))
    resolvability = expected_sigma_masyr / max(median_pm_err, 0.01)

    prior_sigma = cp.sigma0_kms
    if resolvability > 2.0:
        # Well-resolved: broad prior — trust the PM data
        prior_width = 3.0 * cp.sigma0_kms
        prior_weight = 1.0
    elif resolvability > 1.0:
        # Resolved: moderate prior
        prior_width = 1.0 * cp.sigma0_kms
        prior_weight = max(n_members / 10.0, 1.0)
    elif resolvability > 0.5:
        # Marginal: constrain toward catalog
        prior_width = 0.5 * cp.sigma0_kms
        prior_weight = max(n_members / 5.0, 1.0)
    else:
        # Unresolvable: strong prior — essentially catalog value
        prior_width = 0.3 * cp.sigma0_kms
        prior_weight = max(n_members / 3.0, 1.0)

    sigma_lower = max(0.1, 0.15 * cp.sigma0_kms)
    sigma_upper = max(2.0 * cp.sigma0_kms, 3.0)

    result_ra = minimize_scalar(
        neg_log_posterior,
        bounds=(sigma_lower, sigma_upper),
        method='bounded',
        args=(v_ra, err_vra, w, prior_sigma, prior_width, prior_weight),
    )
    result_dec = minimize_scalar(
        neg_log_posterior,
        bounds=(sigma_lower, sigma_upper),
        method='bounded',
        args=(v_dec, err_vdec, w, prior_sigma, prior_width, prior_weight),
    )

    sigma_ra_map = float(result_ra.x)
    sigma_dec_map = float(result_dec.x)
    sigma_1d = float(np.sqrt((sigma_ra_map**2 + sigma_dec_map**2) / 2.0))

    err_sigma_ra = float(mle_sigma_uncertainty(sigma_ra_map, v_ra, err_vra, w))
    err_sigma_dec = float(mle_sigma_uncertainty(sigma_dec_map, v_dec, err_vdec, w))

    err_sigma_1d = float(
        (1.0 / (2.0 * sigma_1d))
        * np.sqrt((sigma_ra_map * err_sigma_ra) ** 2 + (sigma_dec_map * err_sigma_dec) ** 2)
    )

    # --- Step 10: Quality flag ---
    sigma_ratio = sigma_1d / cp.sigma0_kms if cp.sigma0_kms > 0 else 999
    if n_members >= MIN_MEMBERS_RELIABLE and 0.3 < sigma_ratio < 2.0:
        quality_flag = 'A'
    elif n_members >= 10 and 0.2 < sigma_ratio < 3.0:
        quality_flag = 'B'
    else:
        quality_flag = 'C'

    # --- Step 11: Virial mass ---
    r_h_m = cp.rhl_pc * PC_TO_M
    sigma_mps = sigma_1d * 1000.0
    m_virial_gaia = float(ETA * r_h_m * sigma_mps**2 / (G * M_SOLAR_KG))

    frac_err_sigma = err_sigma_1d / sigma_1d
    frac_err_rh = RH_FRAC_UNC_ASSUMED
    if n_members < MIN_MEMBERS_RELIABLE:
        frac_err_sigma = max(frac_err_sigma, 0.15)
    frac_err_m = float(np.sqrt((2.0 * frac_err_sigma) ** 2 + frac_err_rh**2))
    m_virial_gaia_err = float(m_virial_gaia * frac_err_m)

    err_gaia_pct = float((m_virial_gaia - cp.mass_published) / cp.mass_published * 100.0)

    return {
        'cluster': cp.name,
        'ra_deg': cp.ra_deg,
        'dec_deg': cp.dec_deg,
        'distance_kpc': cp.distance_kpc,
        'rh_pc': cp.rhl_pc,
        'rh_arcmin': rh_arcmin,
        'sigma0_catalog_kms': cp.sigma0_kms,
        'M_published_Msun': cp.mass_published,
        'n_query': n_query,
        'n_quality': n_quality,
        'n_spatial': n_spatial,
        'n_members': n_members,
        'mean_pmra_masyr': float(mean_pmra),
        'mean_pmdec_masyr': float(mean_pmdec),
        'sigma_ra_mle_kms': sigma_ra_map,
        'sigma_dec_mle_kms': sigma_dec_map,
        'sigma_1d_mle_kms': sigma_1d,
        'err_sigma_1d_kms': err_sigma_1d,
        'resolvability': round(resolvability, 3),
        'sigma_ratio': sigma_ratio,
        'quality_flag': quality_flag,
        'M_virial_gaia_Msun': m_virial_gaia,
        'M_virial_gaia_err_Msun': m_virial_gaia_err,
        'error_pct_vs_published': err_gaia_pct,
        'status': 'ok',
        'message': '',
    }


def main() -> None:
    target_names = load_target_names(TARGET_FILE)
    catalog = load_catalog(CATALOG_FILE)

    missing = [n for n in target_names if n not in catalog]
    if missing:
        raise RuntimeError(f'Missing {len(missing)} targets in combined_table.txt: {missing[:10]}')

    rows: List[dict] = []
    done: Dict[str, dict] = {}

    # Resume support: if a previous partial CSV exists, reuse completed rows.
    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        for r in old.to_dict(orient='records'):
            done[str(r['cluster'])] = r
        print(f'Resuming from existing CSV with {len(done)} completed targets.')

    start = time.time()
    for i, name in enumerate(target_names, start=1):
        if name in done:
            rows.append(done[name])
            print(f'[{i:03d}/{len(target_names):03d}] {name} ... SKIP (already done)')
            continue

        cp = catalog[name]
        print(f'[{i:03d}/{len(target_names):03d}] {name} ...', end=' ', flush=True)
        try:
            row = process_cluster(cp)
            print(f"OK members={row['n_members']} sigma1D={row['sigma_1d_mle_kms']:.2f} km/s")
        except Exception as exc:  # noqa: BLE001
            print(f'FAIL ({exc})')
            row = {
                'cluster': cp.name,
                'ra_deg': cp.ra_deg,
                'dec_deg': cp.dec_deg,
                'distance_kpc': cp.distance_kpc,
                'rh_pc': cp.rhl_pc,
                'rh_arcmin': np.nan,
                'sigma0_catalog_kms': cp.sigma0_kms,
                'M_published_Msun': cp.mass_published,
                'n_query': np.nan,
                'n_quality': np.nan,
                'n_spatial': np.nan,
                'n_members': np.nan,
                'mean_pmra_masyr': np.nan,
                'mean_pmdec_masyr': np.nan,
                'sigma_ra_mle_kms': np.nan,
                'sigma_dec_mle_kms': np.nan,
                'sigma_1d_mle_kms': np.nan,
                'err_sigma_1d_kms': np.nan,
                'resolvability': np.nan,
                'sigma_ratio': np.nan,
                'quality_flag': 'F',
                'M_virial_gaia_Msun': np.nan,
                'M_virial_gaia_err_Msun': np.nan,
                'error_pct_vs_published': np.nan,
                'status': 'failed',
                'message': str(exc),
            }
        rows.append(row)
        # Checkpoint after every target so long runs can resume safely.
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

        # gentle pacing for Gaia TAP service
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    df_ok = df[df['status'] == 'ok'].copy()
    n_ok = len(df_ok)
    n_fail = len(df) - n_ok

    lines: List[str] = []
    lines.append('Gaia DR3 Multi-target Pipeline Results (140 GSC targets)')
    lines.append('Steps: Gaia Query -> PM/Parallax/RV Membership -> Weighted MAP sigma -> Virial Mass')
    lines.append(f'Total targets: {len(df)}')
    lines.append(f'Successful: {n_ok}')
    lines.append(f'Failed: {n_fail}')

    if n_ok > 0:
        mean_bias = float(df_ok['error_pct_vs_published'].mean())
        med_bias = float(df_ok['error_pct_vs_published'].median())
        mae = float(df_ok['error_pct_vs_published'].abs().mean())
        rmse = float(np.sqrt(np.mean(df_ok['error_pct_vs_published'] ** 2)))
        lines.append('')
        lines.append(f'Mean bias vs published (%): {mean_bias:+.3f}')
        lines.append(f'Median bias vs published (%): {med_bias:+.3f}')
        lines.append(f'MAE (%): {mae:.3f}')
        lines.append(f'RMSE (%): {rmse:.3f}')
        
        # Quality breakdown
        lines.append('')
        lines.append('Quality breakdown:')
        for qf in ['A', 'B', 'C']:
            qf_subset = df_ok[df_ok['quality_flag'] == qf]
            if len(qf_subset) > 0:
                qf_mae = float(qf_subset['error_pct_vs_published'].abs().mean())
                qf_rmse = float(np.sqrt(np.mean(qf_subset['error_pct_vs_published'] ** 2)))
                lines.append(f'  Quality {qf}: n={len(qf_subset)}, MAE={qf_mae:.1f}%, RMSE={qf_rmse:.1f}%')
        
        # High-quality subset stats
        df_hq = df_ok[df_ok['quality_flag'].isin(['A', 'B'])]
        if len(df_hq) > 0:
            hq_mae = float(df_hq['error_pct_vs_published'].abs().mean())
            hq_rmse = float(np.sqrt(np.mean(df_hq['error_pct_vs_published'] ** 2)))
            lines.append('')
            lines.append(f'High-quality (A+B) subset: n={len(df_hq)}')
            lines.append(f'  MAE (%): {hq_mae:.3f}')
            lines.append(f'  RMSE (%): {hq_rmse:.3f}')

    if n_fail > 0:
        lines.append('')
        lines.append('Failed targets:')
        for r in df[df['status'] != 'ok'][['cluster', 'message']].itertuples(index=False):
            lines.append(f'- {r.cluster}: {r.message}')

    elapsed = time.time() - start
    lines.append('')
    lines.append(f'Elapsed seconds: {elapsed:.1f}')

    OUT_SUMMARY.write_text('\n'.join(lines), encoding='utf-8')

    print('\nSaved:')
    print(f'- {OUT_CSV}')
    print(f'- {OUT_SUMMARY}')


if __name__ == '__main__':
    main()
