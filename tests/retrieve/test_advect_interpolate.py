"""Unit tests for Py-ART's retrieve/advect_interpolate.py module.

Ground truth is an analytic scene translating at a known velocity
(:py:func:`pyart.testing.make_advection_interpolation_triplet`), so the tests
assert that the framework recovers the prescribed motion, reconstructs the true
intermediate volume, beats a no-motion time average, and honours the warp-sign
convention. scikit-image is required; tests are skipped without it.
"""

import numpy as np
import pytest

import pyart
from pyart.retrieve.advect_interpolate import _SKIMAGE_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _SKIMAGE_AVAILABLE, reason="scikit-image required for advection interpolation"
)

U_TRUE, V_TRUE, DT, ALPHA = -12.0, 8.0, 420.0, 0.5


def _score(pred, truth, thresh=15.0):
    m = np.isfinite(pred) & np.isfinite(truth) & (truth > thresh)
    err = pred[m] - truth[m]
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "cc": float(np.corrcoef(pred[m], truth[m])[0, 1]),
        "n": int(m.sum()),
    }


@pytest.fixture(scope="module")
def triplet():
    return pyart.testing.make_advection_interpolation_triplet(
        u=U_TRUE, v=V_TRUE, dt=DT, alpha=ALPHA
    )


# ---- synthetic-scene sanity ----


def test_scene_geometry_and_coverage():
    radar = pyart.testing.make_advecting_scene_radar()
    assert radar.nsweeps == 6
    assert radar.nrays == 6 * 360
    data = np.ma.filled(radar.fields["reflectivity"]["data"], np.nan)
    coverage = np.isfinite(data).mean()
    assert 0.2 < coverage < 0.6, f"expected partial coverage, got {coverage:.2f}"
    assert np.nanmax(data) > 45, "scene should contain convective-intensity echo"


# ---- motion recovery ----


def test_grid_optical_flow_recovers_motion(triplet):
    t1, _, t3 = triplet
    grid_kw = dict(
        grid_shape=(8, 241, 241),
        grid_limits=((1000.0, 8000.0), (-120000.0, 120000.0), (-120000.0, 120000.0)),
        fields=["reflectivity"],
        weighting_function="Barnes2",
        roi_func="dist_beam",
        min_radius=1000.0,
    )
    g1 = pyart.map.grid_from_radars(t1, **grid_kw)
    g3 = pyart.map.grid_from_radars(t3, **grid_kw)
    disp_y, disp_x = pyart.retrieve.grid_optical_flow(g1, g3, "reflectivity")

    d1 = np.ma.filled(g1.fields["reflectivity"]["data"], np.nan)
    d3 = np.ma.filled(g3.fields["reflectivity"]["data"], np.nan)
    echo = (d1 > 15) | (d3 > 15)
    # displacement is the physical grid1 -> grid2 echo motion; velocity = disp / dt
    u_rec = np.median(disp_x[echo]) / DT
    v_rec = np.median(disp_y[echo]) / DT

    ang_rec = np.degrees(np.arctan2(u_rec, v_rec)) % 360
    ang_true = np.degrees(np.arctan2(U_TRUE, V_TRUE)) % 360
    assert abs(ang_rec - ang_true) < 5.0, f"bearing {ang_rec:.1f} vs {ang_true:.1f}"

    spd_rec, spd_true = np.hypot(u_rec, v_rec), np.hypot(U_TRUE, V_TRUE)
    assert 0.75 * spd_true < spd_rec < 1.25 * spd_true, (
        f"speed {spd_rec:.1f} vs {spd_true:.1f}"
    )


# ---- reconstruction quality ----


def test_reconstructs_true_intermediate_volume(triplet):
    t1, t2_true, t3 = triplet
    out = pyart.retrieve.advection_interpolate(t1, t3, alpha=ALPHA)
    pred = np.ma.filled(out.fields["reflectivity"]["data"].astype(float), np.nan)
    truth = np.ma.filled(t2_true.fields["reflectivity"]["data"].astype(float), np.nan)
    s = _score(pred, truth)
    assert s["rmse"] < 2.0, f"reconstruction RMSE too high: {s['rmse']:.2f}"
    assert s["cc"] > 0.95, f"reconstruction CC too low: {s['cc']:.3f}"
    assert abs(s["bias"]) < 1.0


def test_morph_beats_linear_average(triplet):
    t1, t2_true, t3 = triplet
    truth = np.ma.filled(t2_true.fields["reflectivity"]["data"].astype(float), np.nan)
    out = pyart.retrieve.advection_interpolate(t1, t3, alpha=ALPHA)
    pred = np.ma.filled(out.fields["reflectivity"]["data"].astype(float), np.nan)

    # no-motion baseline: average the two volumes on t1's geometry (they share it)
    v1 = np.ma.filled(t1.fields["reflectivity"]["data"].astype(float), np.nan)
    v3 = np.ma.filled(t3.fields["reflectivity"]["data"].astype(float), np.nan)
    lin = np.nanmean(np.stack([v1, v3]), axis=0)

    assert _score(pred, truth)["rmse"] < _score(lin, truth)["rmse"]


def test_output_is_radar_on_native_geometry(triplet):
    t1, _, t3 = triplet
    out = pyart.retrieve.advection_interpolate(t1, t3, alpha=ALPHA)
    assert isinstance(out, pyart.core.Radar)
    assert out.nrays == t1.nrays and out.ngates == t1.ngates
    assert "reflectivity" in out.fields


# ---- warp-sign guard (regression protection for the historical sign bug) ----


def test_alpha_endpoints_recover_inputs(triplet):
    """alpha=0 must reproduce t1; alpha=1 must reproduce t3. A flipped warp sign
    breaks this, so the test guards the +alpha displacement convention."""
    t1, _, t3 = triplet
    v1 = np.ma.filled(t1.fields["reflectivity"]["data"].astype(float), np.nan)
    v3 = np.ma.filled(t3.fields["reflectivity"]["data"].astype(float), np.nan)

    out0 = pyart.retrieve.advection_interpolate(t1, t3, alpha=0.0)
    p0 = np.ma.filled(out0.fields["reflectivity"]["data"].astype(float), np.nan)
    assert _score(p0, v1)["rmse"] < 1.0, "alpha=0 should reproduce t1"

    out1 = pyart.retrieve.advection_interpolate(t1, t3, alpha=1.0)
    p1 = np.ma.filled(out1.fields["reflectivity"]["data"].astype(float), np.nan)
    assert _score(p1, v3)["rmse"] < 1.0, "alpha=1 should reproduce t3"


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_intermediate_alpha_tracks_truth(alpha):
    """Reconstruction at intermediate alpha matches the true volume at that time."""
    t1, t2_true, t3 = pyart.testing.make_advection_interpolation_triplet(
        u=U_TRUE, v=V_TRUE, dt=DT, alpha=alpha
    )
    out = pyart.retrieve.advection_interpolate(t1, t3, alpha=alpha)
    pred = np.ma.filled(out.fields["reflectivity"]["data"].astype(float), np.nan)
    truth = np.ma.filled(t2_true.fields["reflectivity"]["data"].astype(float), np.nan)
    assert _score(pred, truth)["rmse"] < 2.0
