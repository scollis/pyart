"""
Temporal interpolation of radar volumes by advection.

Reconstruct a radar volume at a time between two observed volumes by estimating
a dense, height-resolved motion field with optical flow and advecting the
bracketing volumes to the target time before blending. Unlike a single rigid
advection vector (see :py:func:`pyart.retrieve.grid_displacement_pc`), the
spatially varying field handles storms in which convective cells and stratiform
regions move differently.

The interpolation is performed on the radar's native (azimuth, range, elevation)
gate geometry, so the returned object is an ordinary
:py:class:`~pyart.core.Radar` rather than a Cartesian grid.

"""

import copy

import numpy as np
from netCDF4 import num2date
from scipy.interpolate import RegularGridInterpolator

from ..exceptions import MissingOptionalDependency
from ..map import grid_from_radars

try:
    from skimage.registration import optical_flow_tvl1

    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


def _volume_mean_time(radar):
    """Representative time of a volume as a datetime (mean of the ray times)."""
    return num2date(np.mean(radar.time["data"]), radar.time["units"])


def _normalize(field_data, vmin, vmax, floor):
    """Map a (masked) reflectivity-like field to [0, 1] for optical flow.

    Non-finite / masked gates are set to ``floor`` (below ``vmin``) so that
    no-echo regions read as a uniform low background rather than as NaNs.
    """
    data = np.ma.filled(np.ma.masked_invalid(field_data).astype("float64"), floor)
    data = np.clip(data, vmin, vmax)
    return (data - vmin) / (vmax - vmin)


def grid_optical_flow(
    grid1,
    grid2,
    field,
    vmin=0.0,
    vmax=60.0,
    floor=-10.0,
    attachment=15.0,
    tightness=0.3,
    num_warp=5,
    num_iter=10,
):
    """
    Estimate a dense motion field between two grids using optical flow.

    A total-variation L1 optical flow (TV-L1) is computed independently on each
    vertical level of the two grids and converted to a physical echo
    displacement. The displacement is defined in the ``grid1`` -> ``grid2``
    sense: a feature located at ``(x, y)`` in ``grid1`` is found near
    ``(x + disp_x, y + disp_y)`` in ``grid2``.

    Requires scikit-image.

    Parameters
    ----------
    grid1, grid2 : Grid
        Py-ART Grid objects separated in time, sharing the same shape and axes.
    field : str
        Name of the field to track. Must be present in both grids.
    vmin, vmax : float, optional
        Reflectivity range (dBZ) used to normalize the field to [0, 1] before
        estimating flow. Defaults to 0 and 60.
    floor : float, optional
        Value assigned to masked / no-echo gates before normalization. Should
        be at or below ``vmin``. Default -10.
    attachment, tightness, num_warp, num_iter : optional
        Parameters passed to :py:func:`skimage.registration.optical_flow_tvl1`.

    Returns
    -------
    disp_y, disp_x : ndarray
        Physical echo displacement in metres from ``grid1`` to ``grid2``, each
        with shape ``(nz, ny, nx)`` matching the grid. ``disp_x`` is the
        eastward (grid-x) component and ``disp_y`` the northward (grid-y)
        component. Divide by the time separation of the grids to obtain a
        velocity.

    See Also
    --------
    grid_displacement_pc : single rigid displacement by phase correlation.
    advection_interpolate : reconstruct an intermediate volume using this field.

    """
    if not _SKIMAGE_AVAILABLE:
        raise MissingOptionalDependency(
            "scikit-image is required for grid_optical_flow but is not installed"
        )

    data1 = np.ma.filled(grid1.fields[field]["data"], np.nan)
    data2 = np.ma.filled(grid2.fields[field]["data"], np.nan)
    if data1.shape != data2.shape:
        raise ValueError("grid1 and grid2 must have identical shapes")

    dx = float(grid1.x["data"][1] - grid1.x["data"][0])
    dy = float(grid1.y["data"][1] - grid1.y["data"][0])

    disp_x = np.zeros_like(data1, dtype="float64")
    disp_y = np.zeros_like(data1, dtype="float64")
    for kz in range(data1.shape[0]):
        # optical_flow_tvl1(reference, moving) returns (v, u) such that
        # reference(x, y) ~ moving(x + u, y + v). With grid1 as reference and
        # grid2 as moving, (u, v) is the grid1 -> grid2 echo displacement in
        # pixels; multiply by spacing for metres.
        flow_v, flow_u = optical_flow_tvl1(
            _normalize(data1[kz], vmin, vmax, floor),
            _normalize(data2[kz], vmin, vmax, floor),
            attachment=attachment,
            tightness=tightness,
            num_warp=num_warp,
            num_iter=num_iter,
        )
        disp_x[kz] = flow_u * dx
        disp_y[kz] = flow_v * dy
    return disp_y, disp_x


def _fill_nonecho(disp, echo_mask):
    """Replace non-echo displacement with the per-level mean (defined everywhere)."""
    out = disp.copy()
    for kz in range(out.shape[0]):
        lvl = out[kz]
        m = echo_mask[kz]
        lvl[~m] = lvl[m].mean() if m.any() else 0.0
    return out


def _displacement_interpolators(disp_y, disp_x, echo_mask, z, y, x):
    """RegularGridInterpolators for (disp_y, disp_x) over (z, y, x)."""
    dy = _fill_nonecho(disp_y, echo_mask)
    dx = _fill_nonecho(disp_x, echo_mask)
    kw = dict(bounds_error=False, fill_value=None)
    fy = RegularGridInterpolator((z, y, x), dy, **kw)
    fx = RegularGridInterpolator((z, y, x), dx, **kw)
    return fy, fx


def _sweep_sampler(radar, sweep, field):
    """Return (azimuth_sorted_deg, range_m, values, ) for one sweep, az-sorted."""
    start, end = radar.get_start_end(sweep)
    az = radar.azimuth["data"][start : end + 1]
    data = np.ma.filled(
        radar.fields[field]["data"][start : end + 1].astype("float64"), np.nan
    )
    order = np.argsort(az)
    return az[order], radar.range["data"], data[order]


def _sample_native(az_query, range_query, sampler):
    """Bilinear-sample a sweep at query azimuth (deg) / slant range (m).

    The azimuth axis is tiled by +/- 360 degrees so that queries wrap across
    the 0/360 discontinuity.
    """
    az_sorted, range_sorted, values = sampler
    az_ext = np.concatenate([az_sorted - 360.0, az_sorted, az_sorted + 360.0])
    val_ext = np.concatenate([values, values, values], axis=0)
    rgi = RegularGridInterpolator(
        (az_ext, range_sorted), val_ext, bounds_error=False, fill_value=np.nan
    )
    pts = np.column_stack([az_query.ravel(), range_query.ravel()])
    return rgi(pts).reshape(az_query.shape)


def advection_interpolate(
    radar1,
    radar2,
    alpha=None,
    target_time=None,
    field=None,
    grid_shape=(8, 241, 241),
    grid_limits=((1000.0, 8000.0), (-120000.0, 120000.0), (-120000.0, 120000.0)),
    echo_threshold=5.0,
    interp_field_name=None,
    gridding_kwargs=None,
    **flow_kwargs,
):
    """
    Reconstruct a radar volume at a time between two observed volumes.

    A dense, height-resolved motion field is estimated from gridded versions of
    the two input volumes with :py:func:`grid_optical_flow`, then each gate of
    the output volume is filled by advecting the two input volumes to the target
    time along that field and blending. The output is on the native geometry of
    ``radar1``.

    Requires scikit-image.

    Parameters
    ----------
    radar1, radar2 : Radar
        Radar volumes bracketing the target time (``radar1`` earlier). They must
        contain ``field`` and are assumed to share a scan strategy.
    alpha : float, optional
        Fractional time of the target between ``radar1`` (0.0) and ``radar2``
        (1.0). If None, it is derived from ``target_time`` or defaults to 0.5.
    target_time : datetime, optional
        Absolute target time. Used to compute ``alpha`` from the mean volume
        times when ``alpha`` is not given.
    field : str, optional
        Field to interpolate. Defaults to ``"reflectivity"`` if present.
    grid_shape : 3-tuple of int, optional
        (nz, ny, nx) of the intermediate Cartesian grid used for flow estimation.
    grid_limits : 3-tuple of 2-tuple of float, optional
        ((zmin, zmax), (ymin, ymax), (xmin, xmax)) in metres for that grid.
    echo_threshold : float, optional
        Field value (dBZ) above which a grid cell is treated as echo when
        extending the motion field into no-echo regions. Default 5.
    interp_field_name : str, optional
        Name of the field in the returned radar. Defaults to ``field``.
    gridding_kwargs : dict, optional
        Extra keyword arguments passed to
        :py:func:`pyart.map.grid_from_radars`.
    **flow_kwargs
        Passed through to :py:func:`grid_optical_flow`.

    Returns
    -------
    radar_out : Radar
        A copy of ``radar1`` whose ``interp_field_name`` field holds the
        reconstructed volume at the target time.

    Notes
    -----
    For a scene translating rigidly at a known velocity the method recovers the
    motion direction and reconstructs the intermediate volume to within a small
    fraction of a dBZ. The advection advantage over a plain time average is
    largest at long range, where gates are large and echo moves several
    gate-widths between volumes. Purely kinematic morphing cannot represent
    storm growth or decay, so residual error concentrates at cell edges.

    See Also
    --------
    grid_optical_flow : the dense motion-field estimator used here.
    grid_displacement_pc : single rigid displacement by phase correlation.

    """
    if not _SKIMAGE_AVAILABLE:
        raise MissingOptionalDependency(
            "scikit-image is required for advection_interpolate but is not installed"
        )

    if field is None:
        field = "reflectivity"
    if interp_field_name is None:
        interp_field_name = field
    if gridding_kwargs is None:
        gridding_kwargs = {}

    # target fractional time
    if alpha is None:
        if target_time is not None:
            t1 = _volume_mean_time(radar1)
            t2 = _volume_mean_time(radar2)
            span = (t2 - t1).total_seconds()
            alpha = 0.5 if span == 0 else (target_time - t1).total_seconds() / span
        else:
            alpha = 0.5
    alpha = float(alpha)

    # grid both volumes and estimate the dense motion field
    grid_kw = dict(
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        fields=[field],
        weighting_function="Barnes2",
        roi_func="dist_beam",
        min_radius=1000.0,
    )
    grid_kw.update(gridding_kwargs)
    grid1 = grid_from_radars(radar1, **grid_kw)
    grid2 = grid_from_radars(radar2, **grid_kw)

    disp_y, disp_x = grid_optical_flow(grid1, grid2, field, **flow_kwargs)

    z = grid1.z["data"]
    y = grid1.y["data"]
    x = grid1.x["data"]
    d1 = np.ma.filled(grid1.fields[field]["data"], np.nan)
    d2 = np.ma.filled(grid2.fields[field]["data"], np.nan)
    echo = (d1 > echo_threshold) | (d2 > echo_threshold)
    fy, fx = _displacement_interpolators(disp_y, disp_x, echo, z, y, x)

    # advect each gate of radar1's geometry to the target time and blend.
    # displacement disp is the physical grid1 -> grid2 echo motion, so a feature
    # seen at gate g at the target time was at g - alpha * disp in radar1 and at
    # g + (1 - alpha) * disp in radar2.
    zlo, zhi = z[0], z[-1]
    out = np.full((radar1.nrays, radar1.ngates), np.nan)
    gx_all = radar1.gate_x["data"]
    gy_all = radar1.gate_y["data"]
    gz_all = radar1.gate_z["data"]
    for sweep in range(radar1.nsweeps):
        start, end = radar1.get_start_end(sweep)
        elev = np.deg2rad(radar1.fixed_angle["data"][sweep])
        cos_e = max(np.cos(elev), 1.0e-3)
        gx = gx_all[start : end + 1]
        gy = gy_all[start : end + 1]
        gz = gz_all[start : end + 1]
        pts = np.column_stack(
            [np.clip(gz, zlo, zhi).ravel(), gy.ravel(), gx.ravel()]
        )
        du = fx(pts).reshape(gx.shape)
        dv = fy(pts).reshape(gx.shape)

        x1 = gx - alpha * du
        y1 = gy - alpha * dv
        x2 = gx + (1.0 - alpha) * du
        y2 = gy + (1.0 - alpha) * dv

        az1 = np.rad2deg(np.arctan2(x1, y1)) % 360.0
        r1 = np.hypot(x1, y1) / cos_e
        az2 = np.rad2deg(np.arctan2(x2, y2)) % 360.0
        r2 = np.hypot(x2, y2) / cos_e

        v1 = _sample_native(az1, r1, _sweep_sampler(radar1, sweep, field))
        v2 = _sample_native(az2, r2, _sweep_sampler(radar2, sweep, field))

        both = np.isfinite(v1) & np.isfinite(v2)
        blended = np.where(
            both,
            (1.0 - alpha) * v1 + alpha * v2,
            np.where(np.isfinite(v1), v1, v2),
        )
        out[start : end + 1] = blended

    radar_out = copy.deepcopy(radar1)
    field_dict = copy.deepcopy(radar1.fields[field])
    field_dict["data"] = np.ma.masked_invalid(out)
    radar_out.fields = {interp_field_name: field_dict}
    return radar_out
