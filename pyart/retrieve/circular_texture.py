"""
pyart.correct.circular_textur
=========================
Application of a circular texture algorithm that works on data that
folds between an interval to produce a texture field for future
proocessing
.. autosummary::
    :toctree: generated/
    calculate_attenuation
"""
import copy

import numpy as np
from scipy.signal importfind_peaks_cwt
from ..config import get_metadata
from ..util import interval_std
from scipy.ndimage import filters

def generic_circ_filter(radar, field, interval,
        texture_footprint = (4,4), median_footprint = None):
    """
    Use the standard deviation of a moment that is valid (ie folds)
    between an interval to calculate a texture field
    ----------
    radar : Radar
        Radar object containing field to calculate the texture fron
    interval : two element list
        folding interval
    texture_footprint: twople
        footprint of the standard deviation calculation in azimuth index and
        range index. Defaults to (4,4)
    median_footprint: twople
        footprint of the median filter to be applied after
        in azimuth index and range index. Defaults to None which means no
        median filter applied.
    Returns
    -------
    texture_field : dict
        Field dictionary containing texure information. Texture
        array is stored under the 'data' key.
    """
    data = filters.generic_filter(\
            radar.fields[field]['data'],
            interval_std, size = footprint,
            extra_arguments = interval)
    if median_footprint == None:
        filtered_data = data
    else:
        filtered_data = filters.median_filter(data, size = median_footprint)
    texture_field = pyart.config.get_metadata(key)
    texture_field['data'] = filtered_data
    texture_field['standard_name'] = 'texture_of_' +\
            texture_field['standard_name']
    return texture_field



