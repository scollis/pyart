"""
pyart.retrieve.texture
=========================

texture

.. autosummary::
    :toctree: generated/

    texture
"""

from ..correct.phase_proc import smooth_and_trim
from numpy import zeros_like, sqrt

def texture(field, npts):
    """
    Return the root mean squared "noise" where the mean is a butterworth
    filter of length npts

    Parameters
    ----------
    field : Field
        Field object containing the moment/measurement from which the texture
        will be derived
    npts : the number of points over which the butterworth filter will be
        performed

    Other Parameters
    ----------------
    None

    Returns
    -------
    texture_field : Field
        A field containing the texture information.
    """

    texture_field = field.copy()
    tex_array = zeros_like(field['data'])
    for i in range(texture_field['data'].shape[0]):
        this_ray_of_data = field['data'][i,:]
        mean = smooth_and_trim(this_ray_of_data, window_len = npts)
        texture = smooth_and_trim(\
                     sqrt((this_ray_of_data - mean) ** 2), window_len = npts)
        tex_array[i,:] = texture
    texture_field['data'] = tex_array
    texture_field['standard_name'] = "texture of " \
                                                + texture_field['standard_name']
    texture_field['units'] = 'unitless'
    return texture_field
