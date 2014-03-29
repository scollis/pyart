"""
"""

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('retrieve', parent_package, top_path)
    #config.add_data_dir('tests')
    config.add_extension('background',
                         sources=['background.pyf','src/background.f90'],
                         f2py_options=[])
    config.add_extension('continuity',
                         sources=['continuity.pyf','src/continuity.f90'],
                         f2py_options=[])
    config.add_extension('divergence',
                         sources=['divergence.pyf','src/divergence.f90'],
                         f2py_options=[])
    config.add_extension('gradient',
                         sources=['gradient.pyf','src/gradient.f90'],
                         f2py_options=[])
    config.add_extension('laplace',
                         sources=['laplace.pyf','src/laplace.f90'],
                         f2py_options=[])
    config.add_extension('smooth',
                         sources=['smooth.pyf','src/smooth.f90'],
                         f2py_options=[])
    
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
