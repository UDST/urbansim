version = '1.1dev'

from .patsypatch import patch_patsy
patch_patsy()

from .geopandaspatch import patch_geopandas
patch_geopandas()
