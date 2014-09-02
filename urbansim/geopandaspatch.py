from urbansim.maps import dframe_explorer
import pandas as pd


def explore(self,
            dataframe_d=None,
            center=None,
            zoom=11,
            geom_name=None,  # from JSON file, use index if None
            join_name='zone_id',  # from data frames
            precision=2,
            port=8765,
            host='localhost',
            testing=False):

    if dataframe_d is None:
        dataframe_d = {}

    # add the geodataframe
    df = pd.DataFrame(self)
    if geom_name is None:
        df[join_name] = df.index
    dataframe_d["local"] = df

    # need to check if it's already 4326
    if self.crs != 4326:
        self = self.to_crs(epsg=4326)

    bbox = self.total_bounds
    if center is None:
        center = [(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2]

    self.to_json()

    dframe_explorer.start(
        dataframe_d,
        center=center,
        zoom=zoom,
        shape_json=self.to_json(),
        geom_name=geom_name,  # from JSON file
        join_name=join_name,  # from data frames
        precision=precision,
        port=port,
        host=host,
        testing=testing
    )


def patch_geopandas():
    """
    Add a new function to the geodataframe called explore which uses the
    urbansim function dataframe_explorer.
    """
    import geopandas
    geopandas.GeoDataFrame.explore = explore
