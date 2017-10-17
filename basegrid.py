import pandas as pd
import math
import numpy as np


class Basegrid(object):
    basegrid = None

    def __int__(self):
        pass

    def load_csv(self, path, x='X', y='Y', sep='\t'):
        """Load csv"""
        self.basegrid = pd.read_csv(path, sep=sep)
        try:
            self.basegrid['E'] = self.basegrid[x]
            self.basegrid['N'] = self.basegrid[y]
        except ValueError:
            print "Oops!  The provided (x,y) column names do not exist.  Try again..."

    def load_shp(self, path):
        """Load csv"""
        try:
            import geopandas as gpd
        except RuntimeError:
            print('Need geopandas for that. Try: pip install geopandas')
            raise

        self.basegrid = gpd.read_file(path)
        try:
            self.basegrid.loc[:,('E')] = map(lambda p: self.basegrid.geometry.get_values()[p].x, range(len(self.basegrid)))
            self.basegrid.loc[:,('N')] = map(lambda p:self.basegrid.geometry.get_values()[p].y, range(len(self.basegrid)))
        except ValueError:
            print "Oops!  The provided file does not have a proper (x,y) geometry column.  Try again..."

