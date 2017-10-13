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

