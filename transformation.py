import pandas as pd
import math
import numpy as np
from scipy.spatial.distance import cdist, pdist


class Transformation(object):

    def __int__(self, basepath, basegrid):

        self.basepath = basepath
        self.basegrid = basegrid

    def calculate_bp(self):
        """ 
        :param self: bp (distance bp)
        :return: calculate distance matrix from basepath to basegrid
        """
         self.bp = cdist(self.basepath[['E','N']].as_matrix(), self.basegrid[['E','N']].as_matrix())

    def calculate_vbp(self):
        """ 
        :param self: vbp (vector bp)
        :return: None
        """

        try:
            a = np.stack([self.basepath[['E', 'N']].as_matrix()]*len(self.basegrid))
            b = np.stack([self.basegrid[['E', 'N']].as_matrix()]*len(self.basepath)).reshape(a.shape)

            # a little magic
            c = pd.DataFrame(np.vstack(a-b))
            c[0] = c[0]/ np.hstack(bp)
            c[1] = c[1]/ np.hstack(bp)

            self.vbp = c.as_matrix().reshape(np.stack(a-b).shape)

        except RuntimeError:
            print('Runtime error is raised')
            raise

    def calculate_pbc(self):

        vbc = self.basepath.vbc.as_matrix()
        cd