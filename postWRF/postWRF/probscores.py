"""Probabilistic scores, using Buizza 2001, MWR.
"""

class ProbScores:
    def __init__(self,og=None,pfg=None):
        """
        self.og      :   True/False field (1D)
        self.pfg     :   Prob. forecast of event (0.0-1.0)
        """
        self.og = og
        self.pfg = pfg
        assert self.og.size == self.pfg.size

    def compute_briar(self,self.og,self.pfg):
        Ng = self.og.size
        self.og = self.og.flatten()
        self.pfg = self.pfg.flatten()
        BS = (1/Ng) * N.sum((self.pfg-self.og.astype(float))**2)
        return BS
    
    def compute_bss(self):
        """TODO
        """BS = self.compute_briar(self.og,self.pfg)
        BSS = 0
        return

    def compute_rps(self,thresholds):
        """
        Arguments:
            thresholds  : 1D list/array, and in same order
                            of first dimension of pfg.
            self.pfg    : 3D array: [probability index,lat,lon]
                    
        Note:
        RPSg is the grid-point RPS.
        RPS is the area-average (ranked probability score)
        """
        assert len(self.og.shape) == 3
        Ng = self.og[0,:,:].size
        # evs = N.array((sorted(thresholds) )
        Jev = len(thresholds)
        RPSgs = N.zeros_like(self.og)

        # RPS for every grid point, for each threshold
        for thidx,th in enumerate(thresholds):
            RPSgs[thidx,:,:] = (N.sum(pfg[:th+1,:,:],axis=0)-
                                N.sum(og[:th+1,:,:],axis=0))**2

        # Sum up over all thresholds for RPS at each point
        RPSg = N.sum(RPSgs,axis=0)

        # Average over all grid points for RPS.
        RPS1 = (1/Ng) * N.sum(RPSg)
        RPS2 = N.mean(RPSg)
        assert RPS1 == RPS2
        return RPS2


    def compute_poutl(self,E,obs):
        """Percentage of outliers.

        E       :   Ensemble class instance
        obs     :   observation array the same 2D size.
        """
        pass
