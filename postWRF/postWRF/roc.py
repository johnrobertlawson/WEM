"""Relative Operative Statistics.

Mixin of deterministic and probabilistic scores.
"""

class ROC(DetScores,ProbScores):
    def __init__(self,nens,pf,ob):
        """
        nens        :   number of ensemble members
        pf          :   probability of an event (1d array)
        ob (bool)   :   whether it happened (1d array)
        thresh      :   probability threshold(s)
                        These must be 1/nens, 2/nens... nens/nens.
                        Turned off for now.
                        Default is all thresholds.
        """
        self.nens = nens
        self.pf = pf
        self.ob = ob

        self.a,self.b,self.c,self.d = self.compute_roc()

    def compute_roc(self):
        thresholds = []
        X = N.zeros([self.nens])
        Y = N.zeros([self.nens])
        for nidx,n in enumerate(range(1,self.nens+1)):
            thresholds.append(n/self.nens)
            if self.pf[nidx] < thresholds[-1]:
                if self.ob[nidx]:
                    X[nidx] =+ 1
                else:
                    Y[nidx] =+ 1

        a = N.zeros([len(thresholds)])
        b = N.copy(a)
        c = N.copy(a)
        d = N.copy(a)
        for thidx,th in enumerate(thresholds):
            a[thidx] = N.sum(X[thidx+1:])
            b[thidx] = N.sum(Y[thidx+1:])
            c[thidx] = N.sum(X[:thidx+1])
            d[thidx] = N.sum(Y[:thidx+1])

    def compute_roca(self):
        """Area under ROC curve.

        TODO."""
        pass

    def compute_rocas(self):
        """ROC area skill score.
        """
        pass


