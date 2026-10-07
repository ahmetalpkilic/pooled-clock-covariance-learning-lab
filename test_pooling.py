import unittest
import numpy as np
import lab

class PoolingTests(unittest.TestCase):
    def test_independent_covariance_and_event_means(self):
        rng=np.random.default_rng(85);z=rng.normal(size=(40,4,2))
        expected=np.stack([np.cov(x,rowvar=False) for x in z])
        np.testing.assert_allclose(lab.estimate(z),expected,atol=1e-14)
        pooled=lab.estimate(z,lab.CLASSES)
        for i,g in enumerate(lab.CLASSES):np.testing.assert_allclose(pooled[i],expected[lab.CLASSES==g].mean(axis=0))
        # Removing an event-specific mean protects against pooling time span.
        np.testing.assert_allclose(lab.estimate(z+np.arange(40)[:,None,None]*10,lab.CLASSES),pooled,atol=1e-13)
        perm=rng.permutation(40)
        np.testing.assert_allclose(lab.estimate(z[perm],lab.CLASSES[perm]),pooled[perm])

    def test_exact_mse_and_mismatch_floor(self):
        self.assertAlmostEqual(lab.predicted_relative_mse(lab.covariances(1)),2/60)
        rng=np.random.default_rng(92)
        # Independent chi-square construction for variance estimates.
        for q in [1,4]:
            v=lab.covariances(q)[:,0,0]
            samples=v*rng.chisquare(3,size=(30000,40))/3
            pooled=np.stack([samples[:,lab.CLASSES==g].mean(axis=1) for g in lab.CLASSES],axis=1)
            mse=np.mean((pooled/v-1)**2)
            self.assertAlmostEqual(mse/lab.predicted_relative_mse(lab.covariances(q)),1,delta=.025)
        self.assertGreater(lab.predicted_relative_mse(lab.covariances(4),m=100000),1)

    def test_independent_seed_tradeoff(self):
        report=lab.experiment(trials=180,seed=333)
        first,last=report['experiment'][0],report['experiment'][-1]
        self.assertLess(first['class_pool']['rmse_ms'],.85*first['per_event']['rmse_ms'])
        self.assertGreater(last['class_pool']['rmse_ms'],1.2*last['per_event']['rmse_ms'])
        for row in report['experiment']:
            for key in ['oracle','per_event','class_pool']:self.assertEqual(row[key]['failed_trials'],0)

    def test_bad_inputs(self):
        for z in [np.ones((40,2,2)),np.ones((40,4,2)),np.zeros((3,4)),np.full((40,4,2),np.nan)]:
            with self.assertRaises(ValueError):lab.estimate(z)
        with self.assertRaises(ValueError):lab.estimate(np.random.default_rng(0).normal(size=(40,4,2)),[0,1])
        with self.assertRaises(ValueError):lab.covariances(0)

if __name__=='__main__':unittest.main()
