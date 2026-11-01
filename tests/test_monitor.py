import unittest
import numpy as np
from UQ_test.pdn_monitor import build_template, estimate_monotone_phase
from UQ_test.prodm_detector import ProDMDetector
from UQ_test.conformal import cumulative_band


class MainMonitorTests(unittest.TestCase):
    def test_monotonic_phase_uses_no_future_observations(self):
        template={'mu_d':np.zeros(1),'sd_d':np.ones(1),
                  'mu_t':np.arange(4)[:,None],'T':4}
        trajectory=np.array([[0],[2],[1],[3]])
        expected=np.array([0,2,2,3])
        for length in range(1,5):
            np.testing.assert_array_equal(estimate_monotone_phase(template,trajectory[:length]),expected[:length])

    def test_direct_template_uses_fit_normalization(self):
        x=np.arange(24,dtype=np.float32).reshape(3,4,2)
        template=build_template(x)
        flat=x.reshape(-1,2)
        np.testing.assert_array_equal(template['mu_t'],((x-flat.mean(0))/(flat.std(0)+1e-6)).mean(0))

    def test_monitor_does_not_fit_on_test_or_failed_data(self):
        rng=np.random.RandomState(7)
        obs=rng.normal(size=(2000,4,2)).astype(np.float32)
        density=np.abs(rng.normal(size=(2000,4)))
        success=np.ones(2000,dtype=bool)
        success[:1000:7]=False
        first=ProDMDetector(.025).fit(obs,density,success)
        obs[1000:]=np.nan
        density[1000:]=np.nan
        obs[:1000][~success[:1000]]=np.nan
        density[:1000][~success[:1000]]=np.nan
        second=ProDMDetector(.025).fit(obs,density,success)
        np.testing.assert_array_equal(first.band,second.band)
        self.assertFalse(np.intersect1d(first.fit_ids,first.cal_ids).size)
        np.testing.assert_array_equal(first.model['tmpl']['mu_t'],second.model['tmpl']['mu_t'])

    def test_small_calibration_set_fails_explicitly(self):
        with self.assertRaises(ValueError):
            cumulative_band([np.ones(4)],[np.ones(4)],.01)


if __name__=='__main__':
    unittest.main()
