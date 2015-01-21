from __future__ import absolute_import
from __future__ import print_function

import unittest
import numpy as np
import time

from mozsci import histogram


class TestHistogram1D(unittest.TestCase):
    def test_histogram1d(self):

        h = histogram.Histogram1DFast(10, 0, 10)
        self.assertTrue((np.abs(h.bin_edges - np.arange(11)) < 1.0e-12).all())

        x = np.array([-1.0, 0.5, 3.2, 0.77, 9.99, 10.1, 8.2])
        h.update(x)

        xc = np.array([1.5, 2.5, 8.3])
        cc = np.array([10, 5, 22])
        h.update_counts(xc, cc)
        self.assertTrue((h.bin_count == np.array([3, 10, 5, 1, 0, 0, 0, 0, 23, 2])).all())

        # check compute_indices
        self.assertTrue((h.compute_indices(np.arange(12) - 0.5) == np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9])).all())


        # benchmark
        x = np.random.randn(1e7)
        time1 = time.time()
        h = histogram.Histogram1DFast(100, -5, 5)
        h.update(x)
        time2 = time.time()
        out = np.histogram(x, bins=100, range=[-5, 5])
        time3 = time.time()

        print("Time for fast = " + str(time2 - time1) + " s")
        print("Time for numpy = " + str(time3- time2) + " s")


        # check sampler
        t1 = time.time()
        samples = h.sample(5e6)
        t2 = time.time()

        (counts, edges) = np.histogram(samples, 50, normed=True)
        centers = 0.5 * (edges[1:] + edges[0:-1])
        actual_pdf = 1.0 / np.sqrt(2.0 * 3.14159) * np.exp(-centers ** 2 / 2.0)
        self.assertTrue(np.allclose(counts, actual_pdf, atol=5e-3))

    def test_stratified_sample(self):
        hist = histogram.Histogram1DFast(5, 0, 5)
        hist.update_counts(np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
                           np.array([5e6, 1e6, 1e4, 1e3, 2]))

        hist.compute_pdf_cdf()

        # generate a sample
        x = hist.sample(int(hist.bin_count.sum()))

        # now do a stratified sample of the large sample
        sample_size = [5000, 3000, 1000, 250, 2]
        x_stratified_sample = hist.stratified_sample(x, sample_size)
        hist_check = histogram.Histogram1DFast(5, 0, 5)
        hist_check.update(x_stratified_sample)

        # check that the actual sample distribution matches the expected
        # one.  We expect a small relative difference in all entries
        # except the last (where we expect a small absolute difference)
        self.assertTrue(np.allclose(1.0,
            hist_check.bin_count[:-1].astype(np.float) / sample_size[:-1],
            atol=0.10, rtol=0.0))
        self.assertTrue(abs(hist_check.bin_count[-1] - sample_size[-1]) < 3)


if __name__ == "__main__":
    unittest.main()


