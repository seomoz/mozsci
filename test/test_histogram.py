
import unittest
import numpy as np
import time

from mozsci import histogram
import pylab as plt


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

        print "Time for fast = " + str(time2 - time1) + " s"
        print "Time for numpy = " + str(time3- time2) + " s"


        # check sampler
        t1 = time.time()
        samples = h.sample(3e6)
        t2 = time.time()
        print "Time to sample 1D for 3e6 = " + str(t2 - t1) + " s"

        # TODO: replace this "eye norm" with an actual norm
        (counts, edges) = plt.histogram(samples, 50, normed=True)
        centers = 0.5 * (edges[1:] + edges[0:-1])
        actual_pdf = 1.0 / np.sqrt(2.0 * 3.14159) * np.exp(-centers ** 2 / 2.0)
        fig = plt.figure(1); fig.clf()
        plt.plot(centers, counts, label="Sample")
        plt.plot(centers, actual_pdf, label="Actual")
        plt.legend()
        fig.show()

    def test_stratified_sample(self):
        hist = histogram.Histogram1DFast(5, 0, 5)
        hist.update_counts(np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
                           np.array([5e6, 1e6, 1e4, 1e3, 2]))

        hist.compute_pdf_cdf()

        # generate a 1e6 size sample
        x = hist.sample(int(hist.bin_count.sum()))

        # now sample the large sample in 2 ways
        #  uniformly
        #  stratified
        sample_size = [500, 300, 100, 98, 2]
        x_stratified_sample = hist.stratified_sample(x, sample_size)
        hist_check = histogram.Histogram1DFast(5, 0, 5)
        hist_check.update(x_stratified_sample)

        # this "eye norm" too needs to be replaced
        fig = plt.figure(101)
        fig.clf()
        plt.plot(sample_size, 'bo', label='ideal')
        plt.plot(hist_check.bin_count, 'rx', label='actual sample')
        plt.legend()
        plt.title("1D stratified sampling")
        fig.show()

if __name__ == "__main__":
    unittest.main()




