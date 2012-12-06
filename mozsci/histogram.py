"""
 Fast 1D, 2D and 3D empirical histograms samplers.

 Efficently compute binned histograms from large streaming
 data sets, using scipy.weave to speed up the slow steps.
 The speed is typically 10-100X faster then the corresponding numpy
 routine.

 Provides capabilities to estimate a probability density
 function from data, sample from a given distribution,
 plot, serialize to/from a file.
"""

import numpy as np
import scipy.weave

class Histogram1DFast(object):
    """A fast 1D histogram sampler
    for evenly spaced bins"""
    def __init__(self, bins, mn, mx):
        """bins evenly spaced bins from mn to mx"""
        self.bins = int(bins)
        self.bin_width = (mx - mn) / float(bins)
        self.bin_count = np.zeros((bins, ), np.int)
        self.bin_edges = mn + self.bin_width * np.arange(self.bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[0:-1] + self.bin_edges[1:])
        self.mx = float(mx)
        self.mn = float(mn)
        self._pdf_updated = False
        self.pdf = np.zeros((bins, ), np.float)
        self.cdf = np.zeros((bins, ), np.float)

    def update(self, data):
        """data is a 1D array to update histogram with
        Note: pdf, cdf are not updated after updating the counts
            if updated values are needed, client should call self.compute_pdf_cdf()
            before accessing.  TODO: .pdf and .cdf attributes that lazily
            compute/return based on the value of self._pdf_updated"""
        ndata = len(data)
        bin_count = self.bin_count
        bin_width = self.bin_width
        mn = self.mn
        bins1 = self.bins - 1
        code = """
            for (int i = 0; i < ndata; i++)
            {
                int bin_index = floor((data(i) - mn) / bin_width);
                bin_index = std::min(std::max(bin_index, 0), bins1);
                bin_count(bin_index)++;
            }
            """
        scipy.weave.inline(code,
                ['ndata', 'bin_count', 'bin_width', 'data', 'bins1', 'mn'],
                type_converters=scipy.weave.converters.blitz)
        self._pdf_updated = False

    def plot(self, ti, fignum):
        """Plots the current histogram count
        ti = the title
        fignum = make this figure number
        plots both counts and log(counts)
        returns fig"""
        import pylab as plt

        fig = plt.figure(fignum)
        fig.clf()
    
        plt.subplot(211)
        plt.plot(self.bin_centers, self.bin_count)
        plt.ylabel("# " + ti)
    
        plt.subplot(212)
        plt.plot(self.bin_centers, np.log(self.bin_count + 1))
        plt.ylabel("log(# " + ti + ')')
    
        return fig


    def update_counts(self, data, counts):
        """data is a 1D array of x values, counts is a 1D array
           of counts to add"""
        ndata = len(data)
        assert len(counts) == ndata
        bin_count = self.bin_count
        bin_width = float(self.bin_width)
        mn = float(self.mn)
        bins1 = self.bins - 1
        code = """
            for (int i = 0; i < ndata; i++)
            {
                int bin_index = floor((data(i) - mn) / bin_width);
                bin_index = std::min(std::max(bin_index, 0), bins1);
                bin_count(bin_index) += counts(i);
            }
            """
        scipy.weave.inline(code,
                ['ndata', 'bin_count', 'bin_width', 'data', 'bins1', 'mn', 'counts'],
                type_converters=scipy.weave.converters.blitz)
        self._pdf_updated = False

    def compute_indices(self, data):
        """Compute the indices in the histogram corresponding to data,
        but do not update"""
        ndata = len(data)
        mn = self.mn
        bins1 = self.bins - 1
        bin_index = np.zeros(data.shape, np.int)
        bin_width = self.bin_width
        code = """
            for (int i = 0; i < ndata; i++)
            {
                int this_index = floor((data(i) - mn) / bin_width);
                bin_index(i) = std::min(std::max(this_index, 0), bins1);
            }
            """
        scipy.weave.inline(code,
                ['ndata', 'bin_width', 'data', 'bins1', 'mn', 'bin_index'],
                type_converters=scipy.weave.converters.blitz)
        return bin_index


    def compute_pdf_cdf(self):
        """Compute and store the pdf and cdf of bin_count"""
        if not self._pdf_updated:
            ndata = self.bin_count.sum()
            if ndata > 0:
                self.pdf = self.bin_count / float(self.bin_count.sum())
                self.cdf = self.pdf.cumsum()
            else:
                self.pdf = None
                self.cdf = None
            self._pdf_updated = True

    def sample(self, N, return_edge=False, return_index=False):
        """Returns N samples of x
        if return_edge = True then returns the left bin_edge
            instead of a random sample from the interval
        if return_index = True then return the index of
            the selected bin
        Can't have both return_index and return_edge"""
        assert not (return_edge and return_index)
        if not self._pdf_updated:
            self.compute_pdf_cdf()

        # sample the cdf
        # numpy's searchsorted uses binary search
        # and returns the left bin edge index
        rand1 = np.random.rand(N)
        samples = self.cdf.searchsorted(rand1)
        if return_index:
            ret = samples
        elif return_edge:
            ret = self.bin_edges[samples]
        else:
            rand2 = np.random.rand(N)
            ret = self.bin_edges[samples] + rand2 * self.bin_width
        return ret

    def stratified_sample(self, x, sample_size=None, indices=False, empty_bin_rate=0.01):
        """Input:
                X = (N, ) numpy vector of samples from this distribution,
                sample_size = (self.bins, ) vector.  This gives the
                    total number of samples to take from this distribution
                    for each of the histogram bins.
                    If None, then uses the last cached value
                empty_bin_rate = if the bin_count == 0 for any bins, then the sampling
                    rate in them is set to empty_bin_rate.
                    Note: this is only used if sample_size is also provided
           Output:
            if indices == False, return a sample from X stratified according
                to sample_size
            if indices == True, return the indices into X to make that sample"""

        if sample_size is not None:
            # update sampling rate
            gt0_count = self.bin_count > 0
            sz = np.asarray(sample_size)
            self._stratified_sampling_rate = np.zeros(sz.shape)
            self._stratified_sampling_rate[gt0_count] = sz[gt0_count] / self.bin_count[gt0_count].astype(np.float)
            self._stratified_sampling_rate[~gt0_count] = empty_bin_rate

        # strategy: find the sampling rate for each point in the input
        # vector x.  choose it with that sampling rate
        xindices = self.compute_indices(x)
        nsamples = len(xindices)
        r = np.random.rand(nsamples)
        indices_accept = np.arange(nsamples)[r < self._stratified_sampling_rate[xindices]]
        if indices:
            return indices_accept
        else:
            return x[indices_accept]


