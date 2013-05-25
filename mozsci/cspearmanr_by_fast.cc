#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <algorithm>

template <class iter_t>
struct indexed_compare
{
    iter_t begin;
    indexed_compare(iter_t begin) : begin(begin) {}
    bool operator()(std::size_t a, std::size_t b) const {
        // sort in ascending order
        return *(begin+a) < *(begin+b);
    }
};

typedef indexed_compare<std::vector<double>::iterator> index_compare_double_vector;

std::vector<double> to_ranked (std::vector<double> a)
{
    int n = a.size();
    std::vector<size_t> ret(n);
    for (std::size_t i = 0; i < n; i++) ret[i] = i;
    std::sort<std::vector<size_t>::iterator, index_compare_double_vector>(ret.begin(), ret.end(), index_compare_double_vector(a.begin()));

    // need to take ties into account and assign the average rank
    std::vector<double> ret2(a.size());
    int sumranks = 0.0;
    int dupcount = 0.0;
    double eps = 1.0e-8;
    for (std::size_t i = 0; i < n; i++)
    {
        sumranks = sumranks + i;
        dupcount++;
        if (i == (n - 1) || abs(a[ret[i]] != a[ret[i+1]]) > eps)
        {
            double avgrank = double(sumranks) / double(dupcount) + 1;
            for (int j = i - dupcount + 1; j < i + 1; j++)
            {
                ret2[ret[j]] = avgrank;
            }
            sumranks = 0;
            dupcount = 0;
        }
    }

    return ret2;
}

template <class T>
double pearson_correlation(std::vector<T> a, std::vector<T> b) 
{
//    for (int i = 0; i < a.size(); i ++) {
//      std::cout << a[i] << "  " << b[i] << "  " << std::endl;
//    }
    if (a.size() != b.size()) abort();

    double sum_a_b = inner_product(a.begin(), a.end(), b.begin(), 0.0);
    double sum_a = accumulate(a.begin(), a.end(), 0.0);
    double sum_b = accumulate(b.begin(), b.end(), 0.0);
    double sum_a_a = inner_product(a.begin(), a.end(), a.begin(), 0.0);
    double sum_b_b = inner_product(b.begin(), b.end(), b.begin(), 0.0);
    double n = a.size();

    double r = (sum_a_b - sum_a*sum_b/n) / sqrt((sum_a_a- sum_a*sum_a/n)*(sum_b_b-sum_b*sum_b/n));
    return r;
}

double spearman_correlation(std::vector<double> a, std::vector<double> b)
{
//    for (int i = 0; i < a.size(); i ++) {
//        std::cout << a[i] << "  " << b[i] << "  " << std::endl;
//    }
    return pearson_correlation(to_ranked(a), to_ranked(b));
}

double spearman_by(std::vector<double> a, std::vector<double> b, std::vector<std::size_t> byvar)
{
    // data must be sorted byvar in ascending order
    double ret = 0.0;
    int ngroups = 0;

    // the minimum number of elements in a by group to add into the overall result
    int min_by = 25;

    std::size_t last_by = byvar[0];
    int nby = 0;
    int start_index = 0;
    for (std::size_t k = 0; k < a.size(); k++)
    {
        if (byvar[k] == last_by)
        {
            nby += 1;
        }
        else
        {
            // we are at a new group
            if (nby >= min_by)
            {
                // compute stuff
                std::vector<double>  a_by_group(&a[start_index], &a[start_index + nby]);
                std::vector<double>  b_by_group(&b[start_index], &b[start_index + nby]);
                double sc = spearman_correlation(a_by_group, b_by_group);
                if (!isnan(sc))
                {
                    ret = ret + sc;
                    ngroups++;
                }
            }

            // reset
            nby = 1;
            start_index = k;
            last_by = byvar[k];
        }
    }

    // last group
    if (nby >= min_by)
    {
        // compute stuff
        std::vector<double>  a_by_group(&a[start_index], &a[start_index + nby]);
        std::vector<double>  b_by_group(&b[start_index], &b[start_index + nby]);
        double sc = spearman_correlation(a_by_group, b_by_group);
        if (!isnan(sc)) 
        {
            ret = ret + sc;
            ngroups++;
        }
    }

    return ret / ngroups;
}

extern "C" double c_spearman_for_python(double* a, double* b, std::size_t* byvar, std::size_t n)
{
    // wrapper function for python
    std::vector<double> avec (a, a + n);
    std::vector<double> bvec (b, b + n);
    std::vector<std::size_t> byvarvec (byvar, byvar + n);
    return spearman_by(avec, bvec, byvarvec);

}


int main(void)
{
    // initialize vectors
    //static const double arr[] = {1, 2, 3, 4, 5};
    //std::vector<double> position (arr, arr + sizeof(arr) / sizeof(arr[0]) );

    //static const double arr_pa[] = {0.4, 0.1, 0.22, -0.88, 0.55};
    //std::vector<double> pa (arr_pa, arr_pa + sizeof(arr_pa) / sizeof(arr_pa[0]) );

    static const double arr[] = {  0.33117374,   0.80947619,   3.        ,   0.25457016,
         0.52897721,   3.        ,   0.51733111,   0.60862871,
         0.21389315,   0.35368557,  10.        ,  10.        ,
         0.72061731,   0.23078359,   0.38791586,   0.43954613,
         0.91398124,   0.29594647,  10.        ,   0.78991894};
    std::vector<double> position (arr, arr + sizeof(arr) / sizeof(arr[0]) );

    static const double arr_pa[] = { 0.10526316,  1.15789474,  1.94736842,  2.21052632, -1.73684211,
       -1.47368421, -0.68421053,  1.68421053,  0.63157895,  0.36842105,
       -0.94736842,  1.42105263,  3.        , -0.42105263,  0.89473684,
        2.47368421, -1.21052632, -0.15789474,  2.73684211, -2.        };
    std::vector<double> pa (arr_pa, arr_pa + sizeof(arr_pa) / sizeof(arr_pa[0]) );


    std::cout << spearman_correlation(position, pa) << std::endl;

    static const double by_arr_pa[] = { 51.73402682,  52.19589972,  44.97281905,  54.73404694,
        47.6719409 ,  45.96619825,  50.36193419,  46.27607543,
        48.18824048,  54.88529706,  42.67667074,  41.80373588,
        37.29934119,  57.98812747,  45.04782628,  38.10858417,
        46.44031713,  40.59823939,  26.29936944,  23.96820474,
        47.98343799,  36.4455311 ,  43.92931621,  55.19172514,
        33.44633285,  37.38381116,  39.03392758,  41.43285553,
        28.63082987,  31.86069758,  41.19551474,  29.04928565,
        39.09690404,  36.75441683,  29.66390582,  70.4035713 ,
        63.53532854,  49.78916058,  64.39911984,  65.41353192,
        48.42353021,  60.38572122,  42.44357922,  42.86378695,
        58.93821467,  61.93862217,  36.23459784,  64.57533596,
        40.09399141,  45.57233379,  44.7748158 ,  50.88705955,
        47.24016865,  51.75866967,  36.17935042,  46.73933887,
        52.7136634 ,  47.0337377 ,  34.19077012,  18.5836512 ,
        41.63257011,   9.8698871 ,  37.63277795,  47.71676464,
        34.89667886,  35.10845963,  44.56638481,  36.70884056,
        57.9185177 ,  50.65260932,  58.53307806,  43.25154747,
        40.59802125,  38.97005406,  35.19682907,  51.94755877,
        44.04430199,  35.84048228,  36.25006727,  46.35317423,
        37.44668618,  16.90596421,  38.87970562,  47.33515849,
        27.41230181,  29.47142008 } ;
    std::vector<double> by_pa (by_arr_pa, by_arr_pa + sizeof(by_arr_pa) / sizeof(by_arr_pa[0]) );

    static const double by_arr_position[] = { 1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  12.,
        13.,  15.,  16.,  17.,  19.,  23.,  24.,  25.,  26.,  27.,  28.,
        29.,   1.,   2.,   3.,   6.,   8.,   9.,  11.,  12.,  13.,  17.,
        19.,  21.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,
        10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,
        22.,  23.,  24.,  25.,  26.,  27.,   1.,   2.,   4.,   5.,   6.,
         7.,   8.,   9.,  10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,
        18.,  20.,  21.,  22.,  23.,  24.,  25.,  26.,  27. };
    std::vector<double> by_position (by_arr_position, by_arr_position + sizeof(by_arr_position) / sizeof(by_arr_position[0]) );

    static const std::size_t by_arr_queryid[] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,
        3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
        3,  3,  3,  3,  3,  3,  3,  3};
    std::vector<std::size_t> by_queryid (by_arr_queryid, by_arr_queryid + sizeof(by_arr_queryid) / sizeof(by_arr_queryid[0]) );

    std::cout << spearman_by(by_pa, by_position, by_queryid) << std::endl;



}

