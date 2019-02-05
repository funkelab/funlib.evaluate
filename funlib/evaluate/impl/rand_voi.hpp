#ifndef IMPL_RAND_VOI_H__
#define IMPL_RAND_VOI_H__

#include <map>
#include <math.h>

struct Metrics {

	double voi_split;
	double voi_merge;
	double rand_split;
	double rand_merge;
};

template <typename V1, typename V2>
Metrics
rand_voi_arrays(
		std::size_t size,
		const V1* gt,
		const V2* ws){

	double total = 0;

	// number of co-occurences of label i and j
	std::map<uint64_t, std::map<uint64_t, double>> p_ij;

	// number of occurences of label i and j in the respective volumes
	std::map<uint64_t, double> s_i, t_j;

	for (std::ptrdiff_t i = 0; i < size; ++i) {

		uint64_t wsv = ws[i];
		uint64_t gtv = gt[i];

		if (gtv) {

			++total;

			++p_ij[gtv][wsv];
			++s_i[wsv];
			++t_j[gtv];
		}
	}

	// sum of squares in p_ij
	double sum_p_ij = 0;
	for ( auto& a: p_ij )
		for ( auto& b: a.second )
			sum_p_ij += b.second * b.second;

	// sum of squares in t_j
	double sum_t_k = 0;
	for ( auto& a: t_j )
		sum_t_k += a.second * a.second;

	// sum of squares in s_i
	double sum_s_k = 0;
	for ( auto& a: s_i )
		sum_s_k += a.second * a.second;

	// we have everything we need for RAND, normalize histograms for VOI

	for ( auto& a: p_ij )
		for ( auto& b: a.second )
			b.second /= total;

	for ( auto& a: t_j )
		a.second /= total;

	for ( auto& a: s_i )
		a.second /= total;

	// compute entropies

	// H(s,t)
	double H_st = 0;
	for ( auto& a: p_ij )
		for ( auto& b: a.second )
			if(b.second)
				H_st -= b.second * log2(b.second);

	// H(t)
	double H_t = 0;
	for ( auto& a: t_j )
		if(a.second)
			H_t -= a.second * log2(a.second);

	// H(s)
	double H_s = 0;
	for ( auto& a: s_i )
		if(a.second)
			H_s -= a.second * log2(a.second);

	double rand_split = sum_p_ij/sum_t_k;
	double rand_merge = sum_p_ij/sum_s_k;

	// H(s|t)
	double voi_split = H_st - H_t;
	// H(t|s)
	double voi_merge = H_st - H_s;

	Metrics metrics;
	metrics.rand_split = rand_split;
	metrics.rand_merge = rand_merge;
	metrics.voi_split  = voi_split;
	metrics.voi_merge  = voi_merge;

	return metrics;
}

#endif // IMPL_RAND_VOI_H__

