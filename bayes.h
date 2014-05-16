#ifndef BAYES_H_
#define BAYES_H_

#include <armadillo>

using namespace arma;

template<typename T>
struct Distribution {
	Col<T> *candidates;
	vec *probs;
};

/* Bayes Update */
template<typename T>
void update(Distribution<T> *h, double (*likelihood)(T *hypo, T *data), T *data);

/* Expectated value */
template<typename T>
double E(Distribution<T> *h, double (*val)(T *hypo));

/* Identity function for expectation computation */
inline double identity(double *hypo);

/* Probability mass function -> cumulated mass function */
inline vec pmf2cdf(const vec *pmf);

inline vec cdf2pmf(const vec *cdf);

/* Monte Carlo */
static inline double norm_rand() {
	return (double)rand() / RAND_MAX;
}

inline int random_draw(const double *pmf, int l) {
	double r = norm_rand();
	double mass = 0.0;
	for(int i=0;i<l;i++) {
		mass += pmf[i];
		if(mass + 0.00000001 > r) {
			return i;
		}
	}
	return l-1;
}

/* Draw n times from the distribution and take the maximum result. */
vec n_draws(const vec *pmf, int n);

/* im[o,i] the probability of improving by i if the optimum is o away. */
mat improvement_given_optimum(vec *values, double(*prob)(double improvement, double optimum), int draws);

/* New belief given an observed improvement. */
vec belief_update(const vec* initial_belief, const mat *im, int improvement);

int best_action(const vec *O, const mat *ims, uint periods, double *best_action_value);

double V_static(const vec *O, const mat *im, uint periods);
double V_static_MC(Distribution<double> *hypos, mat *im, int servers, uint periods);
vec V_repeated_MC(const vec *belief, const mat *im, uint periods);

#endif
