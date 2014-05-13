#include "bayes.h"
#include <iostream>
#include <armadillo>
#include "parameters.h"

using namespace std;
using namespace arma;

/* Bayes Update */
template<typename T>
void update(Hypotheses<T> *h, double (*likelihood)(T *hypo, T *data), T *data) {
	Col<T> *c = h->candidates;
	vec *p = h->probs;
	int l = h->probs.n_elem;
	for(int i=0; i<l; i++)
		p[i] = p[i] * likelihood(&(c[i]), data);
	h->probs = normalise(*p, 1);
	free(p);
}

/* Expectated value */
template<typename T>
double E(Hypotheses<T> *h, double (*val)(T *hypo)) {
	double e = 0.0;
	vec *p = h->probs;
	Col<T> *c = h->candidates;
	int l = p->n_elem;
	for(int i=0; i<l; i++)
		e += val(c[i]) * p[i];
	return e;
}

/* Identity function for expectation computation */
inline double identity(double *hypo) {
	return *hypo;
}

/* Probability mass function -> cumulated mass function */
inline vec pmf2cdf(const vec *pmf) {
	int l = pmf->n_elem;
	vec cdf(l);
	cdf[0] = pmf->at(0);
	for(int i=1; i<l; i++)
		cdf[i] = cdf[i-1] + (*pmf)[i];
	return cdf;
}

inline vec cdf2pmf(const vec *cdf) {
	int l = cdf->n_elem;
	vec pmf(l);
	pmf[0] = cdf->at(0);
	for(int i=1; i<l; i++)
		pmf[i] = cdf->at(i) - cdf->at(i-1);
	return pmf;
}

/* Draw n times from the distribution and take the maximum result. */
vec n_draws(const vec *pmf, int n) {
	vec p = pow(pmf2cdf(pmf),n);
	return cdf2pmf(&p);
}

/* im[i,o] the probability of improving by i if the optimum is o away. */
mat improvement_given_optimum(vec *values, double(*prob)(double improvement, double optimum), int draws) {
	int l = values->n_elem;
	mat im(l,l, fill::zeros);
	
	if (draws == 0) {
		im.row(0).fill(1.0);
		return im;
	}

	im.at(0,0) = 1.0; // for o=0, no improvements can be made.
	for (int o=1; o<l; o++) {
		vec pi_o = zeros<vec>(l);
		for (int i=0; i<=o; i++)
			pi_o[i] = prob(values->at(i), values->at(o));
		pi_o = normalise(pi_o,1);
		im.col(o) = n_draws(&pi_o, draws);
	}
	return im;
}

/* Belief update where improvements are already taken into account. */
vec belief_update(const vec* initial_belief, const mat *im, int improvement) {
	int l = im->n_cols; // #possible improvements
	vec nb = zeros<vec>(l);
	for (int o=0; o<l-improvement;o++) // o=distance to optimum
		nb.at(o) = im->at(improvement, o+improvement) * initial_belief->at(o+improvement); // P(H|D) = P(D|H) * P(H)
	return normalise(nb, 1);
}

int best_action(const vec *O, const mat *ims, uint periods, double *best_action_value) {
	int best_action = -1;
	*best_action_value = -1000000.0;

	for(int a=0;a<action_count;a++) {
		double action_value = V_static(O, &ims[a], periods) - (a * server_cost * periods);
		if(action_value > *best_action_value) {
			best_action = a;
			*best_action_value = action_value;
		}
	}
	return best_action;
}

double V_static(const vec *O, const mat *im, uint period) {
	colvec P_i = *im * *O; // distribution of improvements
	double value = 0.0;
	for(uint k=0;k<O->n_elem;k++)
		value += k * P_i.at(k); // dot(P_i, *hypos->candidates); // expected immediate improvement

	if (period == 1) return value;

	int l = O->n_elem;
	mat pd(l, l, fill::zeros); // pd[o,i] given observed improvement i, the belief on the remaining distance to the optimum o
	for (int i=0; i<l;i++) {
		for (int o=0; o<l-i;o++)
			pd.at(o,i) = im->at(i,o+i) * O->at(o+i); // P(H|D) = P(D|H) * P(H)
		pd.col(i) = normalise(pd.col(i), 1);
	}
	vec Oprime = pd * P_i;
	return value + V_static(&Oprime, im, period-1);
}

double V_static_MC(Hypotheses<double> *hypos, mat *im, int servers, uint period) {
	srand (time(NULL));
	int N = 10000000;
	//vec final_os = zeros<vec>(hypos->probs->n_elem);
	double total_value = 0.0;

	for(int n=0;n<N;n++) {
		int o_pos = random_draw(hypos->probs->memptr(), hypos->probs->n_elem);
		double value = 0.0;
		for(;period>0;period--) {
			value -= server_cost * servers;
			int i_pos = random_draw(im->colptr(o_pos), im->n_rows);
			value += hypos->candidates->at(i_pos);
			o_pos -= i_pos;
		}
		total_value += value;
		//	final_os.at(o_pos) += 1.0;
	}
	//final_os.save("os.csv", raw_ascii);
	return total_value/N;
}

/* Recomputes a new server amount after every observation (improvement) */
double V_dynamic_MC(const vec *orig_belief, const mat *ims, uint period) {
	srand (time(NULL));
	int N = 1000;
	double total_value = 0.0;
	double dummy_value;
	const vec *belief = orig_belief;
	vec new_belief;
	
	for(int n=0;n<N;n++) {
		int o_pos = random_draw(orig_belief->memptr(), orig_belief->n_elem);
		belief = orig_belief;
		for(int p=period;p>0;p--) {
			int action = best_action(belief, ims, p, &dummy_value);
			total_value -= server_cost * action;
			int improvement = random_draw(ims[action].colptr(o_pos), ims[action].n_rows);
			total_value += improvement;
			o_pos -= improvement;
			if(p > 1) {
				new_belief = belief_update(belief, &ims[action], improvement);
				belief = &new_belief;
			}
		}
	}
	return total_value/(double)N;
}
