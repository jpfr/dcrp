#include <iostream>
#include <math.h> 
#include <armadillo>
#include "bayes.h"
#include "utc.h"
#include "parameters.h"

using namespace arma;
using namespace std;

static inline double unnormalised_exp_dist(double x, double lambda) {
	return exp(-lambda * x);
}

static inline double unnormalised_transformed_exp_dist(double x, double opt) {
	double lambda = 10.0;
	return unnormalised_exp_dist(x/opt, lambda);
}

static inline double unnormalised_normal_dist(double x, double mu, double sigma) {
	return exp(-pow(x-mu,2)/(2*pow(sigma,2)));
}

// how often does a certain result occur?
mat frequency(vec *results) {
	int min = results->min();
	int max = results->max();
	mat f = mat(max-min+1,2, fill::zeros);
	for(int i=min;i<=max;i++) {
		f.at(i-min,0) = i;
		for(uint k = 0;k<results->n_elem;k++) {
			if(results->at(k) == i)
				f.at(i-min,1) += 1.0;
		}
	}
	return f;
}

int main(int argc, char** argv) {

	int opt_steps = observation_count; // from parameters.h
#define max_servers action_count - 1   // from parameters.h 
	int periods = 4;

	vec values(opt_steps);
	for (int i=0; i<opt_steps; i++) {
		values[i] = (double)i; // improvement of 0 is worth 0.
	}
	vec belief = zeros<vec>(opt_steps);

	/* Normally distributed*/
	int o_pos = 120;
	for (int i=0; i<opt_steps; i++) {
	 	belief[i] = unnormalised_normal_dist((double)i, (double)o_pos, 20);
	}
	belief = normalise(belief, 1);

	/* Extreme on both ends*/
	// belief.at(0) = 0.5;
	// belief.at(opt_steps-1) = 0.5;

	/*mat output(opt_steps, 2);
	output.col(0) = values;
	output.col(1) = belief;
	output.save("initial_belief.dat", raw_ascii);*/

	//mat i_prob(opt_steps,4);
	//for (int i=0; i<opt_steps; i++) {
	//	i_prob.at(i,1) = unnormalised_transformed_exp_dist((double)i, 50);
	//	i_prob.at(i,2) = unnormalised_transformed_exp_dist((double)i, 100);
	//	i_prob.at(i,3) = unnormalised_transformed_exp_dist((double)i, 150);
	//}
	//i_prob.col(0) = values;
	//i_prob.col(1) = normalise(i_prob.col(1),1);
	//i_prob.col(2) = normalise(i_prob.col(2),1);
	//i_prob.col(3) = normalise(i_prob.col(3),1);
	//i_prob.save("improvement_prob.dat", raw_ascii);

	//mat i_prob(opt_steps,4);
	//for (int i=0; i<opt_steps; i++) {
	//	i_prob.at(i,1) = unnormalised_transformed_exp_dist((double)i, 50);
	//	i_prob.at(i,2) = unnormalised_transformed_exp_dist((double)i, 100);
	//	i_prob.at(i,3) = unnormalised_transformed_exp_dist((double)i, 150);
	//}
	//i_prob.col(0) = values;
	//i_prob.col(1) = n_draws(&vec(normalise(i_prob.col(1),1)),3);
	//i_prob.col(2) = n_draws(&vec(normalise(i_prob.col(2),1)),3);
	//i_prob.col(3) = n_draws(&vec(normalise(i_prob.col(3),1)),3);
	//i_prob.save("improvement_prob_3_servers.dat", raw_ascii);

	// mat im = improvement_given_optimum(&values, unnormalised_transformed_exp_dist, servers);
	// mat O_n(opt_steps,7);
	// O_n.col(0) = values;
	//Distribution<double> hypos = {&values, &belief};
	//O_n.col(1) = V(&hypos, &im, server_costs, servers, 11);
	//O_n.col(2) = V(&hypos, &im, server_costs, servers, 9);
	//O_n.col(3) = V(&hypos, &im, server_costs, servers, 7);
	//O_n.col(4) = V(&hypos, &im, server_costs, servers, 5);
	//O_n.col(5) = V(&hypos, &im, server_costs, servers, 3);
	//O_n.col(6) = V(&hypos, &im, server_costs, servers, 1);
	//O_n.save("O_n_3_servers.dat", raw_ascii);

	//cout << "Value is: " << V(&hypos, &im, server_costs, servers, periods) << "\n";
	//cout << "MC Value is: " << V_MC(&hypos, &im, server_costs, servers, periods) << "\n";

	mat ims[max_servers+1];
	for(int i=0;i<max_servers+1;i++) {
		ims[i] = improvement_given_optimum(&values, unnormalised_transformed_exp_dist, i);
	}

	/* UTC */
	utc_result result = Search(periods, &belief, ims);
	cout << "UTC best action: " << result.best_action << endl;
	cout << "UTC best action value: " << result.best_value << endl;
	onode_show_N(result.root);
	result.convergence.save("utc_convegence.dat", raw_ascii);
	vec utc_res = MC_utc(result.root, &belief, ims, periods);
	frequency(&utc_res).save("utc_results.dat", raw_ascii);
	cout << "UTC MC value: " << mean(utc_res) << endl;

	/* Bayes */
	double best_bayes_value = 0.0;
	int best_bayes_action = best_action(&belief, ims, periods, &best_bayes_value);
	cout << "Static Bayes best action: " << best_bayes_action << endl;
	cout << "Static Bayes best action value: " << best_bayes_value << endl;
	vec repeated_results = V_repeated_MC(&belief, ims, periods);
	cout << "Repeated Bayes MC value: " << mean(repeated_results) << endl;
	frequency(&repeated_results).save("repeated_results.dat", raw_ascii);
		
	cout << "Press Enter to Continue";
	cin.ignore();
	return 0;
}

