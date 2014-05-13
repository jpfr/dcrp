#ifndef UTC_H_
#define UTC_H_

#include <map>
#include "armadillo"
#include "bayes.h"

using namespace std;
using namespace arma;

struct anode;

struct onode {
	int observation_index;
	int N;
	map<int, anode> actions;
	anode *father;
};

struct anode {
	int action_index;
	int N;
	float V;
	map<int, onode> observations;
	onode *father;
};

typedef struct utc_result_t {
	onode *root;
	int best_action;
	float best_value;
} utc_result;

utc_result Search(int periods, const vec *initial_belief, mat *ims);
float MC_utc(onode *h_root, vec *initial_belief, mat *ims, int periods);

#endif
