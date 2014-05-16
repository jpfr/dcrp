#include <random>
#include "float.h"
#include "utc.h"
#include "parameters.h"

using namespace std;

/**
 - n: periods to go
 - A: set of available actions
 - B: initial Belief
 - ims: improvement matrices for 
 - s: true optimum drawn from in each iteration
*/

/* The Generator returns the improvement achieved in this period. */
static inline int Generator(int state, int action, mat *ims) {
	double *improvement_dist = ims[action].colptr(state);
	return random_draw(improvement_dist, observation_count);
}

int onode_count(onode *node) {
	int count = 1 + node->actions.size(); // itself

	auto aend = node->actions.end();
	for(auto ait = node->actions.begin(); ait!=aend;ait++) {
		anode *va = &ait->second;
		auto oend = va->observations.end();
		for(auto oit = va->observations.begin(); oit!=oend; oit++)
			count += onode_count(&oit->second);
	}

	return count;
}

void onode_show_N(onode *node) {
	auto aend = node->actions.end();
	for(auto ait = node->actions.begin(); ait!=aend;ait++)
		cout << "Action" << ait->first << ": " << ait->second.N << endl;
}

vec update_history_belief(onode *h, const vec *initial_belief, const mat *ims) {
	int history_length = 0;
	onode *on = h;
	do {
		anode *an = on->father;
		if(an == NULL) break;
		on = an->father;
		history_length++;
	} while(1);
	if(history_length == 0)
		return *initial_belief;
	
	int *history; // array with aoao...
	history = (int*) malloc(sizeof(int)*2*history_length);
	on = h;
	for(int h=history_length-1;h>=0; h--) {
		history[h*2+1] = on->observation_index;
		anode *an = on->father;
		history[h*2] = an->action_index;
	}

	vec current_belief;
	vec *current_belief_ptr = (vec *) initial_belief;
	// compute current belief based on the action (ims) and the observation (directly in the history)
	for(int i=0;i<history_length;i++) {
		current_belief = belief_update(current_belief_ptr, &ims[history[2*i]], history[2*i+1]);
		current_belief_ptr = &current_belief;
	}
	free(history);
	return current_belief;
}

/*
  Version 1: Choose actions randomly with uniform distribution
  Version 2: Apply the optimization for a static server count and roll out.
*/

float Rollout(int state, onode *h, const vec *initial_belief, int n, mat *ims) {
	if(n == 0) return 0.0;
	float value = 0.0;

#if 1
	/* Version 1 */
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<> dis(0, action_count-1);
	for(; n>0;n--) {
		int action = dis(gen);
		int improvement = Generator(state, action, ims);
		value += improvement - action * server_cost;
		state -= improvement;
	}
#endif

#if 0
	/* Version 2 */
	// compute current belief based on the observed history.
	vec current_belief = update_history_belief(h, initial_belief, ims);
	for(; n>0;n--) {
		int action = best_action(&current_belief, ims, n, NULL);
		int improvement = Generator(state, action, ims);
		value += improvement - action * server_cost;
		state -= improvement;
		current_belief = belief_update(&current_belief, &ims[action], improvement);
	}
#endif
	
	return value;
}

float Simulate(int state, onode *h, const vec *initial_belief, int n, mat *ims) {
	if(n == 0) return 0.0;

	// if no children exist
	if(h->actions.size() == 0) {
		// compute static optimum at this point..
		vec current_belief = update_history_belief(h, initial_belief, ims);
		double best_vstatic = -10000.0;
		for(int a=0; a<action_count;a++){
			double vstatic = V_static(&current_belief, &ims[a], n);
			h->actions.insert(pair<int, anode>(a, {a, 100, (float)vstatic, map<int, onode>(), h})); // initialize anode
			if(vstatic > best_vstatic)
				best_vstatic = vstatic;
		}
		return best_vstatic; //Rollout(state, h, initial_belief, n, ims);
	}

	// look for best action
	anode *best_action_node = NULL;
	int best_action = -1;
	float best_action_value = -FLT_MAX;
	float c = 25.0;
	auto end = h->actions.end();
	for (auto it = h->actions.begin(); it!=end; it++) {
		anode *hb = &it->second;
		float vb = hb->V + c * sqrt(log(h->N+1)/(hb->N+1));
		if(vb > best_action_value) {
			best_action_node = hb;
			best_action = hb->action_index;
			best_action_value = vb;
		}
	}
	if(best_action_node == NULL) exit(1);

	// apply action and observe
	int improvement = Generator(state, best_action, ims);
	int new_state = state-improvement;
	float immediate_value = (float)improvement - ((float)best_action*server_cost);

	auto hao_iter = best_action_node->observations.find(improvement);
	if(hao_iter == best_action_node->observations.end()) // not found
		hao_iter = best_action_node->observations.insert(pair<int, onode>(improvement, {improvement, 0, map<int, anode>(), best_action_node})).first; // improvement_index, N, actions, father
	onode *hao = &hao_iter->second;

	float R = immediate_value + Simulate(new_state, hao, initial_belief, n-1, ims);
	h->N += 1;
	best_action_node->N += 1;
	best_action_node->V += (R - best_action_node->V) / (float)best_action_node->N;

	return R;
}

utc_result Search(int periods, const vec *initial_belief, mat *ims) {
	onode *h_root = new (onode){0, 0, map<int, anode>(), NULL}; // observation_index, N, actions, father
	int N = 200000;
	vec convergence(N);
	
	int best_action = -1;
	float best_value = -100000000.0;
	for(int i=0;i<N;i++) {
		if(i%1000 == 0)
			cout << i << endl;
		int state = random_draw(initial_belief->memptr(), initial_belief->n_elem);
		Simulate(state, h_root, initial_belief, periods, ims);
		best_action = -1;
		best_value = -100000000.0;
		for (int l=0; l<action_count; l++) {
			anode *n = &h_root->actions[l];
			float value = n->V;
			if(value > best_value) {
				best_action = l;
				best_value = value;
			}
		}
		convergence.at(i) = best_value;
	}

	best_action = -1;
	best_value = -100000000.0;
	for (int i=0; i<action_count; i++) {
		anode *n = &h_root->actions[i];
		float value = n->V;
		if(value > best_value) {
			best_action = i;
			best_value = value;
		}
	}

	utc_result res;
	res.root = h_root;
	res.best_action = best_action;
	res.best_value = best_value;
	res.convergence = convergence;
	return res;
}

vec MC_utc(onode *h_root, vec *initial_belief, mat *ims, int periods) {
	srand (time(NULL));
	int N = 500;
	vec results = vec(N);
	float value = 0.0;
	h_root->father = NULL;

	for(int k=0;k<N;k++) {
		cout << k << endl;
		value = 0.0;
		int o_pos = random_draw(initial_belief->memptr(),observation_count);

		onode *current = h_root;

		for(int n=periods;n>0;n--) {
			anode *best_action_node = NULL;
			int best_action = -1;
			float best_action_value = -FLT_MAX;

			auto end = current->actions.end();
			for (auto it = current->actions.begin(); it!=end; it++) {
				anode *hb = &it->second;
				float vb = hb->V;
				if(vb > best_action_value) {
					best_action_node = hb;
					best_action = hb->action_index;
					best_action_value = vb;
				}
			}

			/* If the current node does not have a high enough visitation rate. Then do some more tree searching. */
			while(best_action_node->N < 100) {
				int N2 = 100;
				vec current_belief = update_history_belief(best_action_node->father, initial_belief, ims);
				for(int i=0;i<N2;i++) {
					int state = random_draw(current_belief.memptr(), initial_belief->n_elem);
					Simulate(state, best_action_node->father, &current_belief, n, ims);
				}
			}

			int improvement = random_draw(ims[best_action].colptr(o_pos), ims[best_action].n_rows);
			o_pos = o_pos - improvement;
			value += (float)improvement - (best_action * server_cost);

			int counter = 1;
			if(n>1) {
				while(1) {
						auto next = best_action_node->observations.find(improvement);
						if(next == best_action_node->observations.end()) {
							int N2 = 1000;
							vec current_belief = update_history_belief(best_action_node->father, initial_belief, ims);
							for(int i=0;i<N2;i++) {
								int state = random_draw(current_belief.memptr(), initial_belief->n_elem);
								Simulate(state, best_action_node->father, &current_belief, n, ims);
							}
							counter++;
							if(counter % 100 == 0) {
								improvement = random_draw(ims[best_action].colptr(o_pos), ims[best_action].n_rows);
							}
						} else {
							current = &next->second;
							break;
						}
				}
			}
		}
		results.at(k) = value;
	}
	return results;
}
