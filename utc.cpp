#include <random>
#include "float.h"
#include "utc.h"
#include "parameters.h"

using namespace std;

/* Silver, David, and Joel Veness. "Monte-Carlo Planning in Large POMDPs." NIPS. Vol. 23. 2010. */

/**
 - n: periods to go
 - A: set of available actions
 - B: initial Belief
 - imx: improvement matrices for 
 - s: true optimum drawn from in each iteration
*/

/* The Generator returns the improvement achieved in this period. */
static inline int Generator(int state, int action, mat *ims) {
	double *improvement_dist = ims[action].colptr(state);
	return random_draw(improvement_dist, observation_count);
}

/*
  Version 1: Choose actions randomly with uniform distribution
  Version 2: Apply the optimization for a static server count and roll out.
  Version 3: In every period optimize for the best (static) server count.
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
	int history_length = 0;
	onode *on = h;
	do {
		anode *an = on->father;
		if(an == NULL) break;
		on = an->father;
		history_length++;
	}while(1);
	
	int *history; // array with aoao...
	history = (int*) malloc(sizeof(int)*2*history_length);
	on = h;
	for(int h=history_length-1;h>=0; h--) {
		history[h*2+1] = on->observation_index;
		anode *an = on->father;
		history[h*2] = an->action_index;
	}

	double action_value = 0.0;
	if(history_length > 0){
		const vec *orig_belief = initial_belief;
		vec new_belief;
		for(int i=0;i<history_length;i++) {
			new_belief = belief_update(orig_belief, &ims[history[2*i]], history[2*i+1]);
			orig_belief = &new_belief;
		}
		best_action(&new_belief, ims, n, &action_value);
	} else {
		best_action(initial_belief, ims, n, &action_value);
	}
	free(history);
#endif
	
	return value;
}

float Simulate(int state, onode *h, const vec *initial_belief, int n, mat *ims) {
	if(n == 0) return 0.0;

	// if no children exist
	if(h->actions.size() == 0) {
		for(int a=0; a<action_count;a++)
			h->actions.insert(pair<int, anode>(a, {a, 0, 0.0, map<int, onode>(), h})); // action_index, N, V, observations, father
		return Rollout(state, h, initial_belief, n, ims);
	}

	// look for best action
	anode *best_action_node = NULL;
	int best_action = -1;
	float best_action_value = -FLT_MAX;
	float c = 200.0;
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

utc_result Search(int periods, const vec *initial_belief, mat *ims) {
	onode *h_root = new (onode){0, 0, map<int, anode>(), NULL}; // observation_index, N, actions, father
	int N = 100000;
	
	cout << "start" << endl;
	for(int i=0;i<N;i++) {
		if(i%10000 == 0)
			cout << "i: " << i << endl;
		int state = random_draw(initial_belief->memptr(), initial_belief->n_elem);
		Simulate(state, h_root, initial_belief, periods, ims);
	}
	cout << "finished" << endl;

	int best_action = -1;
	float best_value = -100000000.0;
	for (int i=0; i<action_count; i++) {
		anode *n = &h_root->actions[i];
		float value = n->V;
		cout << "Servers: " << i << " Trys: " << n->N << " Value: " << value << endl;
		if(value > best_value) {
			best_action = i;
			best_value = value;
		}
	}

	cout << "nodes: " << onode_count(h_root) << endl;
	
	utc_result res;
	res.root = h_root;
	res.best_action = best_action;
	res.best_value = best_value;
	return res;
}

float MC_utc(onode *h_root, vec *initial_belief, mat *ims, int periods) {
	srand (time(NULL));
	int N = 100;
	float value = 0.0;
	h_root->father = NULL;

	for(int k=0;k<N;k++) {
		cout << "*****" << endl;
		int o_pos = random_draw(initial_belief->memptr(),observation_count);

		vec current_belief;
		vec *current_belief_ptr = initial_belief;
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
			cout << "a: " << best_action;

			/* If the current node does not have a high enough visitation rate. Then do some more tree searching. */
			cout << " N: " << best_action_node->N;
			if(best_action_node->N < 10000) {
				int N = 10000;
				for(int i=0;i<N;i++) {
					int state = random_draw(initial_belief->memptr(), initial_belief->n_elem);
					Simulate(state, current, initial_belief, n, ims);
				}
			}

			int improvement = random_draw(ims[best_action].colptr(o_pos), ims[best_action].n_rows);
			cout << " o: " << improvement << endl;
			value += (float)improvement - (best_action * server_cost);

			current_belief = belief_update(current_belief_ptr, &ims[best_action], improvement);
			current_belief_ptr = &current_belief;

			if(n>1) {
				auto next = best_action_node->observations.find(improvement);
				if(next == best_action_node->observations.end()) {
					cout << "Ooops. I have never been here..." << endl;
					next = best_action_node->observations.insert(pair<int, onode>(improvement, {improvement, 0, map<int, anode>(), best_action_node})).first; // improvement_index, N, actions, father
				}
				current = &next->second;
			}
		}
	}

	return value/N;
}
