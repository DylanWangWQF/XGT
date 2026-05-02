#ifndef DTFUNC_H
#define DTFUNC_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "../globals.h"

using namespace std;

int compute_information_gain(vector<double> &hostCounter, vector<double> &hostNodeFlag, int &cur_leaf_count, int LCidx, int num_LC, double &newlabel);

vector<int> sort_indexes(const vector<float> &v);

#endif