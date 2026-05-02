#pragma once

#include "../globals.h"
#include "../mpc/RSS.h"
#include "../util/util.cuh"
#include "../util/functors.h"
#include "dtFunc.h"

#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>

template<typename T, template<typename, typename...> typename Share>
class DecisionTree {
    public:
        Share<T> input; // data matrix, after transpose we dont need to adjust original data, dimension is N_F*N_D
        Share<T> input_label; // label matrix, dimension is N_D*1
        Share<uint8_t> input_bs; // data matrix in boolean share. store the items in column-wise (ie in feature-wise)

        Share<T> tree;
        Share<T> leafClass;

        Share<uint8_t> markerFlag; // indicate data belongs to this node or not
        Share<T> nodeFlag; // internal: 0, leaf: 1, dummy: 2
        Share<T> counter;
        Share<T> data_count; // record the number of data that has label = 0/1 in each leaf, size = 2 * num_LC
        Share<uint8_t> maskFlag; // indicate if feature is used or not
        Share<T> split_decisions;

        Share<T> M_T;

        int tree_level;
        // int isFinished;

        DecisionTree();
        ~DecisionTree();

        void counter_increase(std::vector<double> &data, std::vector<double> &data_trans, std::vector<double> &label);
        void compute_info_gain_SGX(std::vector<double> &node_split_decisions, std::vector<double> &hostNodeFlag, std::vector<double> &newleafClass, int &isFinished);
        void compute_info_gain_MPC(int &isFinished);
        void node_split_SGX(std::vector<double> &node_split_decisions, std::vector<double> &hostNodeFlag, std::vector<double> &newleafClass);
        void node_split_MPC();

        // void inference(std::vector<double> &data, std::vector<double> &label);
        void inference_offline();
        void inference_online(Share<T> &inf_data, Share<T> &result_label);
};