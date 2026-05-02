#include "dtFunc.h"
#include <iostream>

int compute_information_gain(
        vector<double> &hostCounter,
        vector<double> &hostNodeFlag, 
        int &cur_leaf_count, 
        int LCidx, 
        int num_LC,
        double &newlabel)
{
    std::vector<float> info_gain_vals(ATTRIBUTE_COUNT_TOTAL);
    std::vector<float> check_same_label(CLASS_COUNT);
    int leaf_counter_row_len = ATTRIBUTE_COUNT_TOTAL * 2;
    // index in Tree array
    int cur_Idx = num_LC + LCidx - 1;
    int leftChildIdx = 2 * cur_Idx + 1;
    int rightChildIdx = 2 * cur_Idx + 2;

    // check if all the labels are same
    for (int i = 0; i < CLASS_COUNT; i++)
    {
        check_same_label[i] = hostCounter[(2 + i) * leaf_counter_row_len] +
        hostCounter[(2 + i) * leaf_counter_row_len + 1];
    }
    std::sort(check_same_label.begin(), check_same_label.end(), std::greater<float>());

    newlabel = check_same_label[0];

    if (check_same_label[0] > 0 && ((int) check_same_label[1]) == 0)
    {
        hostNodeFlag[leftChildIdx] = 2;
        hostNodeFlag[rightChildIdx] = 2;
        cur_leaf_count ++;
        return 0;
    }

    for (int j = 0; j < ATTRIBUTE_COUNT_TOTAL; j++)
    {
        if ((int) hostCounter[leaf_counter_row_len + 2 * j] == 0) // check mask
        {
            info_gain_vals[j] = FLT_MAX;
        }
        else
        {
            float sum_0 = 0.0, sum_00 = 0.0, sum_01 = 0.0;
            for (int k = 0; k < CLASS_COUNT; k++)
            {
                float param_00 = (float) hostCounter[(2 + k) * leaf_counter_row_len + j * 2] / hostCounter[j * 2];
                param_00 = isnan(param_00) ? (float) 0.0 : param_00;
                float log_param_00 = log2f((float) param_00);
                log_param_00 = isnan(log_param_00) ? (float) 0.0 : -log_param_00;

                float param_01 = (float) hostCounter[(2 + k) * leaf_counter_row_len + j * 2 + 1] / hostCounter[j * 2 + 1];
                param_01 = isnan(param_01) ? (float) 0.0 : param_01;
                float log_param_01 = log2f((float) param_01);
                log_param_01 = isnan(log_param_01) ? (float) 0.0 : -log_param_01;

                sum_00 += param_00 * log_param_00; 
                sum_01 += param_01 * log_param_01; 
            }
            sum_0 = sum_00 + sum_01;
            info_gain_vals[j] = isnan(sum_0) ? FLT_MAX : sum_0;
        }
    }

    
    hostNodeFlag[cur_Idx] = 0; // 0 is internal
    vector<int> attr_idx = sort_indexes(info_gain_vals);

    return attr_idx[0];
}

vector<int> sort_indexes(const vector<float> &v)
{
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] < v[i2]; });
    return idx;
}