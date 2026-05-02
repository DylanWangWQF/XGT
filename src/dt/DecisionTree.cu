#pragma once

#include "DecisionTree.h"

extern nlohmann::json gst_config;

using namespace thrust::placeholders;

template<typename T, template<typename, typename...> typename Share>
DecisionTree<T, Share>::DecisionTree() : 
        input(INSTANCE_COUNT_PER_ITER * ATTRIBUTE_COUNT_TOTAL), 
        input_bs(INSTANCE_COUNT_PER_ITER * ATTRIBUTE_COUNT_TOTAL), 
        input_label(INSTANCE_COUNT_PER_ITER), 
        tree(NODE_COUNT),
        counter(ATTRIBUTE_COUNT_TOTAL * 2),
        data_count(2),
        leafClass(1), // resize after training finish
        nodeFlag(NODE_COUNT),
        markerFlag(INSTANCE_COUNT_TOTAL), // resize for each layer, INSTANCE_COUNT_TOTAL * num_leaf
        maskFlag(ATTRIBUTE_COUNT_TOTAL), // resize for each layer, INSTANCE_COUNT_TOTAL * num_leaf
        split_decisions(1),
        M_T(1){
    tree.zero(); // first is root
    tree_level = 0; // root, 0-th level
    nodeFlag.fill(1); // init all nodes are leaf
    markerFlag.fill(1); // indicate all the data is in the root 
    maskFlag.fill(1); // init all features are not used
}

template<typename T, template<typename, typename...> typename Share>
DecisionTree<T, Share>::~DecisionTree()
{
	// do nothing
}

template<typename T, template<typename, typename...> typename Share>
void DecisionTree<T, Share>::counter_increase(std::vector<double> &data, std::vector<double> &data_trans, std::vector<double> &label){

    input.setPublic(data, false); // data matrix, A share, N_F*N_D (data-wise)
    input_bs.setPublic(data_trans, false); // data matrix, B share (feature-wise)
    input_label.setPublic(label, false); // label matrix, A share, N_D*1
    int num_LC = 1 << tree_level; // number of leaves in current layer

    // 1. update markerFlag and generate marker matrix M_chi
    if (tree_level != 0)
    {
        int old_num_LC = 1 << (tree_level - 1);
        // 1.1 get assigned feature from tree array
        Share<T> ParentFeaidx(old_num_LC);
        thrust::copy_n(tree.getShare(0)->begin() + (old_num_LC - 1), old_num_LC, ParentFeaidx.getShare(0)->begin());
        thrust::copy_n(tree.getShare(1)->begin() + (old_num_LC - 1), old_num_LC, ParentFeaidx.getShare(1)->begin());
        // 1.2 fill selected feature idx for each leaf, aligned with ATTRIBUTE_COUNT_TOTAL
        Share<T> ParentFeaidx_fill(old_num_LC * ATTRIBUTE_COUNT_TOTAL);
        fill_in_batch(ParentFeaidx, ParentFeaidx_fill, ATTRIBUTE_COUNT_TOTAL, 0); 
        // 1.3 generate sequence of feature idx for each leaf
        DeviceData<T> fea_idx_seq(ParentFeaidx_fill.size());
        sequence_in_batch(fea_idx_seq, ATTRIBUTE_COUNT_TOTAL);
        // 1.4 use EQ to determine the selected feature idx in each leaf
        ParentFeaidx_fill -= fea_idx_seq;
        Share<uint8_t> EQ_selectedFea(ParentFeaidx_fill.size());
        EQ(ParentFeaidx_fill, EQ_selectedFea);
        // 1.5 extend each element of EQ to N_D size 
        //     from ATTRIBUTE_COUNT_TOTAL * old_num_LC
        //     to INSTANCE_COUNT_PER_ITER * ATTRIBUTE_COUNT_TOTAL * old_num_LC
        //     = len1 * len2 * len3
        Share<uint8_t> EQ_selectedFea_fill(EQ_selectedFea.size() * INSTANCE_COUNT_PER_ITER); 
        fill_in_batch(EQ_selectedFea, EQ_selectedFea_fill, INSTANCE_COUNT_PER_ITER, 0);
        // 1.6 copy input_bs in old_num_LC times
        //     input_bs_fill.size = INSTANCE_COUNT_PER_ITER * ATTRIBUTE_COUNT_TOTAL
        //     input_bs_fill is B share and in feature-wise
        Share<uint8_t> input_bs_fill(EQ_selectedFea_fill.size());
        thrust::copy_n(thrust::make_permutation_iterator(input_bs.getShare(0)->begin(), thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1%input_bs.size())), input_bs_fill.size(), input_bs_fill.getShare(0)->begin());
        thrust::copy_n(thrust::make_permutation_iterator(input_bs.getShare(1)->begin(), thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1%input_bs.size())), input_bs_fill.size(), input_bs_fill.getShare(1)->begin());
        // 1.7 Mul with delayed reshare, results are selected feature vec in each node of old layer
        //     first aug in AccMulWithDelayShare will be modified
        //     selected_dataVec stores the selected i-th feature's value of all data
        Share<uint8_t> selected_dataVec(INSTANCE_COUNT_PER_ITER * old_num_LC);
        AccMulWithDelayShare(EQ_selectedFea_fill, input_bs_fill, selected_dataVec, INSTANCE_COUNT_PER_ITER, ATTRIBUTE_COUNT_TOTAL, old_num_LC);
        // 1.8 update the marker vector for right child in current layer
        //     markerFlag currently stores marker vecs of parents in old layer
        selected_dataVec *= markerFlag;
        // 1.9 get marker of left child
        markerFlag -= selected_dataVec; // now markerFlag stores updated chi for left child
        // 1.11 merge left and right child into new markerFlag
        Share<uint8_t> new_markerFlag(INSTANCE_COUNT_PER_ITER * num_LC);
        merge_vecs(markerFlag, selected_dataVec, new_markerFlag, INSTANCE_COUNT_PER_ITER);
        markerFlag.resize(new_markerFlag.size());
        markerFlag.zero();
        markerFlag += new_markerFlag;
        // 1.12 B2A for selected_dataVec, only right child stored in M_chi, instead of markerFlag. used for Hamadard product
        // In doing so, we reduce half of MatMul in current layer
        Share<T> ones(selected_dataVec.size());
        ones.fill(1);
        Share<T> zeros(selected_dataVec.size());
        zeros.fill(0);
        Share<T> M_chi(selected_dataVec.size());
        selectShare(zeros, ones, selected_dataVec, M_chi);

        // 2.    After update the marker vectors for all leaf node, we next need to compute Hamadard product.
        // 2.1   compute M' = M_L · M_chi. In the following, we only calculate the right child in current layer.
        // 2.1.1 compute M_L · M_chi in single column (ie size(M_L)=N_D*1, label=1)
        Share<T> input_label_fill(M_chi.size()); // INSTANCE_COUNT_PER_ITER * old_num_LC
        thrust::copy_n(thrust::make_permutation_iterator(input_label.getShare(0)->begin(), thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1%input_label.size())), input_label_fill.size(), input_label_fill.getShare(0)->begin());
        thrust::copy_n(thrust::make_permutation_iterator(input_label.getShare(1)->begin(), thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1%input_label.size())), input_label_fill.size(), input_label_fill.getShare(1)->begin());
        input_label_fill *= M_chi;
        // 2.1.2 computer another column in M_L, opposite of input_label (ie label = 0)
        M_chi -= input_label_fill;
        // 2.1.4 merge two vectors into result matrix M', stored in column-wise
        Share<T> M_L_chi(M_chi.size() * 2); // =2*M_chi/input_label_fill.size()
        merge_vecs(M_chi, input_label_fill, M_L_chi, INSTANCE_COUNT_PER_ITER);
        // 2.1.5 record the number of data, used in calculate n_i0k
        Share<T> dataCount_temp(num_LC); // numLC=old_num_LC*2 indicates 2 cols for each left child, doesn't mean numLC nodes
        auto my_flags_iterator = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1/INSTANCE_COUNT_PER_ITER);
        thrust::reduce_by_key(my_flags_iterator, my_flags_iterator + M_L_chi.size(), M_L_chi.getShare(0)->begin(), thrust::make_discard_iterator(), dataCount_temp.getShare(0)->begin());
        thrust::reduce_by_key(my_flags_iterator, my_flags_iterator + M_L_chi.size(), M_L_chi.getShare(1)->begin(), thrust::make_discard_iterator(), dataCount_temp.getShare(1)->begin());
        Share<T> dataCount_LeftChild(dataCount_temp.size());
        dataCount_LeftChild += data_count;
        dataCount_LeftChild -= dataCount_temp;
        data_count.resize(num_LC * 2);
        data_count.zero();
        merge_vecs(dataCount_LeftChild, dataCount_temp, data_count, 2); // L: n0,n1 + R: n0,n1 + L: n0,n1 ...

        // 3. Compute M_Q * M_L_chi (only for right child)
        Share<T> MR_RightChild(ATTRIBUTE_COUNT_TOTAL * num_LC); // ATTRIBUTE_COUNT_TOTAL * num_LC = ATTRIBUTE_COUNT_TOTAL * 2 * old_num_LC 
        matmul(input, M_L_chi, MR_RightChild, ATTRIBUTE_COUNT_TOTAL, num_LC, INSTANCE_COUNT_PER_ITER, false, false, false);
        Share<T> MR_LeftChild(ATTRIBUTE_COUNT_TOTAL * num_LC);
        MR_LeftChild += counter;
        MR_LeftChild -= MR_RightChild;
        counter.resize(ATTRIBUTE_COUNT_TOTAL * 2 * num_LC);
        counter.zero();
        merge_vecs(MR_LeftChild, MR_RightChild, counter, ATTRIBUTE_COUNT_TOTAL * 2);
    }
    else
    {
        // get query for label=0
        Share<T> opposite_label0(INSTANCE_COUNT_PER_ITER);
        opposite_label0 += input_label;
        opposite_label0 *= (T) -1;
        opposite_label0 += 1;
        // merge for M_L_chi
        Share<T> M_L_chi(INSTANCE_COUNT_PER_ITER * 2);
        merge_vecs(opposite_label0, input_label, M_L_chi, INSTANCE_COUNT_PER_ITER);

        // compute count for label = 0/1
        auto my_flags_iterator = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1/INSTANCE_COUNT_PER_ITER);
        thrust::reduce_by_key(my_flags_iterator, my_flags_iterator + M_L_chi.size(), M_L_chi.getShare(0)->begin(), thrust::make_discard_iterator(), data_count.getShare(0)->begin());
        thrust::reduce_by_key(my_flags_iterator, my_flags_iterator + M_L_chi.size(), M_L_chi.getShare(1)->begin(), thrust::make_discard_iterator(), data_count.getShare(1)->begin());
        
        // counter is initialized with ATTRIBUTE_COUNT_TOTAL*2
        matmul(input, M_L_chi, counter, ATTRIBUTE_COUNT_TOTAL, 2, INSTANCE_COUNT_PER_ITER, false, false, false);
    }
    
}

// This part is similar to GTree. A SPU-based version is available here: https://github.com/secretflow/sml/blob/main/sml/tree/tree.py
template<typename T, template<typename, typename...> typename Share>
void DecisionTree<T, Share>::compute_info_gain_MPC(int &isFinished)
{
    if (tree_level + 1 == TREE_DEPTH)
    {
        isFinished = 1;
        return;
    }

    int num_LC = 1 << tree_level;
    int next_num_LC = 1 << (tree_level + 1);

    int leaf_counter_row_len = ATTRIBUTE_COUNT_TOTAL * 2;
    size_t num_cValTotal = num_LC * LEAF_COUNTER_SIZE;

    split_decisions.resize(num_LC);
    split_decisions.zero();
    
    Share<uint8_t> cur_isDummy(num_LC);
    Share<uint8_t> next_isDummy(next_num_LC);

    // 1. compute Gini Index for all counter (1,2)
    // 1.1 extract the rows from changed counters
    Share<T> counter_1st_row(num_LC * leaf_counter_row_len);
    Share<T> counter_2nd_row(num_LC * leaf_counter_row_len);
    Share<T> counter_3rd_row(num_LC * leaf_counter_row_len);
    Share<T> counter_4th_row(num_LC * leaf_counter_row_len);

    int offset1 = 0, offset2 =0, offset3 =0, offset4 =0;
    for (int i = 0; i < num_LC; i++)
    {
        offset1 = i * LEAF_COUNTER_SIZE;
        offset2 = i * LEAF_COUNTER_SIZE + leaf_counter_row_len;
        offset3 = i * LEAF_COUNTER_SIZE + 2 * leaf_counter_row_len;
        offset4 = i * LEAF_COUNTER_SIZE + 3 * leaf_counter_row_len;
        // 1st row
        thrust::copy_n(counter.getShare(0)->begin() + offset1, leaf_counter_row_len, counter_1st_row.getShare(0)->begin() + i * leaf_counter_row_len);
        thrust::copy_n(counter.getShare(1)->begin() + offset1, leaf_counter_row_len, counter_1st_row.getShare(1)->begin() + i * leaf_counter_row_len);
        // 2nd row
        thrust::copy_n(counter.getShare(0)->begin() + offset2, leaf_counter_row_len, counter_2nd_row.getShare(0)->begin() + i * leaf_counter_row_len);
        thrust::copy_n(counter.getShare(1)->begin() + offset2, leaf_counter_row_len, counter_2nd_row.getShare(1)->begin() + i * leaf_counter_row_len);
        // 3rd row
        thrust::copy_n(counter.getShare(0)->begin() + offset3, leaf_counter_row_len, counter_3rd_row.getShare(0)->begin() + i * leaf_counter_row_len);
        thrust::copy_n(counter.getShare(1)->begin() + offset3, leaf_counter_row_len, counter_3rd_row.getShare(1)->begin() + i * leaf_counter_row_len);
        // 4th row
        thrust::copy_n(counter.getShare(0)->begin() + offset4, leaf_counter_row_len, counter_4th_row.getShare(0)->begin() + i * leaf_counter_row_len);
        thrust::copy_n(counter.getShare(1)->begin() + offset4, leaf_counter_row_len, counter_4th_row.getShare(1)->begin() + i * leaf_counter_row_len);
    }
    // 1.2 get the mask flag
    Share<uint8_t> isUsed(num_LC * leaf_counter_row_len); // check mask row
    EQ(counter_2nd_row, isUsed);
    // 1.2.1 replace with 1 in counter_1st_row (denominator);
    Share<T> ones(counter_1st_row.size());
    ones.fill(1);
    Share<T> pre_counter_1st_row(counter_1st_row.size());
    selectShare(counter_1st_row, ones, isUsed, pre_counter_1st_row);
    counter_1st_row.zero();
    counter_1st_row += pre_counter_1st_row;
    // 1.2.2 replace with 0 om 3rd and 4th row
    Share<T> zeros(counter_3rd_row.size());
    zeros.zero();
    Share<T> pre_counter_3rd_row(counter_3rd_row.size());
    Share<T> pre_counter_4th_row(counter_4th_row.size());
    selectShare(counter_3rd_row, zeros, isUsed, pre_counter_3rd_row);
    selectShare(counter_4th_row, zeros, isUsed, pre_counter_4th_row);
    counter_3rd_row.zero();
    counter_3rd_row += pre_counter_3rd_row;
    counter_4th_row.zero();
    counter_4th_row += pre_counter_4th_row;

    // 1.3 Gini Index: mul + inverse
    // 1.3.1 calculate numerator
    Share<T> numerator(num_LC * leaf_counter_row_len);
    // |D^v|^2
    numerator += counter_1st_row;
    numerator *= counter_1st_row;
    // Sum( (n_k)^2 )
    counter_3rd_row *= counter_3rd_row;
    counter_4th_row *= counter_4th_row;
    counter_3rd_row += counter_4th_row;
    // |D^v|^2 - Sum( (n_k)^2 )
    numerator -= counter_3rd_row;

    // 1.3.2 calculate denominator
    Share<T> denominator(num_LC * leaf_counter_row_len);
    dividePublic(counter_1st_row, (T)scaleFactor);

    // 1.3.3 calculate inverse 
    inverse(counter_1st_row, denominator);

    // 1.3.4 calculate final Gini Index
    numerator *= denominator;

    // 1.4 aggregate each feature's Gini Index
    Share<T> giniIndex(num_LC * ATTRIBUTE_COUNT_TOTAL);
    auto my_flags_iterator = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1/2);
    thrust::reduce_by_key(my_flags_iterator, my_flags_iterator + numerator.size(), numerator.getShare(0)->begin(), thrust::make_discard_iterator(), giniIndex.getShare(0)->begin());
    thrust::reduce_by_key(my_flags_iterator, my_flags_iterator + numerator.size(), numerator.getShare(1)->begin(), thrust::make_discard_iterator(), giniIndex.getShare(1)->begin());

    // 1.5 use reduce_sum to replace the value when mask = 0;
    Share<T> pre_reduceMax(num_LC);
    auto my_flags_iterator1 = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1/ATTRIBUTE_COUNT_TOTAL);
    thrust::reduce_by_key(my_flags_iterator1, my_flags_iterator1 + giniIndex.size(), giniIndex.getShare(0)->begin(), thrust::make_discard_iterator(), pre_reduceMax.getShare(0)->begin());
    thrust::reduce_by_key(my_flags_iterator1, my_flags_iterator1 + giniIndex.size(), giniIndex.getShare(1)->begin(), thrust::make_discard_iterator(), pre_reduceMax.getShare(1)->begin());
    
    // 1.5.1 expand size
    Share<T> reduceMax(num_LC * ATTRIBUTE_COUNT_TOTAL);
    fill_in_batch(pre_reduceMax, reduceMax, ATTRIBUTE_COUNT_TOTAL, 0);

    // 1.5.2 replace using the isUsed flag
    Share<uint8_t> isUsed_single(num_LC * ATTRIBUTE_COUNT_TOTAL);
    for (int i = 0; i < num_LC; i++)
    {
        for (int j = 0; j < ATTRIBUTE_COUNT_TOTAL; j++)
        {
            isUsed_single.getShare(0)->begin()[i * ATTRIBUTE_COUNT_TOTAL + j] = isUsed.getShare(0)->begin()[i * leaf_counter_row_len + j * 2];
            isUsed_single.getShare(1)->begin()[i * ATTRIBUTE_COUNT_TOTAL + j] = isUsed.getShare(1)->begin()[i * leaf_counter_row_len + j * 2];
        }
    }

    Share<T> finalGiniIndex(num_LC * ATTRIBUTE_COUNT_TOTAL);
    selectShare(giniIndex, reduceMax, isUsed_single, finalGiniIndex);
    
    // 1.6 select the index of the feature with the minimum Gini Index
    // fill finalGiniIndex if ATTRIBUTE_COUNT_TOTAL is not a pow-of-two
    int expand_length = 0;
    if ((ATTRIBUTE_COUNT_TOTAL & (ATTRIBUTE_COUNT_TOTAL - 1)) == 0)
    {
        expand_length = ATTRIBUTE_COUNT_TOTAL;
    }
    else
    {
        int scale = 1;
        while (scale < ATTRIBUTE_COUNT_TOTAL)
        {
            scale <<= 1;
        }
        expand_length = scale;
    }
    // std::cout << "Check expand_length: " << expand_length << std::endl;
    Share<T> expandedGiniIndex(num_LC * expand_length);
    if ((ATTRIBUTE_COUNT_TOTAL & (ATTRIBUTE_COUNT_TOTAL - 1)) == 0)
    {
        expandedGiniIndex += finalGiniIndex;
    }
    else
    {
        int off1 = 0, off2 = 0;
        for (int i = 0; i < num_LC; i++)
        {
            off1 = i * ATTRIBUTE_COUNT_TOTAL;
            off2 = i * expand_length;
            // off3 = i * expand_length + ATTRIBUTE_COUNT_TOTAL;
            thrust::copy_n(finalGiniIndex.getShare(0)->begin() + off1, ATTRIBUTE_COUNT_TOTAL, expandedGiniIndex.getShare(0)->begin() + off2);
            thrust::copy_n(finalGiniIndex.getShare(1)->begin() + off1, ATTRIBUTE_COUNT_TOTAL, expandedGiniIndex.getShare(1)->begin() + off2);
            
            fill_in_batch_withIdx(pre_reduceMax, expandedGiniIndex, expand_length, ATTRIBUTE_COUNT_TOTAL, (expand_length - ATTRIBUTE_COUNT_TOTAL));
        }
    }
    
    // find the maximum (*-1, minimum)
    expandedGiniIndex *= (T) (-1);
    Share<T> result(num_LC);
    Share<uint8_t> dresult(expandedGiniIndex.size());
    maxpool(expandedGiniIndex, result, dresult, expand_length);

    // fetch the index
    Share<T> featureIdx(expandedGiniIndex.size());
    featureIdx.zero();
    DeviceData<T> plainFeatureIdx(expandedGiniIndex.size());
    sequence_in_batch(plainFeatureIdx, expand_length);
    featureIdx += plainFeatureIdx;
    zeros.resize(expandedGiniIndex.size());
    zeros.zero();
    Share<T> pre_featureIdx(expandedGiniIndex.size());

    selectShare(zeros, featureIdx, dresult, pre_featureIdx);
    
    // reduce to the final index
    Share<T> pre_split_decisions(num_LC);
    auto my_flags_iterator2 = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1/expand_length);
    thrust::reduce_by_key(my_flags_iterator2, my_flags_iterator2 + pre_featureIdx.size(), pre_featureIdx.getShare(0)->begin(), thrust::make_discard_iterator(), pre_split_decisions.getShare(0)->begin());
    thrust::reduce_by_key(my_flags_iterator2, my_flags_iterator2 + pre_featureIdx.size(), pre_featureIdx.getShare(1)->begin(), thrust::make_discard_iterator(), pre_split_decisions.getShare(1)->begin());
    
    // 1.7 assign 0 to the dummy node as the split decision
    Share<T> node_flag(num_LC);
    thrust::copy_n(nodeFlag.getShare(0)->begin() + num_LC - 1, num_LC, node_flag.getShare(0)->begin());
    thrust::copy_n(nodeFlag.getShare(1)->begin() + num_LC - 1, num_LC, node_flag.getShare(1)->begin());
    DeviceData<T> dummyFlag(num_LC);
    dummyFlag.fill(2);
    node_flag -= dummyFlag;
    EQ(node_flag, cur_isDummy);
    zeros.resize(num_LC);
    zeros.zero();
    selectShare(pre_split_decisions, zeros, cur_isDummy, split_decisions);

    // 2 update node flag
    // 2.1 check whether the label is the same
    Share<T> label1(num_LC);
    Share<T> label2(num_LC);
    for (int i = 0; i < num_LC; i++)
    {
        label1.getShare(0)->begin()[i] = counter.getShare(0)->begin()[i * LEAF_COUNTER_SIZE + 2 * leaf_counter_row_len] + counter.getShare(0)->begin()[i * LEAF_COUNTER_SIZE + 2 * leaf_counter_row_len + 1];
        label1.getShare(1)->begin()[i] = counter.getShare(1)->begin()[i * LEAF_COUNTER_SIZE + 2 * leaf_counter_row_len] + counter.getShare(1)->begin()[i * LEAF_COUNTER_SIZE + 2 * leaf_counter_row_len + 1];

        label2.getShare(0)->begin()[i] = counter.getShare(0)->begin()[i * LEAF_COUNTER_SIZE + 3 * leaf_counter_row_len] + counter.getShare(0)->begin()[i * LEAF_COUNTER_SIZE + 3 * leaf_counter_row_len + 1];
        label2.getShare(1)->begin()[i] = counter.getShare(1)->begin()[i * LEAF_COUNTER_SIZE + 3 * leaf_counter_row_len] + counter.getShare(1)->begin()[i * LEAF_COUNTER_SIZE + 3 * leaf_counter_row_len + 1];
    }
    Share<uint8_t> check1(num_LC);
    Share<uint8_t> check2(num_LC);
    EQ(label1, check1);
    EQ(label2, check2);
    check1 ^= check2; // 1 means only one label
    // 2.2 update parent level
    node_flag.zero();
    zeros.resize(num_LC);
    zeros.zero();
    ones.resize(num_LC);
    ones.fill(1);
    selectShare(zeros, ones, check1, node_flag);
    
    Share<T> twos(num_LC);
    twos.fill(2);
    result.zero();
    selectShare(node_flag, twos, cur_isDummy, result); // until here
    // 2.3 update child level
    Share<T> childFlag(next_num_LC);
    ones.resize(next_num_LC);
    ones.fill(1);
    check1 |= cur_isDummy; // determine who is dummy
    fill_in_batch(check1, next_isDummy, 2, 0);
    twos.resize(next_num_LC);
    twos.fill(2);
    selectShare(ones, twos, next_isDummy, childFlag);
    // 2.4 copy the updated flag to node flag
    thrust::copy_n(result.getShare(0)->begin(), num_LC, nodeFlag.getShare(0)->begin() + num_LC - 1);
    thrust::copy_n(result.getShare(1)->begin(), num_LC, nodeFlag.getShare(1)->begin() + num_LC - 1);

    thrust::copy_n(childFlag.getShare(0)->begin(), next_num_LC, nodeFlag.getShare(0)->begin() + next_num_LC - 1);
    thrust::copy_n(childFlag.getShare(1)->begin(), next_num_LC, nodeFlag.getShare(1)->begin() + next_num_LC - 1);

    DeviceData<uint8_t> numDummy(next_num_LC);
    reconstruct(next_isDummy, numDummy);
    std::vector<double> hostnumDummy(next_num_LC);
    copyToHost(numDummy, hostnumDummy, false);
    int temp = 1;
    for (int i = 0; i < next_num_LC; i++)
    {
        temp *= (int) hostnumDummy[i];
    }
    if (temp == 1)
    {
        isFinished = 1;
        return;
    }
}

// This part is similar to GTree. A SPU-based version is available here: https://github.com/secretflow/sml/blob/main/sml/tree/tree.py
template<typename T, template<typename, typename...> typename Share>
void DecisionTree<T, Share>::node_split_MPC()
{
    int leaf_counter_row_len = ATTRIBUTE_COUNT_TOTAL * 2;
    int old_num_LC = 1 << tree_level;
    int new_tree_level = tree_level + 1;
    int new_num_LC = 1 << new_tree_level;

    int offset_sta = 0;
    int offset_end = 0;
    int old_mask_start_pos = 0;
    int new_mask_start_pos = 0;

    // 1. update the mask
    // only nodeFlag = 0, we need to update the mask
    // mask_idx: store the split_decisions
    Share<T> mask_idx(old_num_LC * ATTRIBUTE_COUNT_TOTAL);
    DeviceData<T> mask_idx_plain(mask_idx.size());
    Share<T> cur_node_flag(mask_idx.size());

    // fill each node's split decisions to mask_idx
    // old_num_LC || old_num_LC * ATTRIBUTE_COUNT_TOTAL
    fill_in_batch(split_decisions, mask_idx, ATTRIBUTE_COUNT_TOTAL, 0);

    // fill each node's flag to cur_node_flag
    // old_num_LC || old_num_LC * ATTRIBUTE_COUNT_TOTAL
    fill_in_batch(nodeFlag, cur_node_flag, ATTRIBUTE_COUNT_TOTAL, old_num_LC - 1);

    sequence_in_batch(mask_idx_plain, ATTRIBUTE_COUNT_TOTAL);
    
    mask_idx -= mask_idx_plain;
    
    Share<uint8_t> mask_idx_EQ(mask_idx.size()); // confirm the selected index in mask row
    EQ(mask_idx, mask_idx_EQ);

    Share<uint8_t> cur_node_flag_EQ(mask_idx.size()); // confirm the selected node in current level
    // internal (0) is 1, other (1, 2) is 0
    // so split_decisions = 0 for leaf (1) or dummy (2) doesnt matter
    EQ(cur_node_flag, cur_node_flag_EQ);

    mask_idx_EQ &= cur_node_flag_EQ;

    // expand size from old_num_LC * ATTRIBUTE_COUNT_TOTAL to new_num_LC * ATTRIBUTE_COUNT_TOTAL * 2
    // 1). from (old_num_LC * ATTRIBUTE_COUNT_TOTAL) to (old_num_LC * ATTRIBUTE_COUNT_TOTAL * 2)
    Share<uint8_t> pre_final_mask_flag(mask_idx.size() * 2);
    fill_in_batch(mask_idx_EQ, pre_final_mask_flag, 2, 0);
    
    // 2). from old_num_LC * ATTRIBUTE_COUNT_TOTAL * 2 to new_num_LC * ATTRIBUTE_COUNT_TOTAL * 2
    Share<uint8_t> final_mask_flag(new_num_LC * leaf_counter_row_len);
    // new_num_LC / old_num_LC = 2
    copy_in_batch(pre_final_mask_flag, final_mask_flag, leaf_counter_row_len, leaf_counter_row_len * 2); 
    

    // mask_raw: store the original mask of counter. mask_select: all 0
    Share<T> mask_raw(new_num_LC * leaf_counter_row_len);
    Share<T> mask_select(mask_raw.size());
    Share<T> mask_updated(mask_raw.size()); // result
    mask_select.fill(0);

    for (int i = 0; i < old_num_LC; i++)
    {
        old_mask_start_pos = i * LEAF_COUNTER_SIZE + leaf_counter_row_len;
        new_mask_start_pos = 2 * i * leaf_counter_row_len;
        // TODO: counter is not a independent vector, so copy_in_batch doesnot fit here
        thrust::copy_n(counter.getShare(0)->begin() + old_mask_start_pos, leaf_counter_row_len, mask_raw.getShare(0)->begin() + new_mask_start_pos);
        thrust::copy_n(counter.getShare(1)->begin() + old_mask_start_pos, leaf_counter_row_len, mask_raw.getShare(1)->begin() + new_mask_start_pos);
        thrust::copy_n(counter.getShare(0)->begin() + old_mask_start_pos, leaf_counter_row_len, mask_raw.getShare(0)->begin() + new_mask_start_pos + leaf_counter_row_len);
        thrust::copy_n(counter.getShare(1)->begin() + old_mask_start_pos, leaf_counter_row_len, mask_raw.getShare(1)->begin() + new_mask_start_pos + leaf_counter_row_len);
    }
    
    selectShare(mask_raw, mask_select, final_mask_flag, mask_updated);

    // 2. new_counter_zero: all values are zero except mask row
    //    new_counter_inherit: all values are herited from the parent
    Share<T> new_counter(new_num_LC * LEAF_COUNTER_SIZE);
    Share<T> new_counter_zero(new_counter.size());
    Share<T> new_counter_inherit(new_counter.size());
    new_counter_zero.zero();
    new_counter_inherit.zero();

    // 2.1) set new_counter_inherit
    int offset_sta1 = 0;
    for (int i = 0; i < old_num_LC; i++)
    {
        offset_sta = i * LEAF_COUNTER_SIZE;
        offset_sta1 = 2 * i * LEAF_COUNTER_SIZE;
        thrust::copy_n(counter.getShare(0)->begin() + offset_sta, LEAF_COUNTER_SIZE, new_counter_inherit.getShare(0)->begin() + offset_sta1);
        thrust::copy_n(counter.getShare(1)->begin() + offset_sta, LEAF_COUNTER_SIZE, new_counter_inherit.getShare(1)->begin() + offset_sta1);
        thrust::copy_n(counter.getShare(0)->begin() + offset_sta, LEAF_COUNTER_SIZE, new_counter_inherit.getShare(0)->begin() + offset_sta1 + LEAF_COUNTER_SIZE);
        thrust::copy_n(counter.getShare(1)->begin() + offset_sta, LEAF_COUNTER_SIZE, new_counter_inherit.getShare(1)->begin() + offset_sta1 + LEAF_COUNTER_SIZE);
    }

    // 2.2) set new_counter_zero (new_counter_inherit is same)
    // new_num_LC * leaf_counter_row_len || new_num_LC * LEAF_COUNTER_SIZE
    // have to modify lambada logic in copy_in_batch if we replace the below with copy_in_batch
    for (int i = 0; i < new_num_LC; i++)
    {
        offset_sta = i * LEAF_COUNTER_SIZE + leaf_counter_row_len;
        new_mask_start_pos = i * leaf_counter_row_len;
        
        thrust::copy_n(mask_updated.getShare(0)->begin() + new_mask_start_pos, leaf_counter_row_len, new_counter_zero.getShare(0)->begin() + offset_sta);
        thrust::copy_n(mask_updated.getShare(1)->begin() + new_mask_start_pos, leaf_counter_row_len, new_counter_zero.getShare(1)->begin() + offset_sta);
    }

    // 2.3) generate select share according to nodeFlag[]
    // rezie may disable the iterator 
    Share<T> node_flag(old_num_LC);
    thrust::copy_n(nodeFlag.getShare(0)->begin() + old_num_LC - 1, old_num_LC, node_flag.getShare(0)->begin());
    thrust::copy_n(nodeFlag.getShare(1)->begin() + old_num_LC - 1, old_num_LC, node_flag.getShare(1)->begin());

    Share<uint8_t> node_flag_EQ(old_num_LC);
    EQ(node_flag, node_flag_EQ);

    Share<uint8_t> node_flag_EQ_expand(new_counter.size());
    fill_in_batch(node_flag_EQ, node_flag_EQ_expand, 2 * LEAF_COUNTER_SIZE, 0);

    // 2.4) obliviously update all leaf counters
    // node_flag_EQ = 1 <=> node = 0 <=> internal node <=> reset 0 in leaf counter except the mask row
    selectShare(new_counter_inherit, new_counter_zero, node_flag_EQ_expand, new_counter);

    // 2.5) replace old_level's counter with next_level's counter 
    counter.resize(new_counter.size());
    counter.zero();
    counter += new_counter;

    // 3. update tree array
    //    update tree[] in both old_level and new_level
    // 3.1) old_level: for leaf (1) or dummy (2), tree[] always be 0.
    thrust::copy_n(split_decisions.getShare(0)->begin(), old_num_LC, tree.getShare(0)->begin() + old_num_LC - 1);
    thrust::copy_n(split_decisions.getShare(1)->begin(), old_num_LC, tree.getShare(1)->begin() + old_num_LC - 1);

    // 3.2) new_level: each tree[] stores the index [0, new_num_LC - 1]
    DeviceData<T> next_level_LCidx(new_num_LC);
    // DeviceData<T> next_level_LCidx(tree.size());
    thrust::sequence(next_level_LCidx.begin(), next_level_LCidx.end());

    // next_level_LCidx += (new_num_LC - 1); // we just need [0, new_num_LC - 1], not [0 + (new_num_LC - 1), new_num_LC - 1 + (new_num_LC - 1)]
    if (partyNum == 0) { 
        // PARTY_A
        thrust::transform(tree.getShare(0)->begin() + (new_num_LC - 1), tree.getShare(0)->begin() + (2 * new_num_LC - 1), next_level_LCidx.begin(), tree.getShare(0)->begin() + (new_num_LC - 1), thrust::plus<T>());
    } else if (partyNum == 2) {
        // PARTY_C
        thrust::transform(tree.getShare(1)->begin() + (new_num_LC - 1), tree.getShare(1)->begin() + (2 * new_num_LC - 1), next_level_LCidx.begin(), tree.getShare(1)->begin() + (new_num_LC - 1), thrust::plus<T>());
    }

    // 4. update tree level
    tree_level = new_tree_level;
    // std::cout << "tree_level after update = " << tree_level << std::endl;
 }

template<typename T, template<typename, typename...> typename Share>
void DecisionTree<T, Share>::inference_offline()
{
    int path_num = 1 << tree_level;
    M_T.resize(2 * ATTRIBUTE_COUNT_TOTAL * path_num);
    // 1. generate select bit corresponds to feature for each tree interal node
    // 1.1 get tree items except the last layer
    Share<T> tree_no_lastlayer(path_num - 1);
    thrust::copy_n(tree.getShare(0)->begin(), path_num - 1, tree_no_lastlayer.getShare(0)->begin());
    thrust::copy_n(tree.getShare(1)->begin(), path_num - 1, tree_no_lastlayer.getShare(1)->begin());
    // 1.2 fill tree items in ATTRIBUTE_COUNT_TOTAL times
    Share<T> tree_no_lastlayer_fill((path_num - 1) * ATTRIBUTE_COUNT_TOTAL);
    fill_in_batch(tree_no_lastlayer, tree_no_lastlayer_fill, ATTRIBUTE_COUNT_TOTAL, 0); 
    // 1.3 generate sequence of feature idx for each internal node
    DeviceData<T> fea_idx_seq(tree_no_lastlayer_fill.size());
    sequence_in_batch(fea_idx_seq, ATTRIBUTE_COUNT_TOTAL);
    // 1.4 use EQ to determine the selected feature idx in each node
    tree_no_lastlayer_fill -= fea_idx_seq;
    Share<uint8_t> EQ_selectedFea(tree_no_lastlayer_fill.size());
    EQ(tree_no_lastlayer_fill, EQ_selectedFea);

    // 2. generate path_cost vector for nodes in each layer
    // 2.1 select A shares for all nodes
    Share<T> ones(tree_no_lastlayer_fill.size());
    ones.fill(1);
    Share<T> zeros(tree_no_lastlayer_fill.size());
    zeros.fill(0);
    Share<T> sel_fea_Ashare(tree_no_lastlayer_fill.size());
    selectShare(zeros, ones, EQ_selectedFea, sel_fea_Ashare);
    // 2.2 copy double items, since path cost is 01 or 10, 00
    Share<T> sel_fea_Ashare_fill(tree_no_lastlayer_fill.size() * 2);
    fill_in_batch(sel_fea_Ashare, sel_fea_Ashare_fill, 2, 0);
    // 2.2.1 compute for left path cost of each node
    // initialize left path cost 01
    DeviceData<T> left_path_cost_plain((path_num - 1) * ATTRIBUTE_COUNT_TOTAL * 2);
    left_path_cost_plain.fill(1);
    negateAddDev(left_path_cost_plain, 2, 1); // To-be-done: negateAddDev has not been tested
    Share<T> left_path_cost(left_path_cost_plain.size());
    left_path_cost += sel_fea_Ashare_fill;
    left_path_cost *= left_path_cost_plain; // keep the cost in the target feature
    // 2.2.2 compute for right path cost of each node
    // initialize right path cost 10
    DeviceData<T> right_path_cost_plain(left_path_cost_plain.size());
    right_path_cost_plain.fill(1);
    negateAddDev(right_path_cost_plain, 2, 0);
    Share<T> right_path_cost(left_path_cost_plain.size());
    right_path_cost += sel_fea_Ashare_fill;
    right_path_cost *= right_path_cost_plain; // keep the cost in the target feature
    // 2.3 merge left and right path cost of each internal node
    Share<T> total_path_cost(left_path_cost_plain.size() * 2);
    merge_vecs(left_path_cost, right_path_cost, total_path_cost, ATTRIBUTE_COUNT_TOTAL * 2);
    // 2.3 in each layer, mul EQ and path_cost, copy t times for left/right for each layer
    int node_single_cost = ATTRIBUTE_COUNT_TOTAL * 2;
    int node_lr_costs = ATTRIBUTE_COUNT_TOTAL * 4;
    for (int i = 0; i < tree_level; i++)
    {
        int num_L = 1 << i;
        // t-th node in this layer has num_path_single_node left/right paths
        int num_path_single_node = 1 << (tree_level - 1 - i);
        // get the cost in current layer
        Share<T> current_path_cost(num_L * node_lr_costs);
        int cost_start_pos = (num_L - 1) * node_lr_costs; // previous layers' nodes
        thrust::copy_n(total_path_cost.getShare(0)->begin() + cost_start_pos, current_path_cost.size(), current_path_cost.getShare(0)->begin());
        thrust::copy_n(total_path_cost.getShare(1)->begin() + cost_start_pos, current_path_cost.size(), current_path_cost.getShare(1)->begin());
        // copy all cost into path_num costs
        Share<T> current_path_cost_fill(current_path_cost.size() * num_path_single_node);
        copy_in_batch(current_path_cost, current_path_cost_fill, node_single_cost, node_single_cost * num_path_single_node);
        // add the path cost of current layer to the tree matrix
        M_T += current_path_cost_fill;
    }
}

template<typename T, template<typename, typename...> typename Share>
void DecisionTree<T, Share>::inference_online(Share<T> &inf_data, Share<T> &result_label)
{
    // inf_data's dim is N_I * 2(d-1)
    int path_num = 1 << tree_level; // equal to leafClass size
    // 1. compute inf result matrix, N_I * path_num
    Share<T> M_R(INF_COUNT_TOTAL * path_num);
    matmul(inf_data, M_T, M_R, INF_COUNT_TOTAL, path_num, 2 * ATTRIBUTE_COUNT_TOTAL, false, false, false);
    // 2. transpose M_R
    Share<T> M_R_trans(path_num * INF_COUNT_TOTAL);
    gpu::transpose(M_R.getShare(0), M_R_trans.getShare(0), INF_COUNT_TOTAL, path_num);
    gpu::transpose(M_R.getShare(1), M_R_trans.getShare(1), INF_COUNT_TOTAL, path_num);
    cudaDeviceSynchronize();
    // 3. select the target label
    M_R_trans -= tree_level; // tree_level is the maximum cost of each path
    Share<uint8_t> EQ_path(M_R_trans.size());
    EQ(M_R_trans, EQ_path);
    Share<T> inf_res_pre(M_R_trans.size());
    Share<T> zeros(M_R_trans.size());
    zeros.fill(0);
    Share<T> leafClass_cp(M_R_trans.size());
    thrust::copy_n(thrust::make_permutation_iterator(leafClass.getShare(0)->begin(), thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1%leafClass.size())), leafClass_cp.size(), leafClass_cp.getShare(0)->begin());
    thrust::copy_n(thrust::make_permutation_iterator(leafClass.getShare(1)->begin(), thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1%leafClass.size())), leafClass_cp.size(), leafClass_cp.getShare(1)->begin());
    selectShare(zeros, leafClass_cp, EQ_path, inf_res_pre);
    Share<T> inf_res(INF_COUNT_TOTAL);
    auto my_flags_iterator = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1/path_num);
    thrust::reduce_by_key(my_flags_iterator, my_flags_iterator + inf_res_pre.size(), inf_res_pre.getShare(0)->begin(), thrust::make_discard_iterator(), inf_res.getShare(0)->begin());
    thrust::reduce_by_key(my_flags_iterator, my_flags_iterator + inf_res_pre.size(), inf_res_pre.getShare(1)->begin(), thrust::make_discard_iterator(), inf_res.getShare(1)->begin());
    
}

template class DecisionTree<uint32_t, RSS>;
template class DecisionTree<uint64_t, RSS>;