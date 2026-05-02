/*
 * RSS.h
 */

#pragma once

#include <cstddef>
#include <initializer_list>

#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include "../gpu/DeviceData.h"
#include "../gpu/StridedRange.cuh"
#include "../globals.h"

using namespace thrust::placeholders;
using namespace std;

template <typename T, typename I>
class RSSBase {

    protected:
        
        RSSBase(DeviceData<T, I> *a, DeviceData<T, I> *b);

    public:

        // default: PARTY_A = 0, PARTY_B = 1, PARTY_C = 2
        enum Party { PARTY_A, PARTY_B, PARTY_C };
        static const int numParties = 3;

        void set(DeviceData<T, I> *a, DeviceData<T, I> *b);
        size_t size() const;
        void zero();
        void fill(T val);
        void setPublic(std::vector<double> &v, bool convertToFixedPoint);
        DeviceData<T, I> *getShare(int i);
        const DeviceData<T, I> *getShare(int i) const;
        static int numShares();
        static int nextParty(int party);
        static int prevParty(int party);
        typedef T share_type;
        //using share_type = T;
        typedef I iterator_type;


        RSSBase<T, I> &operator+=(const T rhs);
        RSSBase<T, I> &operator-=(const T rhs);
        RSSBase<T, I> &operator*=(const T rhs);
        RSSBase<T, I> &operator>>=(const T rhs);

        template<typename I2>
        RSSBase<T, I> &operator+=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator-=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator*=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator^=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator&=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator>>=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator<<=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator+=(const RSSBase<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator-=(const RSSBase<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator*=(const RSSBase<T, I2> &rhs);
        // template<typename I2>
        // RSSBase<T, I> &operator/=(const RSSBase<T, I2> &rhs); // new func mul with delayed reshare
        template<typename I2>
        RSSBase<T, I> &operator^=(const RSSBase<T, I2> &rhs);
        template<typename I2>
        RSSBase<T, I> &operator|=(const RSSBase<T, I2> &rhs); // new func OR
        template<typename I2>
        RSSBase<T, I> &operator&=(const RSSBase<T, I2> &rhs);

    protected:
        
        DeviceData<T, I> *shareA;
        DeviceData<T, I> *shareB;
};

template<typename T, typename I = BufferIterator<T> >
class RSS : public RSSBase<T, I> {

    public:

        RSS(DeviceData<T, I> *a, DeviceData<T, I> *b);
};

template<typename T>
class RSS<T, BufferIterator<T> > : public RSSBase<T, BufferIterator<T> > {

    public:

        RSS(DeviceData<T> *a, DeviceData<T> *b);
        RSS(size_t n);
        RSS(std::initializer_list<double> il, bool convertToFixedPoint = true);

        void resize(size_t n);

    private:

        DeviceData<T> _shareA;
        DeviceData<T> _shareB;
};

// Functionality

template<typename T, typename I>
void dividePublic(RSS<T, I> &a, T denominator);

template<typename T, typename I, typename I2>
void dividePublic(RSS<T, I> &a, DeviceData<T, I2> &denominators);

template<typename T, typename I, typename I2>
void reconstruct(RSS<T, I> &in, DeviceData<T, I2> &out);

template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const RSS<T, I> &x, const RSS<T, I2> &y, const RSS<U, I3> &b, RSS<T, I4> &z);

template<typename T, typename I, typename I2>
void sqrt(const RSS<T, I> &in, RSS<T, I2> &out);

template<typename T, typename I, typename I2>
void inverse(const RSS<T, I> &in, RSS<T, I2> &out);

template<typename T, typename I, typename I2>
void sigmoid(const RSS<T, I> &in, RSS<T, I2> &out);

template<typename T>
void matmul(const RSS<T> &a, const RSS<T> &b, RSS<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c);

// TODO change into 2 arguments with subtraction, pointer NULL indicates compare w/ 0
template<typename T, typename U, typename I, typename I2>
void dReLU(const RSS<T, I> &input, RSS<U, I2> &result);
    
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const RSS<T, I> &input, RSS<T, I2> &result, RSS<U, I3> &dresult);

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(const RSS<T, I> &input, RSS<T, I2> &result, RSS<U, I3> &dresult, int k);

// new MPC protocols
template<typename T, typename U, typename I, typename I2>
void EQ(const RSS<T, I> &input, RSS<U, I2> &result);

template<typename T, typename I, typename I2, typename I3>
void AccMulWithDelayShare(RSS<T, I> &v1, RSS<T, I2> &v2, RSS<T, I3> &result, int len1, int len2, int len3);

template<typename T, typename I, typename I2, typename I3>
void OAA_MultipleArray(RSS<T, I> &inst_reached_idx, RSS<T, I2> &arr, int subArrLen, RSS<T, I3> &result);

template<typename T, typename I, typename I2, typename I3>
void OAA(RSS<T, I> &inst_reached_idx, RSS<T, I2> &arr, RSS<T, I3> &result);

template<typename T, typename I>
void negateAdd(RSS<T, I> &input, int k);

template<typename T, typename I>
void negateAddDev(DeviceData<T, I> &input, int k, int EvenOrOdd);

// thrust batch operation functors with fancy iterators
template<typename T, typename I, typename I2>
void fill_in_batch(RSS<T, I> &arr1, RSS<T, I2> &arr2, int len1, int arr1StaIdx);

template<typename T, typename I, typename I2>
void fill_in_batch_withIdx(RSS<T, I> &arr1, RSS<T, I2> &arr2, int len1, int staIdx, int batch);

template<typename T, typename I, typename I2>
void copy_in_batch(RSS<T, I> &arr1, RSS<T, I2> &arr2, int len1, int len2);

template<typename T, typename I, typename I2>
void replace_in_batch(RSS<T, I> &arr1, RSS<T, I2> &arr2, int batch);

template<typename T, typename I, typename I2, typename I3>
void merge_vecs(RSS<T, I> &arr1, RSS<T, I2> &arr2, RSS<T, I3> &arr3, int batch);

template<typename T, typename I>
void sequence_in_batch(DeviceData<T, I> &arr1, int len1);

template<typename T, typename I>
void mult_in_batch_withIdx(RSS<T, I> &arr2, int len1, int staIdx, int batch, int mulVal);

template<typename T, typename I>
void sum_in_batch_withIdx(RSS<T, I> &arr2, int len1, int staIdx, int batch, int sumVal, int partyNum);

template<typename T, typename I, typename I2>
void reduce_in_batch(RSS<T, I> &arr1, RSS<T, I2> &arr2, int len1, int len2);

#include "RSS.inl"

