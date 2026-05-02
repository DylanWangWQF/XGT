/*
 * RSS.inl
 */

#pragma once

#include "RSS.h"

#include "../gpu/bitwise.cuh"
#include "../gpu/convolution.cuh"
#include "../gpu/DeviceData.h"
#include "../gpu/functors.cuh"
#include "../gpu/matrix.cuh"
#include "../gpu/gemm.cuh"
#include "../gpu/StridedRange.cuh"
#include "../globals.h"
#include "Precompute.h"
#include "../util/functors.h"
#include "../util/Profiler.h"

extern Precompute PrecomputeObject;
extern Profiler comm_profiler;
extern Profiler func_profiler;

extern nlohmann::json gst_config;

using namespace thrust::placeholders;
// Functors

struct rss_convex_comb_functor {
    const int party;
    rss_convex_comb_functor(int _party) : party(_party) {}
    
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        // b, c share A, c share B, d share A, d share B
        if (thrust::get<0>(t) == 1) {
            switch(party) {
                case RSS<uint64_t>::PARTY_A: // doesn't really matter what type RSS is templated at here
                    thrust::get<3>(t) = 1 - thrust::get<1>(t);
                    thrust::get<4>(t) = -thrust::get<2>(t);
                    break;
                case RSS<uint64_t>::PARTY_B:
                    thrust::get<3>(t) = -thrust::get<1>(t);
                    thrust::get<4>(t) = -thrust::get<2>(t);
                    break;
                case RSS<uint64_t>::PARTY_C:
                    thrust::get<3>(t) = -thrust::get<1>(t);
                    thrust::get<4>(t) = 1 - thrust::get<2>(t);
                    break;
            }
        } else {
            thrust::get<3>(t) = thrust::get<1>(t);
            thrust::get<4>(t) = thrust::get<2>(t);
        }
    }
};

// Prototypes

template<typename T, typename I, typename I2>
void reconstruct3out3(DeviceData<T, I> &input, DeviceData<T, I2> &reconst);

template<typename T, typename I, typename I2>
void reshare(DeviceData<T, I> &c, RSSBase<T, I2> &out);

template<typename T, typename I, typename I2>
void dividePublic3out3(DeviceData<T, I> &a, T denominator, RSSBase<T, I2> &out);

template<typename T>
void localMatMul(const RSS<T> &a, const RSS<T> &b, DeviceData<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c);

template<typename T, typename I, typename I2>
void carryOut(RSS<T, I> &p, RSS<T, I> &g, int k, RSS<T, I2> &out);

template<typename T, typename I, typename I2>
void carryOut_EQ(RSS<T, I> &p, int k, RSS<T, I2> &out);

template<typename T, typename I, typename I2>
void getPowers(const RSS<T, I> &in, DeviceData<T, I2> &pow);

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(const RSS<T, I> &in, RSS<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn);

template<typename T, typename U, typename I, typename I2>
void convex_comb(RSS<T, I> &a, RSS<T, I> &c, DeviceData<U, I2> &b);

// RSS class implementation 

template<typename T, typename I>
RSSBase<T, I>::RSSBase(DeviceData<T, I> *a, DeviceData<T, I> *b) : 
                shareA(a), shareB(b) {}

template<typename T, typename I>
void RSSBase<T, I>::set(DeviceData<T, I> *a, DeviceData<T, I> *b) {
    shareA = a;
    shareB = b; 
}

template<typename T, typename I>
size_t RSSBase<T, I>::size() const {
    return shareA->size();
}

template<typename T, typename I>
void RSSBase<T, I>::zero() {
    shareA->zero();
    shareB->zero();
};

template<typename T, typename I>
void RSSBase<T, I>::fill(T val) {
    shareA->fill(partyNum == PARTY_A ? val : 0);
    shareB->fill(partyNum == PARTY_C ? val : 0);
}

template<typename T, typename I>
void RSSBase<T, I>::setPublic(std::vector<double> &v, bool convertToFixedPoint) {
    std::vector<T> shifted_vals;
    for (double f : v) {
        if (convertToFixedPoint) {
            shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
        } else {
            shifted_vals.push_back((T) f);
        }
    }

    switch (partyNum) {
        case PARTY_A:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), shareA->begin());
            shareB->zero();
            break;
        case PARTY_B:
            shareA->zero();
            shareB->zero();
            break;
        case PARTY_C:
            shareA->zero();
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), shareB->begin());
            break;
    }
};

template<typename T, typename I>
DeviceData<T, I> *RSSBase<T, I>::getShare(int i) {
    switch (i) {
        case 0:
            return shareA;
        case 1:
            return shareB;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const DeviceData<T, I> *RSSBase<T, I>::getShare(int i) const {
    switch (i) {
        case 0:
            return shareA;
        case 1:
            return shareB;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
RSSBase<T, I> &RSSBase<T, I>::operator+=(const T rhs) {
    if (partyNum == PARTY_A) {
        *shareA += rhs;
    } else if (partyNum == PARTY_C) {
        *shareB += rhs;
    }
    return *this;
}

template<typename T, typename I>
RSSBase<T, I> &RSSBase<T, I>::operator-=(const T rhs) {
    if (partyNum == PARTY_A) {
        *shareA -= rhs;
    } else if (partyNum == PARTY_C) {
        *shareB -= rhs;
    }
    return *this;
}

template<typename T, typename I>
RSSBase<T, I> &RSSBase<T, I>::operator*=(const T rhs) {
    *shareA *= rhs;
    *shareB *= rhs;
    return *this;
}

template<typename T, typename I>
RSSBase<T, I> &RSSBase<T, I>::operator>>=(const T rhs) {
    *shareA >>= rhs;
    *shareB >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator+=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_A) {
        *shareA += rhs;
    } else if (partyNum == PARTY_C) {
        *shareB += rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator-=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_A) {
        *shareA -= rhs;
    } else if (partyNum == PARTY_C) {
        *shareB -= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator*=(const DeviceData<T, I2> &rhs) {
    *shareA *= rhs;
    *shareB *= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator^=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_A) {
        *shareA ^= rhs;
    } else if (partyNum == PARTY_C) {
        *shareB ^= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator&=(const DeviceData<T, I2> &rhs) {
    *shareA &= rhs;
    *shareB &= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator>>=(const DeviceData<T, I2> &rhs) {
    *shareA >>= rhs;
    *shareB >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator<<=(const DeviceData<T, I2> &rhs) {
    *shareA <<= rhs;
    *shareB <<= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator+=(const RSSBase<T, I2> &rhs) {
    *shareA += *rhs.getShare(0);
    *shareB += *rhs.getShare(1);
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator-=(const RSSBase<T, I2> &rhs) {
    *shareA -= *rhs.getShare(0);
    *shareB -= *rhs.getShare(1);
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator*=(const RSSBase<T, I2> &rhs) {

    if (gst_config["debug_overflow"]) {
        std::vector<double> lhs_revealed(this->size());
        copyToHost(*static_cast<RSS<T, I> *>(this), lhs_revealed);

        std::vector<double> rhs_revealed(rhs.size());
        copyToHost(
            //*static_cast<RSS<T, I2> *>((const_cast<RSSBase<T, I2> *>(&rhs)),
            *((RSS<T, I2> *) const_cast<RSSBase<T, I2> *>(&rhs)),
            rhs_revealed
        );

        for (int i = 0; i < lhs_revealed.size(); i++) {
            double temp_result = lhs_revealed[i] * rhs_revealed[i];
#if 0 // temporarily disable debug check because on some PRECISIONS it generates a compile error
            if (temp_result >= (1 << (64 - (2 * FLOAT_PRECISION)))) {
                printf("WARNING: overflow in RSS multiplication for precision %d: %f * %f -> %f\n", FLOAT_PRECISION, lhs_revealed[i], rhs_revealed[i], temp_result);
            }
#endif
            //assert(temp_result < (1 << (64 - (2 * FLOAT_PRECISION))) && "overflow in RSS multiplication");
        }
    }

    DeviceData<T> summed(rhs.size());
    summed.zero();
    summed += *rhs.getShare(0);
    summed += *rhs.getShare(1);

    *shareA *= summed;
    *shareB *= *rhs.getShare(0);
    *shareA += *shareB;

    reshare(*shareA, *this);
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator^=(const RSSBase<T, I2> &rhs) {
    *shareA ^= *rhs.getShare(0);
    *shareB ^= *rhs.getShare(1);
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator|=(const RSSBase<T, I2> &rhs) {
    *shareA |= *rhs.getShare(0);
    *shareB |= *rhs.getShare(1);
    return *this;
}

template<typename T, typename I>
template<typename I2>
RSSBase<T, I> &RSSBase<T, I>::operator&=(const RSSBase<T, I2> &rhs) {

    DeviceData<T> summed(rhs.size());
    summed.zero();
    summed ^= *rhs.getShare(0);
    summed ^= *rhs.getShare(1);

    *shareA &= summed;
    *shareB &= *rhs.getShare(0);
    *shareA ^= *shareB;

    reshare(*shareA, *this);
    return *this;
}

template<typename T, typename I>
int RSSBase<T, I>::nextParty(int party) {
    switch(party) {
        case PARTY_A:
            return PARTY_B;
        case PARTY_B:
            return PARTY_C;
        default: // PARTY_C 
            return PARTY_A;
    }   
}

template<typename T, typename I>
int RSSBase<T, I>::prevParty(int party) {
    switch(party) {
        case PARTY_A:
            return PARTY_C;
        case PARTY_B:
            return PARTY_A;
        default: // PARTY_C
            return PARTY_B;
    }   
}

template<typename T, typename I>
int RSSBase<T, I>::numShares() {
    return 2;
}

template<typename T, typename I>
RSS<T, I>::RSS(DeviceData<T, I> *a, DeviceData<T, I> *b) : RSSBase<T, I>(a, b) {}

template<typename T>
RSS<T, BufferIterator<T> >::RSS(DeviceData<T> *a, DeviceData<T> *b) :
    RSSBase<T, BufferIterator<T> >(a, b) {}
template<typename T>
RSS<T, BufferIterator<T> >::RSS(size_t n) :
    _shareA(n),
    _shareB(n),
    RSSBase<T, BufferIterator<T> >(&_shareA, &_shareB) {}

template<typename T>
RSS<T, BufferIterator<T> >::RSS(std::initializer_list<double> il, bool convertToFixedPoint) :
    _shareA(il.size()),
    _shareB(il.size()),
    RSSBase<T, BufferIterator<T> >(&_shareA, &_shareB) {

    std::vector<T> shifted_vals;
    for (double f : il) {
        if (convertToFixedPoint) {
            shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
        } else {
            shifted_vals.push_back((T) f);
        }
    }

    switch (partyNum) {
        case RSS<T>::PARTY_A:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), _shareA.begin());
            break;
        case RSS<T>::PARTY_B:
            // nothing
            break;
        case RSS<T>::PARTY_C:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), _shareB.begin());
            break;
    }
}

template<typename T>
void RSS<T, BufferIterator<T> >::resize(size_t n) {
    _shareA.resize(n);
    _shareB.resize(n); 
}

template<typename T, typename I>
void dividePublic(RSS<T, I> &a, T denominator) {

    RSS<T> r(a.size()), rPrime(a.size());
    PrecomputeObject.getDividedShares<T, RSS<T> >(r, rPrime, denominator, a.size()); 
    a -= rPrime;
    
    DeviceData<T> reconstructed(a.size());
    reconstruct(a, reconstructed);
    reconstructed /= denominator;

    a.zero();
    a += r;
    a += reconstructed;
}

template<typename T, typename I, typename I2>
void dividePublic(RSS<T, I> &a, DeviceData<T, I2> &denominators) {

    assert(denominators.size() == a.size() && "RSS dividePublic powers size mismatch");

    RSS<T> r(a.size()), rPrime(a.size());
    PrecomputeObject.getDividedShares<T, I2, RSS<T> >(r, rPrime, denominators, a.size()); 

    a -= rPrime;

    DeviceData<T> reconstructed(a.size());
    reconstruct(a, reconstructed);
    reconstructed /= denominators;

    a.zero();
    a += r;
    a += reconstructed;
}

template<typename T, typename I, typename I2>
void reconstruct(RSS<T, I> &in, DeviceData<T, I2> &out) {

    // 1 - send shareA to next party
    comm_profiler.start();
    in.getShare(0)->transmit(RSS<T>::nextParty(partyNum));

    // 2 - receive shareA from previous party into DeviceBuffer 
    DeviceData<T> rxShare(in.size());
    rxShare.receive(RSS<T>::prevParty(partyNum));

    in.getShare(0)->join();
    rxShare.join();
    comm_profiler.accumulate("comm-time");

    // 3 - result is our shareB + received shareA
    out.zero();
    out += *in.getShare(0);
    out += *in.getShare(1);
    out += rxShare;

    func_profiler.add_comm_round();
}

template<typename T>
void localMatMul(const RSS<T> &a, const RSS<T> &b, DeviceData<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c) {

    DeviceData<T> acc(c.size());
    gpu::gemm(M, N, K, a.getShare(0), transpose_a, b.getShare(0), transpose_b, &acc, transpose_c);
    c += acc;
    acc.zero();
    gpu::gemm(M, N, K, a.getShare(0), transpose_a, b.getShare(1), transpose_b, &acc, transpose_c);
    c += acc;
    acc.zero();
    gpu::gemm(M, N, K, a.getShare(1), transpose_a, b.getShare(0), transpose_b, &acc, transpose_c);
    c += acc;
}

template<typename T>
void matmul(const RSS<T> &a, const RSS<T> &b, RSS<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c) {

    DeviceData<T> rawResult(M * N);
    localMatMul(a, b, rawResult, M, N, K, transpose_a, transpose_b, transpose_c);

    // printDeviceData(rawResult, "rawResult of matrix mul", false);
    c += rawResult;
    // reshare(rawResult, c);
    // dividePublic3out3(rawResult, (T)1 << truncation, c);
}


template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const RSS<T, I> &x, const RSS<T, I2> &y, const RSS<U, I3> &b, RSS<T, I4> &z) {

    assert(x.size() == y.size() && x.size() == b.size() && x.size() == z.size() && "RSS selectShare input size mismatch");

    RSS<T> c(x.size());
    RSS<U> cbits(b.size());

    // b XOR c, then open -> e
    cbits ^= b;

    DeviceData<U> e(cbits.size());
    reconstruct(cbits, e);

    // d = 1-c if e=1 else d = c       ->        d = (e)(1-c) + (1-e)(c)
    RSS<T> d(e.size());
    convex_comb(d, c, e);

    // z = ((y - x) * d) + x
    RSS<T> result(x.size());
    result += y;
    result -= x;
    result *= d;
    result += x;

    z.zero();
    z += result;
}

template<typename T, typename I, typename I2>
void sqrt(const RSS<T, I> &in, RSS<T, I2> &out) {
    /*
     * Approximations:
     *   > sqrt(x) = 0.424 + 0.584(x)
     *     sqrt(x) = 0.316 + 0.885(x) - 0.202(x^2)
     */

    if (gst_config["debug_sqrt"]) {
        std::vector<double> revealedInput(in.size());
        copyToHost(
            *const_cast<RSS<T, I> *>(&in),
            revealedInput 
        );

        for (int i = 0; i < revealedInput.size(); i++) {
            assert(revealedInput[i] >= 0 && "sqrt got a negative value");
        }
    }

    taylorSeries(in, out, 0.424, 0.584, 0.0, sqrt_lambda());
}

template<typename T, typename I, typename I2>
void inverse(const RSS<T, I> &in, RSS<T, I2> &out) {
    /*
     * Approximations:
     *     1/x = 2.838 - 1.935(x)
     *   > 1/x = 4.245 - 5.857(x) + 2.630(x^2)
     */
    taylorSeries(in, out, 4.245, -5.857, 2.630, inv_lambda());
}

template<typename T, typename I, typename I2>
void sigmoid(const RSS<T, I> &in, RSS<T, I2> &out) {
    /*
     * Approximation:
     *   > sigmoid(x) = 0.494286 + 0.275589(x) + -0.038751(x^2)
     */
    taylorSeries(in, out, 0.494286, 0.275589, -0.038751, sigmoid_lambda());
}


template<typename T, typename I, typename I2, typename I3>
void AccMulWithDelayShare(RSS<T, I> &v1, RSS<T, I2> &v2, RSS<T, I3> &result, int len1, int len2, int len3){
    DeviceData<T> summed(v1.size());
    summed.zero();
    summed += *v2.getShare(0);
    summed += *v2.getShare(1);

    *v1.getShare(0) *= summed;
    *v1.getShare(1) *= *v2.getShare(0);
    *v1.getShare(0) += *v1.getShare(1); // shareA/v1.getShare(0) stores the 3out3 result

    // ------------reduce the sum in v1--------------
    // v1.size() = len1 * len2 * len3
    // result.size() = len1 * len3
    DeviceData<T> reduce_result(len1 * len3);

    // only need to process shareA 
    const auto A_ptr = v1.getShare(0)->raw().data();
    const auto B_ptr = reduce_result.raw().data();

    thrust::for_each_n(
        thrust::make_counting_iterator(0), len1 * len3,
        [len1, len2, A_ptr, B_ptr]
        __host__ __device__ (int idx) {
            const int outer_batch_id = idx / len1;
            const int batch_el = idx % len1;
            const int begin = outer_batch_id * len1 * len2 + batch_el;
            const int end = begin + len1 * len2;
            int sum = 0;
            for (int i = begin; i < end; i += len1) {
                sum += A_ptr[i];
            }
            B_ptr[idx] = sum;
        });
    
    reshare(reduce_result, result);

    // when len1 (size of data samples in GST) is large, reduce_by_key is slower than for_each
    /*
    auto batch_reduction_in_0 =
        thrust::make_permutation_iterator(
            v1.getShare(0)->begin(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [len1, len2] __device__ (int idx) { // "transpose"
                    const int batch_id = idx / len2;
                    const int batch_el = idx % len2;
                    return batch_el * len1 + batch_id;
                }));

    thrust::reduce_by_key(thrust::make_counting_iterator(0),
                          thrust::make_counting_iterator(len1 * len2),
                          batch_reduction_in_0,
                          thrust::make_discard_iterator(),
                          reduce_result.begin(),
                          [len2] __device__ (int idx, int idy) {
                              return idx / len2 == idy / len2;
                           });
    */
   
   // this can be used for only one leaf node. for multiple nodes, we need more for loop
   /*
   thrust::for_each_n(thrust::make_counting_iterator(0), len1,
                       [len1,
                        len2,
                        A = v1.getShare(0)->raw().data(),
                        B = reduce_result.raw().data()]
                       __device__ (int idx) {
                           int sum = 0;
                           // "grid-stride loop"
                           for (int i = idx; i < len1 * len2; i += len1) {
                               sum += A[i];
                           }
                           B[idx] = sum;
                       });
    */

}

template<typename T, typename U, typename I, typename I2>
void EQ(const RSS<T, I> &input, RSS<U, I2> &result) {

    int bitWidth = sizeof(T) * 8;

    RSS<T> r(input.size());
    RSS<U> rbits(input.size() * bitWidth);
    
    rbits.fill(0);

    DeviceData<T> a(input.size());
    r += input;
    reconstruct(r, a);

    DeviceData<U> abits(rbits.size());
    gpu::bitexpand(&a, &abits); 

    RSS<U> p(rbits.size());
    p.zero(); 
    p += rbits;
    p ^= abits; 

    RSS<U> zbits(rbits.size());
    zbits.fill(1);
    p ^= zbits;

    RSS<U> preResult(result.size()); 
    carryOut_EQ(p, bitWidth, preResult);

    result += preResult;
}

template<typename T, typename I>
void negateAdd(RSS<T, I> &input, int k)
{
    using SRIterator = typename StridedRange<I>::iterator;
    StridedRange<I> cEven0Range(input.getShare(0)->begin(), input.getShare(0)->end(), k);
    DeviceData<T, SRIterator> cEven0(cEven0Range.begin(), cEven0Range.end());
    StridedRange<I> cEven1Range(input.getShare(1)->begin(), input.getShare(1)->end(), k);
    DeviceData<T, SRIterator> cEven1(cEven1Range.begin(), cEven1Range.end());
    RSS<T, SRIterator> cEven(&cEven0, &cEven1);
    cEven *= (T) -1;
    cEven += 1;

    // copy output to original counter
    thrust::copy(cEven.getShare(0)->begin(), cEven.getShare(0)->end(), cEven0Range.begin());
    thrust::copy(cEven.getShare(1)->begin(), cEven.getShare(1)->end(), cEven1Range.begin());
}

template<typename T, typename I>
void negateAddDev(DeviceData<T, I> &input, int k, int EvenOrOdd)
{
    using SRIterator = typename StridedRange<I>::iterator;
    if (EvenOrOdd == 1)
    {
        StridedRange<I> cEvenRange(input.begin(), input.end(), k);
        DeviceData<T, SRIterator> cEven(cEvenRange.begin(), cEvenRange.end());
        cEven *= (T) -1;
        cEven += 1;
        // copy output to original counter
        thrust::copy(cEven.begin(), cEven.end(), cEvenRange.begin());
    }
    else
    {
        int offset = 1;
        StridedRange<I> cOddRange(input.begin() + offset, input.end(), k);
        DeviceData<T, SRIterator> cOdd(cOddRange.begin(), cOddRange.end());
        cOdd *= (T) -1;
        cOdd += 1;
        // copy output to original counter
        thrust::copy(cOdd.begin(), cOdd.end(), cOddRange.begin());
    }
}

// thrust functors with fancy iterators
template<typename T, typename I, typename I2>
void fill_in_batch(RSS<T, I> &arr1, RSS<T, I2> &arr2, int len1, int arr1StaIdx)
{
    // arr1.size() = len2
    // arr2.size() = len1 * len2
    auto batched_fill_in_0 = 
        thrust::make_permutation_iterator(
            arr1.getShare(0)->begin() + arr1StaIdx,
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [len_batch = len1]
                __device__ (int idx) {
                    return idx / len_batch; // corresponds to i
                }));
    auto batched_fill_in_1 = 
        thrust::make_permutation_iterator(
            arr1.getShare(1)->begin() + arr1StaIdx,
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [len_batch = len1]
                __device__ (int idx) {
                    return idx / len_batch; // corresponds to i
                }));
    auto is_valid =
        [len1]
        __device__ (int idx) {
            return true;
        };
    // note: original thrust::identity<int>{} -> thrust::identity<T>{}
    thrust::transform_if(batched_fill_in_0, batched_fill_in_0 + arr2.size(),
                         thrust::make_counting_iterator(0),
                         arr2.getShare(0)->begin(),
                         thrust::identity<T>{},
                         is_valid);
    thrust::transform_if(batched_fill_in_1, batched_fill_in_1 + arr2.size(),
                         thrust::make_counting_iterator(0),
                         arr2.getShare(1)->begin(),
                         thrust::identity<T>{},
                         is_valid);
}

template<typename T, typename I, typename I2>
void fill_in_batch_withIdx(RSS<T, I> &arr1, RSS<T, I2> &arr2, int len1, int staIdx, int batch)
{
    // arr1.size() = len2
    // arr2.size() = len1 * len2
    // staIdx means from specified index for each len1
    // batch indicates how many values are filled for each len1
    auto batched_fill_in_0 = 
        thrust::make_permutation_iterator(
            arr1.getShare(0)->begin(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [staIdx, len_padded = len1]
                __device__ (int idx) {
                    return (idx - staIdx) / len_padded; // corresponds to i
                }));

    auto batched_fill_in_1 = 
        thrust::make_permutation_iterator(
            arr1.getShare(1)->begin(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [staIdx, len_padded = len1]
                __device__ (int idx) {
                    return (idx - staIdx) / len_padded; // corresponds to i
                }));
    
    auto is_valid =
        [staIdx, batch, len_padded = len1]
        __device__ (int idx) {
            return (idx >= staIdx) && ((idx - staIdx) % len_padded < batch);
        };
    
    thrust::transform_if(batched_fill_in_0, batched_fill_in_0 + arr2.size(),
                         thrust::make_counting_iterator(0),
                         arr2.getShare(0)->begin(),
                         thrust::identity<T>{},
                         is_valid);
    thrust::transform_if(batched_fill_in_1, batched_fill_in_1 + arr2.size(),
                         thrust::make_counting_iterator(0),
                         arr2.getShare(1)->begin(),
                         thrust::identity<T>{},
                         is_valid);
}

template<typename T, typename I>
void sequence_in_batch(DeviceData<T, I> &arr1, int len1)
{
    auto batched_sequence_in = 
        thrust::make_transform_iterator(
            thrust::make_counting_iterator(0),
            [len1] __device__ (const int idx) { return idx % len1; });
    thrust::copy_n(batched_sequence_in, arr1.size(), arr1.begin());
}

template<typename T, typename I, typename I2>
void copy_in_batch(RSS<T, I> &arr1, RSS<T, I2> &arr2, int len1, int len2)
{
    // arr1.size() = len1 * len3
    // arr2.size() = len2 * len3
    // len2 % len1 = 0
    auto batched_repeat_copy_in_0 = 
        thrust::make_permutation_iterator(
            arr1.getShare(0)->begin(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [len1, len2] __device__ (const int idx) {
                    const int batch_id = idx / len2;
                    const int batch_el = idx % len1;
                    return batch_id * len1 + batch_el;
                }));
    auto batched_repeat_copy_in_1 = 
        thrust::make_permutation_iterator(
            arr1.getShare(1)->begin(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [len1, len2] __device__ (const int idx) {
                    const int batch_id = idx / len2;
                    const int batch_el = idx % len1;
                    return batch_id * len1 + batch_el;
                }));

    thrust::copy_n(batched_repeat_copy_in_0, arr2.size(), arr2.getShare(0)->begin());
    thrust::copy_n(batched_repeat_copy_in_1, arr2.size(), arr2.getShare(1)->begin());
}

template<typename T, typename I, typename I2>
void replace_in_batch(RSS<T, I> &arr1, RSS<T, I2> &arr2, int batch)
{
    // arr1.size is twice of arr2.size. Use arr2 to replace items in odd batches.
    assert(arr1.size() == 2 * arr2.size());

    const auto scatter_indices_it_0 = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [batch]
        __host__ __device__ (int idx) {
            const int batch_id = idx / batch;
            const int elem_id = idx % batch;
            return (batch_id * 2 + 1) * batch + elem_id;
        });
    const auto scatter_indices_it_1 = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [batch]
        __host__ __device__ (int idx) {
            const int batch_id = idx / batch;
            const int elem_id = idx % batch;
            return (batch_id * 2 + 1) * batch + elem_id;
        });

    thrust::scatter(
        arr2.getShare(0)->begin(), arr2.getShare(0)->end(),
        scatter_indices_it_0,
        arr1.getShare(0)->begin());
    thrust::scatter(
        arr2.getShare(1)->begin(), arr2.getShare(1)->end(),
        scatter_indices_it_1,
        arr1.getShare(1)->begin());
}

template<typename T, typename I, typename I2, typename I3>
void merge_vecs(RSS<T, I> &arr1, RSS<T, I2> &arr2, RSS<T, I3> &arr3, int batch)
{
    assert(arr1.size() == arr2.size());
    const auto out_size = arr1.size() + arr2.size();

    const auto arr1_ptr0 = arr1.getShare(0)->raw().data();
    const auto arr2_ptr0 = arr2.getShare(0)->raw().data();
    const auto arr1_ptr1 = arr1.getShare(1)->raw().data();
    const auto arr2_ptr1 = arr2.getShare(1)->raw().data();

    auto batch_interleave_it_0 = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [batch, arr1_ptr0, arr2_ptr0]
        __host__ __device__ (int idx) -> T {
            const int batch_id = idx / batch;
            const int batch_el = idx % batch;
            const int in_idx = (batch_id / 2) * batch + batch_el;
            const bool even_batch = (batch_id % 2 == 0);
            return even_batch ? arr1_ptr0[in_idx] : arr2_ptr0[in_idx];
        });

    auto batch_interleave_it_1 = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [batch, arr1_ptr1, arr2_ptr1]
        __host__ __device__ (int idx) -> T {
            const int batch_id = idx / batch;
            const int batch_el = idx % batch;
            const int in_idx = (batch_id / 2) * batch + batch_el;
            const bool even_batch = (batch_id % 2 == 0);
            return even_batch ? arr1_ptr1[in_idx] : arr2_ptr1[in_idx];
        });

    
    thrust::copy_n(batch_interleave_it_0, arr3.size(), arr3.getShare(0)->begin());
    thrust::copy_n(batch_interleave_it_1, arr3.size(), arr3.getShare(1)->begin());
}

template<typename T, typename I>
void mult_in_batch_withIdx(RSS<T, I> &arr2, int len1, int staIdx, int batch, int mulVal)
{
    // mulVal is multiplied value
    auto is_valid =
        [staIdx, batch, len1_padded = len1]
        __device__ (int idx) {
            return (idx >= staIdx) && ((idx - staIdx) % len1_padded < batch);
        };
    thrust::transform_if(arr2.getShare(0)->begin(), arr2.getShare(0)->end(),
                         thrust::make_counting_iterator(0),
                         arr2.getShare(0)->begin(),
                         scalar_mult_functor<T>(mulVal),
                         is_valid);
    thrust::transform_if(arr2.getShare(1)->begin(), arr2.getShare(1)->end(),
                         thrust::make_counting_iterator(0),
                         arr2.getShare(1)->begin(),
                         scalar_mult_functor<T>(mulVal),
                         is_valid);
}

template<typename T, typename I>
void sum_in_batch_withIdx(RSS<T, I> &arr2, int len1, int staIdx, int batch, int sumVal, int partyNum)
{
    auto is_valid =
        [staIdx, batch, len1_padded = len1]
        __device__ (int idx) {
            return (idx >= staIdx) && ((idx - staIdx) % len1_padded < batch);
        };
    if (partyNum == 0) { 
        // PARTY_A
        thrust::transform_if(arr2.getShare(0)->begin(), arr2.getShare(0)->end(),
                         thrust::make_counting_iterator(0),
                         arr2.getShare(0)->begin(),
                         scalar_plus_functor<T>(sumVal),
                         is_valid);
    } else if (partyNum == 2) {
        // PARTY_C
        thrust::transform_if(arr2.getShare(1)->begin(), arr2.getShare(1)->end(),
                         thrust::make_counting_iterator(0),
                         arr2.getShare(1)->begin(),
                         scalar_plus_functor<T>(sumVal),
                         is_valid);
    }
}

template<typename T, typename I, typename I2>
void reduce_in_batch(RSS<T, I> &arr1, RSS<T, I2> &arr2, int len1, int len2)
{
    // arr1.size() = len1 * len2
    // arr2.size() = len1
    auto batch_reduction_in_0 =
        thrust::make_permutation_iterator(
            arr1.getShare(0)->begin(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [len1, len2] __device__ (int idx) { // "transpose"
                    const int batch_id = idx / len2;
                    const int batch_el = idx % len2;
                    return batch_el * len1 + batch_id;
                }));
    auto batch_reduction_in_1 =
        thrust::make_permutation_iterator(
            arr1.getShare(1)->begin(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [len1, len2] __device__ (int idx) { // "transpose"
                    const int batch_id = idx / len2;
                    const int batch_el = idx % len2;
                    return batch_el * len1 + batch_id;
                }));

    thrust::reduce_by_key(thrust::make_counting_iterator(0),
                          thrust::make_counting_iterator(len1 * len2),
                          batch_reduction_in_0,
                          thrust::make_discard_iterator(),
                          arr2.getShare(0)->begin(),
                          [len2] __device__ (int idx, int idy) {
                              return idx / len2 == idy / len2;
                           });
    thrust::reduce_by_key(thrust::make_counting_iterator(0),
                          thrust::make_counting_iterator(len1 * len2),
                          batch_reduction_in_1,
                          thrust::make_discard_iterator(),
                          arr2.getShare(1)->begin(),
                          [len2] __device__ (int idx, int idy) {
                              return idx / len2 == idy / len2;
                           });
}

// TODO change into 2 arguments with subtraction, pointer NULL indicates compare w/ 0
template<typename T, typename U, typename I, typename I2>
void dReLU(const RSS<T, I> &input, RSS<U, I2> &result) {

    int bitWidth = sizeof(T) * 8;

    RSS<T> r(input.size());

    RSS<U> rbits(input.size() * bitWidth);
    rbits.fill(1);

    DeviceData<T> a(input.size());
    r += input;
    reconstruct(r, a);
    a += 1;

    DeviceData<U> abits(rbits.size());
    gpu::bitexpand(&a, &abits);

    RSS<U> msb(input.size());

    // setCarryOutMSB overwrites abits/rbits, so make sure if we're party C that we don't accidentally use the modified values (hacky)
    if (partyNum != RSS<U>::PARTY_C) {
        gpu::setCarryOutMSB(*(rbits.getShare(0)), abits, *(msb.getShare(0)), bitWidth, partyNum == RSS<U>::PARTY_A);
        gpu::setCarryOutMSB(*(rbits.getShare(1)), abits, *(msb.getShare(1)), bitWidth, partyNum == RSS<U>::PARTY_C);
    } else {
        gpu::setCarryOutMSB(*(rbits.getShare(1)), abits, *(msb.getShare(1)), bitWidth, partyNum == RSS<U>::PARTY_C);
        gpu::setCarryOutMSB(*(rbits.getShare(0)), abits, *(msb.getShare(0)), bitWidth, partyNum == RSS<U>::PARTY_A);
    }

    RSS<U> g(rbits.size());
    g.zero();
    g += rbits;
    g &= abits;

    RSS<U> p(rbits.size());
    p.zero();
    p += rbits;
    p ^= abits;

    RSS<U> preResult(result.size());
    carryOut(p, g, bitWidth, preResult);

    preResult ^= msb;

    result.fill(1);
    result -= preResult;
}
    
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const RSS<T, I> &input, RSS<T, I2> &result, RSS<U, I3> &dresult) {
    //func_profiler.start();
    dReLU(input, dresult);
    //func_profiler.accumulate("relu-drelu");

    RSS<T> zeros(input.size());

    //func_profiler.start();
    selectShare(zeros, input, dresult, result);
    //func_profiler.accumulate("relu-selectshare");
}

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(const RSS<T, I> &input, RSS<T, I2> &result, RSS<U, I3> &dresult, int k) {

    // d(Maxpool) setup
    dresult.fill(1);

    // split input into even, odd
    using SRIterator = typename StridedRange<I>::iterator;

    int stride = 2;
    int offset = 1;

    func_profiler.start();
    StridedRange<I> even0Range(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> even0(even0Range.begin(), even0Range.end());
    StridedRange<I> even1Range(input.getShare(1)->begin(), input.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> even1(even1Range.begin(), even1Range.end());
    RSS<T, SRIterator> even(&even0, &even1);

    StridedRange<I> odd0Range(input.getShare(0)->begin() + offset, input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> odd0(odd0Range.begin(), odd0Range.end());
    StridedRange<I> odd1Range(input.getShare(1)->begin() + offset, input.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> odd1(odd1Range.begin(), odd1Range.end());
    RSS<T, SRIterator> odd(&odd0, &odd1);
    func_profiler.accumulate("range creation");

    //printf("func-maxpool-post-rangecreate\n");
    //printMemUsage();

    while(k > 2) {

        // -- MP --

        // diff = even - odd
        func_profiler.start();
        RSS<T> diff(even.size());
        diff.zero();
        diff += even;
        diff -= odd;
        func_profiler.accumulate("maxpool-diff");

        //printf("func-maxpool-post-diff-k=%d\n", k);
        //printMemUsage();

        // DRELU diff -> b
        func_profiler.start();
        RSS<U> b(even.size());
        dReLU(diff, b);
        func_profiler.accumulate("maxpool-drelu");

        //printf("func-maxpool-post-drelu-k=%d\n", k);
        //printMemUsage();
        
        selectShare(odd, even, b, even);

        // unzip even -> into even, odd
        stride *= 2;

        //printf("func-maxpool-pre-rangeupdate-k=%d\n", k);
        //printMemUsage();

        func_profiler.start();
        even0Range.set(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
        even0.set(even0Range.begin(), even0Range.end());
        even1Range.set(input.getShare(1)->begin(), input.getShare(1)->end(), stride);
        even1.set(even1Range.begin(), even1Range.end());
        even.set(&even0, &even1);

        odd0Range.set(input.getShare(0)->begin() + stride/2, input.getShare(0)->end(), stride);
        odd0.set(odd0Range.begin(), odd0Range.end());
        odd1Range.set(input.getShare(1)->begin() + stride/2, input.getShare(1)->end(), stride);
        odd1.set(odd1Range.begin(), odd1Range.end());
        odd.set(&odd0, &odd1);
        func_profiler.accumulate("maxpool-unzip");
        
        // -- dMP --

        //printf("func-maxpool-pre-expand-k=%d\n", k);
        //printMemUsage();

        // expandCompare b -> expandedB
        func_profiler.start();
        RSS<U> negated(b.size());
        negated.fill(1);
        negated -= b;
        RSS<U> expandedB(input.size());

        gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));
        gpu::expandCompare(*b.getShare(1), *negated.getShare(1), *expandedB.getShare(1));

        func_profiler.accumulate("maxpool-expandCompare");

        //printf("func-maxpool-post-expand-k=%d\n", k);
        //printMemUsage();
        
        // dresult &= expandedB
        func_profiler.start();
        dresult &= expandedB;
        func_profiler.accumulate("maxpool-dcalc");

        k /= 2;
    }

    // Fencepost - don't unzip the final results after the last comparison and finish
    // calculating derivative.
    
    // -- MP --
    
    // diff = even - odd
    func_profiler.start();
    RSS<T> diff(even.size());
    diff.zero();
    diff += even;
    diff -= odd;
    func_profiler.accumulate("maxpool-z-diff");

    // DRELU diff -> b
    func_profiler.start();
    RSS<U> b(even.size());
    dReLU(diff, b);
    func_profiler.accumulate("maxpool-z-drelu");
    
    // b * even + 1-b * odd
    selectShare(odd, even, b, even);

    func_profiler.start();
    //even *= b;
    //odd *= negated;
    //even += odd;

    result.zero();
    result += even;
    func_profiler.accumulate("maxpool-z-calc");

    // -- dMP --

    // expandCompare b -> expandedB
    func_profiler.start();
    RSS<U> negated(b.size());
    negated.fill(1);
    negated -= b;
    RSS<U> expandedB(input.size());
    gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));
    gpu::expandCompare(*b.getShare(1), *negated.getShare(1), *expandedB.getShare(1));
    func_profiler.accumulate("maxpool-z-expandCompare");
    
    // dresult &= expandedB
    func_profiler.start();
    dresult &= expandedB;
    func_profiler.accumulate("maxpool-z-dcalc");
}

template<typename T, typename I, typename I2>
void reconstruct3out3(DeviceData<T, I> &input, DeviceData<T, I2> &reconst) {

    auto next = RSS<T>::nextParty(partyNum);
    auto prev = RSS<T>::prevParty(partyNum);
    
    DeviceData<T> reconst1(input.size());
    DeviceData<T> reconst2(input.size());

    reconst.zero();
    reconst += input;

    comm_profiler.start();
    reconst1.receive(prev);

    input.transmit(next);
    input.join();

    reconst2.receive(next);

    input.transmit(prev);
    input.join();

    reconst1.join();
    reconst2.join();
    comm_profiler.accumulate("comm-time");

    reconst += reconst1;
    reconst += reconst2;

    func_profiler.add_comm_round();
}

template<typename T, typename I, typename I2>
void reshare(DeviceData<T, I> &c, RSSBase<T, I2> &out) {

    auto next = RSS<T>::nextParty(partyNum);
    auto prev = RSS<T>::prevParty(partyNum);

    DeviceData<T> rndMask(c.size());
    rndMask += c;

    // jank equivalent to =
    out.zero();
    *out.getShare(0) += rndMask;

    comm_profiler.start();
    out.getShare(0)->transmit(prev);
    out.getShare(1)->receive(next);

    out.getShare(0)->join();
    out.getShare(1)->join();
    comm_profiler.accumulate("comm-time");

    func_profiler.add_comm_round();
}

template<typename T, typename I, typename I2>
void dividePublic3out3(DeviceData<T, I> &a, T denominator, RSSBase<T, I2> &out) {

    size_t size = a.size();

    RSS<T> r(size), rPrime(size);
    // r = 1, rPrime = power
    PrecomputeObject.getDividedShares<T, RSS<T> >(r, rPrime, denominator, size); 

    a -= *rPrime.getShare(0);
    
    DeviceData<T> reconstructed(size);
    reconstruct3out3(a, reconstructed);
    reconstructed /= (T)denominator;

    out.zero();
    out += r;
    out += reconstructed;
}

template<typename T, typename I, typename I2>
void carryOut_EQ(RSS<T, I> &p, int k, RSS<T, I2> &out) {

    // get zip iterators on both p and g
    //  -> pEven, pOdd, gEven, gOdd
    
    int stride = 2;
    int offset = 1;

    using SRIterator = typename StridedRange<I>::iterator;

    StridedRange<I> pEven0Range(p.getShare(0)->begin(), p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pEven0(pEven0Range.begin(), pEven0Range.end());
    StridedRange<I> pEven1Range(p.getShare(1)->begin(), p.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> pEven1(pEven1Range.begin(), pEven1Range.end());
    RSS<T, SRIterator> pEven(&pEven0, &pEven1);

    StridedRange<I> pOdd0Range(p.getShare(0)->begin() + offset, p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pOdd0(pOdd0Range.begin(), pOdd0Range.end());
    StridedRange<I> pOdd1Range(p.getShare(1)->begin() + offset, p.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> pOdd1(pOdd1Range.begin(), pOdd1Range.end());
    RSS<T, SRIterator> pOdd(&pOdd0, &pOdd1);

    while(k > 1) {
        // pEven & pOdd
        //  store result in pEven
        pEven &= pOdd;

        //  pEven -> pEven, pOdd
        stride *= 2;

        pEven0Range.set(p.getShare(0)->begin(), p.getShare(0)->end(), stride);
        pEven0.set(pEven0Range.begin(), pEven0Range.end());
        pEven1Range.set(p.getShare(1)->begin(), p.getShare(1)->end(), stride);
        pEven1.set(pEven1Range.begin(), pEven1Range.end());
        pEven.set(&pEven0, &pEven1);

        pOdd0Range.set(p.getShare(0)->begin() + stride/2, p.getShare(0)->end(), stride);
        pOdd0.set(pOdd0Range.begin(), pOdd0Range.end());
        pOdd1Range.set(p.getShare(1)->begin() + stride/2, p.getShare(1)->end(), stride);
        pOdd1.set(pOdd1Range.begin(), pOdd1Range.end());
        pOdd.set(&pOdd0, &pOdd1);
        
        k /= 2;
    }

    // copy output to destination
    // out.zip(gEven, gOdd);
    // the below should be for p.
    StridedRange<I> outputEven0Range(out.getShare(0)->begin(), out.getShare(0)->end(), 2);
    thrust::copy(pEven.getShare(0)->begin(), pEven.getShare(0)->end(), outputEven0Range.begin());

    StridedRange<I> outputEven1Range(out.getShare(1)->begin(), out.getShare(1)->end(), 2);
    thrust::copy(pEven.getShare(1)->begin(), pEven.getShare(1)->end(), outputEven1Range.begin());

    StridedRange<I> outputOdd0Range(out.getShare(0)->begin() + 1, out.getShare(0)->end(), 2);
    thrust::copy(pOdd.getShare(0)->begin(), pOdd.getShare(0)->end(), outputOdd0Range.begin());

    StridedRange<I> outputOdd1Range(out.getShare(1)->begin() + 1, out.getShare(1)->end(), 2);
    thrust::copy(pOdd.getShare(1)->begin(), pOdd.getShare(1)->end(), outputOdd1Range.begin());
}

template<typename T, typename I, typename I2>
void carryOut(RSS<T, I> &p, RSS<T, I> &g, int k, RSS<T, I2> &out) {

    // get zip iterators on both p and g
    //  -> pEven, pOdd, gEven, gOdd
    
    int stride = 2;
    int offset = 1;

    using SRIterator = typename StridedRange<I>::iterator;

    StridedRange<I> pEven0Range(p.getShare(0)->begin(), p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pEven0(pEven0Range.begin(), pEven0Range.end());
    StridedRange<I> pEven1Range(p.getShare(1)->begin(), p.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> pEven1(pEven1Range.begin(), pEven1Range.end());
    RSS<T, SRIterator> pEven(&pEven0, &pEven1);

    StridedRange<I> pOdd0Range(p.getShare(0)->begin() + offset, p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pOdd0(pOdd0Range.begin(), pOdd0Range.end());
    StridedRange<I> pOdd1Range(p.getShare(1)->begin() + offset, p.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> pOdd1(pOdd1Range.begin(), pOdd1Range.end());
    RSS<T, SRIterator> pOdd(&pOdd0, &pOdd1);

    StridedRange<I> gEven0Range(g.getShare(0)->begin(), g.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> gEven0(gEven0Range.begin(), gEven0Range.end());
    StridedRange<I> gEven1Range(g.getShare(1)->begin(), g.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> gEven1(gEven1Range.begin(), gEven1Range.end());
    RSS<T, SRIterator> gEven(&gEven0, &gEven1);

    StridedRange<I> gOdd0Range(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> gOdd0(gOdd0Range.begin(), gOdd0Range.end());
    StridedRange<I> gOdd1Range(g.getShare(1)->begin() + offset, g.getShare(1)->end(), stride);
    DeviceData<T, SRIterator> gOdd1(gOdd1Range.begin(), gOdd1Range.end());
    RSS<T, SRIterator> gOdd(&gOdd0, &gOdd1);

    while(k > 1) {

        // gTemp = pOdd & gEven
        //  store result in gEven
        gEven &= pOdd;

        // pEven & pOdd
        //  store result in pEven
        pEven &= pOdd;

        // gOdd ^ gTemp
        //  store result in gOdd
        gOdd ^= gEven;
        
        // regenerate zip iterators to p and g
        
        //  gOdd -> gEven, gOdd
        gEven0Range.set(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride*2);
        gEven0.set(gEven0Range.begin(), gEven0Range.end());
        gEven1Range.set(g.getShare(1)->begin() + offset, g.getShare(1)->end(), stride*2);
        gEven1.set(gEven1Range.begin(), gEven1Range.end());
        gEven.set(&gEven0, &gEven1);

        offset += stride;

        gOdd0Range.set(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride*2);
        gOdd0.set(gOdd0Range.begin(), gOdd0Range.end());
        gOdd1Range.set(g.getShare(1)->begin() + offset, g.getShare(1)->end(), stride*2);
        gOdd1.set(gOdd1Range.begin(), gOdd1Range.end());
        gOdd.set(&gOdd0, &gOdd1);

        //  pEven -> pEven, pOdd
        stride *= 2;

        pEven0Range.set(p.getShare(0)->begin(), p.getShare(0)->end(), stride);
        pEven0.set(pEven0Range.begin(), pEven0Range.end());
        pEven1Range.set(p.getShare(1)->begin(), p.getShare(1)->end(), stride);
        pEven1.set(pEven1Range.begin(), pEven1Range.end());
        pEven.set(&pEven0, &pEven1);

        pOdd0Range.set(p.getShare(0)->begin() + stride/2, p.getShare(0)->end(), stride);
        pOdd0.set(pOdd0Range.begin(), pOdd0Range.end());
        pOdd1Range.set(p.getShare(1)->begin() + stride/2, p.getShare(1)->end(), stride);
        pOdd1.set(pOdd1Range.begin(), pOdd1Range.end());
        pOdd.set(&pOdd0, &pOdd1);
        
        k /= 2;
    }

    // copy output to destination
    // out.zip(gEven, gOdd);
    StridedRange<I> outputEven0Range(out.getShare(0)->begin(), out.getShare(0)->end(), 2);
    thrust::copy(gEven.getShare(0)->begin(), gEven.getShare(0)->end(), outputEven0Range.begin());

    StridedRange<I> outputEven1Range(out.getShare(1)->begin(), out.getShare(1)->end(), 2);
    thrust::copy(gEven.getShare(1)->begin(), gEven.getShare(1)->end(), outputEven1Range.begin());

    StridedRange<I> outputOdd0Range(out.getShare(0)->begin() + 1, out.getShare(0)->end(), 2);
    thrust::copy(gOdd.getShare(0)->begin(), gOdd.getShare(0)->end(), outputOdd0Range.begin());

    StridedRange<I> outputOdd1Range(out.getShare(1)->begin() + 1, out.getShare(1)->end(), 2);
    thrust::copy(gOdd.getShare(1)->begin(), gOdd.getShare(1)->end(), outputOdd1Range.begin());
}

template<typename T, typename I, typename I2>
void getPowers(const RSS<T, I> &in, DeviceData<T, I2> &pow) {

    RSS<T> powers(pow.size()); // accumulates largest power yet tested that is less than the input val
    RSS<T> currentPowerBit(in.size()); // current power
    RSS<T> diff(in.size());
    RSS<uint8_t> comparisons(in.size());

    for (int bit = 0; bit < sizeof(T) * 8; bit++) {
        currentPowerBit.fill(bit);

        diff.zero();
        diff += in;
        diff -= (((T)1) << bit);

        comparisons.zero();
        dReLU(diff, comparisons); // 0 -> current power is larger than input val, 1 -> input val is larger than current power

        // 0 -> keep val, 1 -> update to current known largest power less than input
        selectShare(powers, currentPowerBit, comparisons, powers);
    }

    reconstruct(powers, pow);
}

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(const RSS<T, I> &in, RSS<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn) {

    out.zero();
    RSS<T> scratch(out.size());

    DeviceData<T> pow(out.size());
    getPowers(in, pow);
    pow += 1;

    DeviceData<T> ones(pow.size());
    ones.fill(1);
    ones <<= pow;

    if (a2 != 0.0) {
        DeviceData<T> a2Coeff(out.size());
        thrust::transform(
            pow.begin(), pow.end(), a2Coeff.begin(),
            tofixed_variable_precision_functor<T>(a2));

        scratch.zero();
        scratch += in;
        scratch *= in;
        dividePublic(scratch, ones);

        scratch *= a2Coeff;
        dividePublic(scratch, ones);
        out += scratch;
    }

    if (a1 != 0.0) {

        DeviceData<T> a1Coeff(out.size());
        thrust::transform(
            pow.begin(), pow.end(), a1Coeff.begin(),
            tofixed_variable_precision_functor<T>(a1));

        scratch.zero();
        scratch += in;
        scratch *= a1Coeff;
        dividePublic(scratch, ones);

        out += scratch;
    }

    DeviceData<T> a0Coeff(out.size());
    thrust::transform(
        pow.begin(), pow.end(), a0Coeff.begin(),
        tofixed_variable_precision_functor<T>(a0));
    out += a0Coeff;

    DeviceData<T> powCoeff(out.size());
    thrust::transform(
        pow.begin(), pow.end(), powCoeff.begin(),
        calc_fn<T, Functor>(fn));
    out *= powCoeff;
    dividePublic(out, ones);

    // turn values back to base (e.g. 20 bit) precision

    pow -= FLOAT_PRECISION;

    DeviceData<T> positivePow(pow.size());
    thrust::transform(
        pow.begin(), pow.end(), positivePow.begin(),
        filter_positive_powers<T>());

    ones.fill(1);
    ones <<= positivePow;
    dividePublic(out, ones);

    DeviceData<T> negativePow(pow.size());
    thrust::transform(
        pow.begin(), pow.end(), negativePow.begin(),
        filter_negative_powers<T>());

    for (int share = 0; share <= 1; share++) {
        thrust::transform(
            out.getShare(share)->begin(), out.getShare(share)->end(), negativePow.begin(), out.getShare(share)->begin(),
            lshift_functor<T>()); 
    }
}

template<typename T, typename U, typename I, typename I2>
void convex_comb(RSS<T, I> &a, RSS<T, I> &c, DeviceData<U, I2> &b) {

    thrust::for_each(
        thrust::make_zip_iterator(
            thrust::make_tuple(b.begin(), c.getShare(0)->begin(), c.getShare(1)->begin(), a.getShare(0)->begin(), a.getShare(1)->begin())
        ),
        thrust::make_zip_iterator(
            thrust::make_tuple(b.end(), c.getShare(0)->end(), c.getShare(1)->end(), a.getShare(0)->end(), a.getShare(1)->end())
        ),
        rss_convex_comb_functor(partyNum)
    );
}

