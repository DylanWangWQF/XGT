
#include "unitTests.h"
extern Profiler comm_profiler;

// #include <cutlass/conv/convolution.h>

template<typename T>
struct FuncTest : public testing::Test {
    using ParamType = T;
};

using namespace thrust::placeholders;

// using Types = testing::Types<RSS<uint64_t>, TPC<uint64_t>, FPC<uint64_t>, OPC<uint64_t> >;
// using Types = testing::Types<RSS<uint64_t>, TPC<uint64_t>, OPC<uint64_t> >;
using Types = testing::Types<RSS<uint64_t>>;
// using Types = testing::Types<RSS<uint32_t>>;
TYPED_TEST_CASE(FuncTest, Types);


void random_vector(std::vector<double> &v, int size) {

    v.clear();
    v.resize(size);

    for (int i = 0; i < v.size(); i++) {
        v[i] = (double)1;
    }
}

TEST(FuncTest, AccMulWithDelayShare) {
    if (partyNum >= 3) return;

    RSS<uint32_t> a({1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}, false); // 4*3*2
    RSS<uint32_t> b(a.size());
    b.fill(1);
    RSS<uint32_t> c(8);

    AccMulWithDelayShare(a, b, c, 4, 3, 2);

    DeviceData<uint32_t> result(c.size());
    reconstruct(c, result);

    printDeviceData(result, "AccMulWithDelayShare", false);
}

TEST(FuncTest, BooleanShareAdd) {

    if (partyNum >= 3) return;

    RSS<uint8_t> p({0, 1, 1, 0}, false);
    RSS<uint8_t> g({0, 1, 0, 0}, false);
    g *= (uint8_t) -1;
    g += 1;
    p *= g;

    DeviceData<uint8_t> result(4);
    reconstruct(p, result);

    printDeviceData(result, "BooleanShareAdd", false);
}

TEST(FuncTest, Transpose) {
    if (partyNum >= 3) return;

    RSS<uint64_t> a({1, 4, 2, 5, 3, 6}, false); // 2 * 3
    RSS<uint64_t> b(a.size()); // 3 * 2

    gpu::transpose(a.getShare(0), b.getShare(0), 2, 3);
    gpu::transpose(a.getShare(1), b.getShare(1), 2, 3);
    cudaDeviceSynchronize();

    DeviceData<uint64_t> result(a.size());
    reconstruct(b, result);

    printDeviceData(result, "Transpose_ShareMat", false);
    // expected = {1, 2, 3, 4, 5, 6};
}

TYPED_TEST(FuncTest, negateAddDev) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    DeviceData<T> a = {1, 2, 2, 1, 1, 2};
    negateAddDev(a, 2, 1);

    printDeviceData(a, "negateAddDev", false);
    // even: 0, 2, -1, 1, 0, 2
    // odd:  1, -1, 2, 0, 1, -1
}

TEST(FuncTest, MatMul_3PC_Self) {

    if (partyNum >= 3) return;

    std::vector<double> rnd_vals;

    std::vector<int> N = {5};
    for (int i = 0; i < N.size(); i++) {

        int n = N[i];

        // matrix is stored in column in the vector
        random_vector(rnd_vals, n * n);
        RSS<uint64_t> a(n*n);
        a.setPublic(rnd_vals, false);

        // random_vector(rnd_vals, n * n);
        RSS<uint64_t> b(n*n);
        b.setPublic(rnd_vals, false);

        RSS<uint64_t> c(n*n);

        Profiler profiler;
        profiler.start();

        matmul(a, b, c, n, n, n, false, false, false);

        profiler.accumulate("matmul");

        // check the matrix result c
        DeviceData<uint64_t> addvec(c.size());
        addvec.fill(1);
        c += addvec;
        DeviceData<uint64_t> result(c.size());
        reconstruct(c, result);

        // if (i == 0) continue; // sacrifice run to spin up GPU
        cout << "Check the random vector." << endl;
        for (int i = 0; i < rnd_vals.size(); i++)
        {
            cout << rnd_vals[i] << " ";
        }
        cout << endl;
        printDeviceData(result, "result of matrix mul", false);
        printf("3PC - matmul (N=%d) - %f sec.\n", n, profiler.get_elapsed("matmul") / 1000.0);
    }
}

TYPED_TEST(FuncTest, Merge_DeviceData) {
    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;
    
    if (partyNum >= Share::numParties) return;

    // std::vector<double> rnd_vals;

    // int len1 = 6, len2 = 3;
    // int batch = len1;

    // random_vector(rnd_vals, len1 * len2);
    // Share arr1(len1 * len2);
    // arr1.setPublic(rnd_vals, false);

    // for (int i = 0; i < rnd_vals.size(); i++) {
    //     rnd_vals[i] = (double)2;
    // }
    // Share arr2(len1 * len2);
    // arr2.setPublic(rnd_vals, false);

    // size_t out_size = arr1.size() + arr2.size();
    // Share arr3(out_size);
    // arr3.zero();

    Share arr1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, false);
    Share arr2({1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3}, false);
    Share arr3(arr1.size() + arr2.size());
    int batch = 5;

    merge_vecs(arr1, arr2, arr3, batch);

    DeviceData<T> result(arr3.size());
    reconstruct(arr3, result);
    printDeviceData(result, "result of merge two vectors", false);
}

TYPED_TEST(FuncTest, Reconstruct) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    // Share a = {1, 1, 2, 8, 12.2, 1};  // 2 x 3
    Share a ({1, 1, 2, 8, 12, 1}, false); 

    DeviceData<T> result(6);

    // func_profiler.clear();
    // func_profiler.start();
    comm_profiler.clear();
    reconstruct(a, result);
    printDeviceData(result, "Reconstruct", false);

    printf("3PC-Reconstruct: TX comm (Byte),%f\n", comm_profiler.get_comm_tx_bytes() / 1.0);
    printf("3PC-Reconstruct: RX comm (Byte),%f\n", comm_profiler.get_comm_rx_bytes() / 1.0);

    // std::vector<double> expected = {1, 1, 2, 8, 12.2, 1};
    // assertDeviceData(result, expected);
}

TYPED_TEST(FuncTest, Mult) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a ({12, 24, 3, 5, -2, -3}, false); 
    Share b ({1, 0, 11, 3, -1, 11}, false);

    DeviceData<T> resultA(a.size());
    DeviceData<T> resultB(b.size());

    // func_profiler.clear();
    // func_profiler.start();
    // b *= a;
    a *= (T) -2;
    reconstruct(a, resultA);
    // b *= (T) -1;
    b *= a;
    reconstruct(b, resultB);

    printDeviceData(resultA, "resultA", false);
    printDeviceData(resultB, "resultB", false);
    // resultA:
    // -24.000000 -48.000000 -6.000000 -10.000000 4.000000 6.000000
    // resultB:
    // -1.000000 0.000000 -11.000000 -3.000000 1.000000 -11.000000

    // std::vector<double> expected = {12, 0, 33, 15, 2, -33};
    // assertDeviceData(result, expected, false);
}

TYPED_TEST(FuncTest, MultFloat) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a ({292, 308, 299, 301, 297, 303}, false); 
    Share b ({292, 308, 299, 301, 297, 303}, false);
    Share c ({0.01, 0.01, 0.01, 0.01, 0.01, 0.01});

    DeviceData<T> resultA(a.size());
    DeviceData<T> resultB(b.size());

    // a *= (T) (0.01 * (1 << FLOAT_PRECISION));
    a *= c;
    reconstruct(a, resultA);

    b >>= (T) 1;
    reconstruct(b, resultB);

    printDeviceData(resultA, "MultFloat_resultA");
    printDeviceData(resultB, "MultFloat_resultB", false);
    // resultA:
    // -24.000000 -48.000000 -6.000000 -10.000000 4.000000 6.000000
    // resultB:
    // -1.000000 0.000000 -11.000000 -3.000000 1.000000 -11.000000

    // std::vector<double> expected = {12, 0, 33, 15, 2, -33};
    // assertDeviceData(result, expected, false);
}

TYPED_TEST(FuncTest, Inverse) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        0.25,
        0.3,
        0.375,
        0.49,
        0.5,
        0.51,
        0.75,
        1,
        1.500e+0,
        400,
        4000,
        40000
    };

    Share result(input.size());
    inverse(input, result);

    std::vector<double> expected = {
        0.25,
        0.3,
        0.375,
        0.49,
        0.5,
        0.51,
        0.75,
        1,
        1.500e+0,
        400,
        4000,
        40000
    };

    cout << "Check real inverse result" << endl;
    for (int i = 0; i < expected.size(); i++) {
        expected[i] = 1 / expected[i];
        cout << expected[i] << " ";
    }
    cout << endl;

    DeviceData<T> super_result(input.size());
    reconstruct(result, super_result);
    printDeviceData(super_result, "MPC inverse result");
    // assertDeviceData(super_result, expected, true, 1e-1);

    // Check real inverse result
    // 4 3.33333 2.66667 2.04082 2 1.96078 1.33333 1 0.666667 0.0025 0.00025 2.5e-05 
    // MPC inverse result:
    // 3.937500 3.312500 2.656250 2.062500 1.976562 1.945312 1.335938 0.984375 0.664062 0.007812 0.007812 0.007812

    // Check real inverse result
    // 4 3.33333 2.66667 2.04082 2 1.96078 1.33333 1 0.666667 0.0025 0.00025 2.5e-05 
    // MPC inverse result:
    // 3.947998 3.354980 2.663208 2.061768 1.973969 1.942017 1.331604 0.986969 0.665802 0.002472 0.000244 0.000000
}

TYPED_TEST(FuncTest, negateAdd) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a ({0, 1, 2, 3, -4, -5}, false);
    negateAdd(a, 2);
    DeviceData<T> resultA(a.size());
    reconstruct(a, resultA);

    printDeviceData(resultA, "negateAdd", false);
    // resultA:
    // 1.000000 1.000000 -1.000000 3.000000 5.000000 -5.000000
}

TYPED_TEST(FuncTest, testOR) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a ({0, 1, 1, 3, -1, -1}, false);
    Share b ({0, 0, 1, 3, -1, 1}, false);
    a |= b;
    

    DeviceData<T> result(a.size());
    reconstruct(a, result);

    printDeviceData(result, "testOR", false);
    // resultA:
    // 1.000000 1.000000 -1.000000 3.000000 5.000000 -5.000000
}

TYPED_TEST(FuncTest, SelectShare) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share x = {1, 2, 10, 1};
    Share y = {4, 5, 1, 6};
    Share b({1, 1, 0, 1}, false);

    Share z(x.size());
    selectShare(x, y, b, z);

    DeviceData<T> result(4);
    reconstruct(z, result);
    
    std::vector<double> expected = {4, 5, 10, 6};
    assertDeviceData(result, expected);
}

TYPED_TEST(FuncTest, Check_Equality) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        0, 1, 2, -3, 5
    };

    //Change Share to TPC<uint8_t>
    Share result(input.size());
    EQ(input, result);

    //Change to <uint8_t>
    DeviceData<T> super_result(result.size());
    reconstruct(result, super_result);
    printDeviceData(super_result, "CheckEquality", false);
    // CheckEquality:
    // 1.000000 0.000000 0.000000 0.000000 0.000000
}

TYPED_TEST(FuncTest, Check_Batch_Op) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share arr1({0, 5, 11, 1, 19}, false);
    Share arr2(arr1.size() * 4);

    size_t len = 4;
    fill_in_batch(arr1, arr2, len, 0);

    DeviceData<T> super_result(arr2.size());
    reconstruct(arr2, super_result);
    printDeviceData(super_result, "Check_Batch_Op", false);
}
