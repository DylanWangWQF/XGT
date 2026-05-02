#include <iostream>
#include <string>
#include <map>
#include <json.hpp>
#include <sys/time.h>
#include <chrono>

#include "globals.h"
#include "mpc/AESObject.h"
#include "mpc/Precompute.h"
#include "util/connect.h"
#include "test/unitTests.h"
#include "util/Profiler.h"
#include "mpc/RSS.h"
#include "util/util.cuh"
#include "../ext/cxxopts.hpp"
#include "dt/DecisionTree.h"

// using namespace std;

int partyNum;
std::vector<AESObject*> aes_objects;
//AESObject* aes_indep;
//AESObject* aes_next;
//AESObject* aes_prev;
Precompute PrecomputeObject;

extern std::string *addrs;
extern BmrNet **communicationSenders;
extern BmrNet **communicationReceivers;

extern Profiler matmul_profiler;
Profiler func_profiler;
Profiler memory_profiler;
Profiler comm_profiler;
Profiler debug_profiler;

nlohmann::json gst_config;

size_t db_bytes = 0;
size_t db_layer_max_bytes = 0;
size_t db_max_bytes = 0;


long TREE_DEPTH_PARAM;
long INSTANCE_COUNT_PER_ITER;
long CLASS_COUNT;
long ATTRIBUTE_COUNT_TOTAL;
long TREE_DEPTH;
long NODE_COUNT;
long LEAF_COUNT;
long LEAF_COUNTER_SIZE;
long LEAF_COUNTERS_SIZE;
long INSTANCE_COUNT_TOTAL;
long INF_COUNT_TOTAL;
int scaleFactor;

template<typename T, template<typename, typename...> typename Share>
void train(DecisionTree<T, Share> *, std::ifstream &, std::map<std::string, int> &);
template<typename T, template<typename, typename...> typename Share>
void inference(DecisionTree<T, Share> *, std::ifstream &, std::map<std::string, int> &);
template<typename T, template<typename, typename...> typename Share>
void inference_simulation(DecisionTree<T, Share> *, std::ifstream &, std::map<std::string, int> &);
template<typename T, template<typename, typename...> typename Share>
void PrintDT(DecisionTree<T, Share> *);

std::vector<std::string> split_attributes(std::string , char );
std::vector<std::string> split(std::string , std::string );
// bool prepare_data(std::ifstream &, std::vector<double> &, std::map<string, double> &, int );
void deleteObjects();


int main(int argc, char** argv) {

    // Parse options -- retrieve party id and config JSON
    cxxopts::Options options("XGT", "Fast and Secure Decision Tree Training and Inference on GPUs");
    options.add_options()
        ("p,party", "Party number", cxxopts::value<int>())
        ("c,config", "Configuration file", cxxopts::value<std::string>())
        ;
    // bypass the undefinited options
    options.allow_unrecognised_options();
    // parse the param
    auto parsed_options = options.parse(argc, argv);
    // Print help
    if (parsed_options.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    partyNum = parsed_options["party"].as<int>();

    // get the json file of configure
    std::ifstream input_config(parsed_options["config"].as<std::string>());
    input_config >> gst_config;

    // Start memory profiler and initialize communication between parties
    // memory_profiler.start();

    //XXX initializeCommunication(options.ip_file, partyNum);
    // get IP address of each party
    std::vector<std::string> party_ips;
    for (int i = 0; i < gst_config["num_parties"]; i++) {
	    party_ips.push_back(gst_config["party_ips"][i]);
    }
    initializeCommunication(party_ips, partyNum, gst_config["num_parties"]);

    synchronize(10000, gst_config["num_parties"]); 
    
    for (size_t i = 0; i < gst_config["num_parties"]; i++) {
        // --------------> AES_TODO
        //Get AES strings from file and create vector of AESObjects
        //options.aes_file;
        //aes_objects[i] = new AESObject(options.)
    }
    //aes_indep = new AESObject(options.aes_indep_file);
    //aes_next = new AESObject(options.aes_next_file);
    //aes_prev = new AESObject(options.aes_prev_file);

    // Unit tests
    if (gst_config["run_unit_tests"]) {
        int returnCode = runTests(argc, argv);
        if (returnCode != 0 || gst_config["unit_test_only"]) {
            exit(returnCode);
        }
    }

    std::cout << "run unit tests? " << gst_config["run_unit_tests"] << std::endl;

    std::string data_path = gst_config["train_data_path"];
    // get params
    INSTANCE_COUNT_PER_ITER = (int) gst_config["tra_param"][0];
    INSTANCE_COUNT_TOTAL = (int) gst_config["tra_param"][1];
    int num_fea = (int) gst_config["tra_param"][2];
    TREE_DEPTH_PARAM = (int) gst_config["tra_param"][3];

    std::string class_path = data_path + "/labels.txt";
    std::string data_file_name = data_path + "/tra_" + std::to_string(INSTANCE_COUNT_TOTAL) + "_" + std::to_string(num_fea) + ".csv";
    std::string attribute_file_path = data_path + "/attributes_" + std::to_string(num_fea) + ".txt";


    // read attribute file
    std::ifstream attribute_file(attribute_file_path);
    std::string line;
    getline(attribute_file, line);

    ATTRIBUTE_COUNT_TOTAL = split(line, ",").size() - 1;
    TREE_DEPTH = TREE_DEPTH_PARAM == -1 ? ATTRIBUTE_COUNT_TOTAL + 1 : TREE_DEPTH_PARAM;
    NODE_COUNT = (1 << TREE_DEPTH) - 1; // 2^TREE_DEPTH - 1
    LEAF_COUNT = (1 << (TREE_DEPTH - 1)); // 2^(TREE_DEPTH - 1)

    // read class/label file
    std::ifstream class_file(class_path);
    std::string class_line;
    // init mapping between class and code
    std::map<std::string, int> class_code_map;
    std::map<int, std::string> code_class_map;
    std::vector<std::string> class_arr = split(class_line, " ");
    std::string code_str, class_str;
    int line_count = 0;
    while (class_file >> class_str) {
        int class_code = line_count;
        class_code_map[class_str] = class_code;
        code_class_map[class_code] = class_str;
        line_count++;
    }

    CLASS_COUNT = line_count;
    LEAF_COUNTER_SIZE = ATTRIBUTE_COUNT_TOTAL * 2 * (CLASS_COUNT + 2);
    LEAF_COUNTERS_SIZE = LEAF_COUNT * LEAF_COUNTER_SIZE;

    // find the scalFactor to scale the denominator in Compute Gini Index()
    if (gst_config["self_define_scale"])
    {
        scaleFactor = (int) gst_config["scale_factor"][0];
    }
    else
    {
        scaleFactor = 1;
        while (scaleFactor < INSTANCE_COUNT_TOTAL)
        {
            scaleFactor <<= 1;
        }
    }

    printf("INSTANCE_COUNT_PER_ITER: %ld\n", INSTANCE_COUNT_PER_ITER);
    printf("INSTANCE_COUNT_TOTAL: %ld\n", INSTANCE_COUNT_TOTAL);
    printf("ATTRIBUTE_COUNT_TOTAL: %ld\n", ATTRIBUTE_COUNT_TOTAL);
    printf("TREE_DEPTH: %ld\n", TREE_DEPTH);
    printf("NODE_COUNT: %ld\n", NODE_COUNT);
    printf("LEAF_COUNT: %ld\n", LEAF_COUNT);
    printf("CLASS_COUNT: %ld\n", CLASS_COUNT);
    printf("LEAF_COUNTER_SIZE: %i\n", LEAF_COUNTER_SIZE);
    printf("scaleFactor: %i\n", scaleFactor);

    std::cout << std::endl << "init Decision Tree: " << std::endl;
    DecisionTree<uint32_t, RSS> dt;
    // DecisionTree<uint64_t, RSS> dt;

    if (gst_config["if_simulate_inference"])
    {
        std::cout << std::endl << "Simulate DT inference for evaluation: " << std::endl;

        INF_COUNT_TOTAL = (int) gst_config["inf_param"][0];
        ATTRIBUTE_COUNT_TOTAL = (int) gst_config["inf_param"][1];
        TREE_DEPTH = (int) gst_config["inf_param"][2];
        printf("INF_COUNT_TOTAL: %i\n", INF_COUNT_TOTAL);
        printf("ATTRIBUTE_COUNT_TOTAL: %i\n", ATTRIBUTE_COUNT_TOTAL);
        printf("TREE_DEPTH: %i\n", TREE_DEPTH);

        std::string inf_data_path = gst_config["inf_data_path"];
        std::ifstream inf_data_file(inf_data_path);

        std::cout << "begin inference simulation: " << std::endl;
        inference_simulation(&dt, inf_data_file, class_code_map);
    }
    else
    {
        std::cout << "begin training: " << std::endl;
        // read data file 
        std::ifstream data_file(data_file_name);
        train(&dt, data_file, class_code_map);
        
        PrintDT(&dt);

        if (gst_config["if_run_inference"])
        {
            INF_COUNT_TOTAL = (int) gst_config["real_inf_param"][0];
            printf("INF_COUNT_TOTAL: %i\n", INF_COUNT_TOTAL);

            std::string inf_data_path = gst_config["real_inf_data_path"];
            std::ifstream inf_data_file(inf_data_path);

            std::cout << "begin inference: " << std::endl;
            inference(&dt, inf_data_file, class_code_map);
        }
    }

    // wait a bit for the prints to flush
    std::cout << std::flush;
    for(int i = 0; i < 10000000; i++);
   
    return 0;
}

bool prepare_data(std::ifstream &data_file, std::vector<double> &data, std::vector<double> &label, std::map<std::string, int> &class_code_map, int data_num) {

    int data_idx = 0;
    int label_idx = 0;

    std::string line;
    for (int instance_idx = 0; instance_idx < data_num; instance_idx++) {
        if (!getline(data_file, line)) {
            // reached end of line
            return false;
        }

        std::vector<std::string> raw_data_row = split(line, ",");

        for (int i = 0; i < ATTRIBUTE_COUNT_TOTAL; i++) {
            double val = stod(raw_data_row[i]);
            data[data_idx++] = val;
        }

        int cur_class_code = class_code_map[raw_data_row[ATTRIBUTE_COUNT_TOTAL]];
        label[label_idx++] = (double) cur_class_code;

    }

    return true;
}

bool prepare_data_onehot(std::ifstream &data_file, std::vector<double> &data, std::vector<double> &label, std::map<std::string, int> &class_code_map, int data_num) {

    int data_idx = 0;
    int label_idx = 0;

    std::string line;
    for (int instance_idx = 0; instance_idx < data_num; instance_idx++) {
        if (!getline(data_file, line)) {
            // reached end of line
            return false;
        }

        std::vector<std::string> raw_data_row = split(line, ",");

        for (int i = 0; i < ATTRIBUTE_COUNT_TOTAL; i++) 
        {
            if (raw_data_row[i].compare("0") == 0)
            {
                data[data_idx] = 0;
                data[++data_idx] = 1;
            }
            else
            {
                data[data_idx] = 1;
                data[++data_idx] = 0;
            }
            data_idx++;
        }

        int cur_class_code = class_code_map[raw_data_row[ATTRIBUTE_COUNT_TOTAL]];
        label[label_idx++] = (double) cur_class_code;
    }

    return true;
}

template<typename T, template<typename, typename...> typename Share>
void train(DecisionTree<T, Share> *dt, std::ifstream &train_data, std::map<std::string, int> &class_code_map)
{
    std::vector<double> data(INSTANCE_COUNT_PER_ITER * ATTRIBUTE_COUNT_TOTAL);
    std::vector<double> label(INSTANCE_COUNT_PER_ITER);
    std::vector<double> data_trans(INSTANCE_COUNT_PER_ITER * ATTRIBUTE_COUNT_TOTAL); // feature/col-wise
    
    double learning_total_comm_tx_mb = 0.0;
    double sgxhc_total_comm_tx_mb = 0.0;
    // double mpchc_total_comm_tx_mb = 0.0;
    double node_total_comm_tx_mb = 0.0;
    comm_profiler.clear();
    
    int tree_iter_count = 0;
    Profiler toplevel_profiler;
    while (true)
    {
        // iteration: tree level
        int data_iter_count = 0;
        train_data.clear();
        train_data.seekg(0,ios::beg);
        while (true)
        {
            toplevel_profiler.start();
            if (!prepare_data(train_data, data, label, class_code_map, INSTANCE_COUNT_PER_ITER)) {
                break; // reached end of line, jump to the first while loop
            }

            // transpose data matrix. (f1, f2, ...) instead of (D1, D2, ...)
            for (int i = 0; i < INSTANCE_COUNT_PER_ITER; i++)
            {
                for (int j = 0; j < ATTRIBUTE_COUNT_TOTAL; j++)
                {
                    data_trans[j * INSTANCE_COUNT_PER_ITER + i] = data[i * ATTRIBUTE_COUNT_TOTAL + j];
                }
            }
            toplevel_profiler.accumulate("process-data");

            toplevel_profiler.start();
            dt->counter_increase(data, data_trans, label);
            toplevel_profiler.accumulate("counter-increase");

            fflush(stdout);
        }

        int isFinished = 0;

        if (gst_config["compute_GiniIndex_SGX"])
        {
            // compute Gini Index in SGX
            int num = 1 << dt->tree_level;
            std::vector<double> node_split_decisions(num);
            std::vector<double> hostNodeFlag(NODE_COUNT);
            std::vector<double> newleafClass(num * 2);
            
            // comm_profiler.clear();
            toplevel_profiler.start();
            dt->compute_info_gain_SGX(node_split_decisions, hostNodeFlag, newleafClass, isFinished);
            toplevel_profiler.accumulate("compute-IG-SGX");

            if (isFinished == 1)
            {
                std::cout << "training completed, isFinished = " << isFinished << std::endl << std::endl;
                break; // all leaves are confirmed, training finishes. jump out the second while loop
            }

            toplevel_profiler.start();
            dt->node_split_SGX(node_split_decisions, hostNodeFlag, newleafClass);
            toplevel_profiler.accumulate("node-split");
        }
        else
        {
            toplevel_profiler.start();
            dt->compute_info_gain_MPC(isFinished);
            toplevel_profiler.accumulate("compute-IG-MPC");

            if (isFinished == 1)
            {
                std::cout << "training completed, isFinished = " << isFinished << std::endl << std::endl;
                break; // all leaves are confirmed, training finishes. jump out the second while loop
            }

            toplevel_profiler.start();
            dt->node_split_MPC();
            toplevel_profiler.accumulate("node-split");
        }
        tree_iter_count++;
    }


    if (gst_config["total_train_stats"]) {
        double total_time_s = toplevel_profiler.get_elapsed_all() / 1000.0;
        double data_time_s = toplevel_profiler.get_elapsed("process-data") / 1000.0;
        double real_total_time_s = total_time_s - data_time_s;
        printf("\ntotal time for process data (s),%f\n", data_time_s);
        printf("total time for training (s),%f\n", real_total_time_s);
        printf("Training: total tx comm (MB),%f\n", ((double)comm_profiler.get_comm_tx_bytes()) / 1024.0 / 1024.0);
        printf("Training: total rx comm (MB),%f\n", ((double)comm_profiler.get_comm_rx_bytes()) / 1024.0 / 1024.0);
        double comm_s = comm_profiler.get_elapsed("comm-time") / 1000.0;
        printf("training communication time (s),%f\n", comm_s);
        printf("training computation time (s),%f\n", real_total_time_s - comm_s);
    }

    if (gst_config["compute_Gini_stats"]) {
        if (gst_config["compute_GiniIndex_SGX"])
        { // use SGX
            printf("\ntotal time for computing Gini Index using SGX (s),%f\n", toplevel_profiler.get_elapsed("compute-IG-SGX") / 1000.0);
        }
        else
        { // use MPC
            printf("\ntotal time for computing Gini Index using MPC (s),%f\n", toplevel_profiler.get_elapsed("compute-IG-MPC") / 1000.0);
        }
    }
    
    if (gst_config["all_Func_stats"]) {
        printf("\ntotal time for learning phase (s),%f\n", toplevel_profiler.get_elapsed("counter-increase") / 1000.0);
        printf("total time for node split (s),%f\n", toplevel_profiler.get_elapsed("node-split") / 1000.0);
    }

    fflush(stdout);
}

template<typename T, template<typename, typename...> typename Share>
void inference(DecisionTree<T, Share> *dt, std::ifstream &inf_data_file, std::map<std::string, int> &class_code_map)
{
    std::vector<double> inf_data_vec_onehot(INF_COUNT_TOTAL * ATTRIBUTE_COUNT_TOTAL * 2);
    std::vector<double> inf_label_vec(INF_COUNT_TOTAL);
    if (!prepare_data_onehot(inf_data_file, inf_data_vec_onehot, inf_label_vec, class_code_map, INF_COUNT_TOTAL))
    {
        return;
    }
    
    std::vector<double> inf_data_vec_onehot_trans(inf_data_vec_onehot.size());
    for (int i = 0; i < INF_COUNT_TOTAL; i++)
    {
        for (int j = 0; j < ATTRIBUTE_COUNT_TOTAL * 2; j++)
        {
            inf_data_vec_onehot_trans[j * INF_COUNT_TOTAL + i] = inf_data_vec_onehot[i * ATTRIBUTE_COUNT_TOTAL * 2 + j];
        }
    }
    
    Share<T> inf_data(inf_data_vec_onehot.size());
    inf_data.setPublic(inf_data_vec_onehot_trans, false);
    Share<T> result_label(INF_COUNT_TOTAL);
    result_label.setPublic(inf_label_vec, false);

    double offline_total_comm_tx_mb = 0.0;
    double online_total_comm_tx_mb = 0.0;
    comm_profiler.clear();
    Profiler toplevel_profiler;

    toplevel_profiler.start();
    dt->inference_offline();
    toplevel_profiler.accumulate("inf-offline");
    offline_total_comm_tx_mb += ((double)comm_profiler.get_comm_tx_bytes()) / 1024.0 / 1024.0;

    comm_profiler.clear();
    toplevel_profiler.start();
    dt->inference_online(inf_data, result_label);
    toplevel_profiler.accumulate("inf-online");
    online_total_comm_tx_mb += ((double)comm_profiler.get_comm_tx_bytes()) / 1024.0 / 1024.0;
    
    if (gst_config["total_inference_stats"]) {
        printf("\n-----------Total Overhead of Inference--------\n");
        printf("total time for inference on %i instances (ms),%f\n", INF_COUNT_TOTAL, toplevel_profiler.get_elapsed_all());
        printf("Amortized inference time for single instance (ms),%f\n", toplevel_profiler.get_elapsed_all() / (double) INF_COUNT_TOTAL);
        printf("Inference: total tx comm (MB),%f\n", (offline_total_comm_tx_mb + online_total_comm_tx_mb));
        printf("\n-----------Overhead of offline and online--------\n");
        printf("Offline time (s),%f\n", toplevel_profiler.get_elapsed("inf-offline") / 1000.0);
        printf("Online time (s),%f\n", toplevel_profiler.get_elapsed("inf-online") / 1000.0);
        printf("Offline tx comm (MB),%f\n", offline_total_comm_tx_mb);
        printf("Online tx comm (MB),%f\n", online_total_comm_tx_mb);
    }
}

template<typename T, template<typename, typename...> typename Share>
void inference_simulation(DecisionTree<T, Share> *dt, std::ifstream &inf_data_file, std::map<std::string, int> &class_code_map)
{
    // std::vector<double> inf_data_vec(INF_COUNT_TOTAL * ATTRIBUTE_COUNT_TOTAL);
    std::vector<double> inf_data_vec_onehot(INF_COUNT_TOTAL * ATTRIBUTE_COUNT_TOTAL * 2);
    std::vector<double> inf_label_vec(INF_COUNT_TOTAL);
    if (!prepare_data_onehot(inf_data_file, inf_data_vec_onehot, inf_label_vec, class_code_map, INF_COUNT_TOTAL))
    {
        return;
    }

    std::vector<double> inf_data_vec_onehot_trans(inf_data_vec_onehot.size());
    for (int i = 0; i < INF_COUNT_TOTAL; i++)
    {
        for (int j = 0; j < ATTRIBUTE_COUNT_TOTAL * 2; j++)
        {
            inf_data_vec_onehot_trans[j * INF_COUNT_TOTAL + i] = inf_data_vec_onehot[i * ATTRIBUTE_COUNT_TOTAL * 2 + j];
        }
    }
    
    
    Share<T> inf_data(inf_data_vec_onehot.size());
    inf_data.setPublic(inf_data_vec_onehot_trans, false);
    Share<T> result_label(INF_COUNT_TOTAL);
    result_label.setPublic(inf_label_vec, false);

    
    dt->tree_level = TREE_DEPTH - 1;
    dt->tree.resize((1 << TREE_DEPTH) - 1);
    dt->tree.zero();
    dt->leafClass.resize(1 << (TREE_DEPTH - 1));
    dt->leafClass.zero();

    double offline_total_comm_tx_mb = 0.0;
    double online_total_comm_tx_mb = 0.0;
    comm_profiler.clear();
    Profiler toplevel_profiler;

    toplevel_profiler.start();
    dt->inference_offline();
    toplevel_profiler.accumulate("inf-offline");
    offline_total_comm_tx_mb += ((double)comm_profiler.get_comm_tx_bytes()) / 1024.0 / 1024.0;

    comm_profiler.clear();
    toplevel_profiler.start();
    dt->inference_online(inf_data, result_label);
    toplevel_profiler.accumulate("inf-online");
    double comm_time = comm_profiler.get_elapsed("comm-time");
    online_total_comm_tx_mb += ((double)comm_profiler.get_comm_tx_bytes()) / 1024.0 / 1024.0;
    
    printf("\n-----------Total Overhead of Inference Simulation--------\n");
    printf("total time for inference on %i instances (ms),%f\n", INF_COUNT_TOTAL, toplevel_profiler.get_elapsed_all());
    printf("Amortized inference time for single instance (ms),%f\n", toplevel_profiler.get_elapsed_all() / (double) INF_COUNT_TOTAL);
    printf("Inference: total tx comm (MB),%f\n", (offline_total_comm_tx_mb + online_total_comm_tx_mb));
    printf("\n-----------Overhead of offline and online Simulation--------\n");
    printf("total Offline time (ms),%f\n", toplevel_profiler.get_elapsed("inf-offline"));
    printf("Amortized Offline time for single instance (ms),%f\n", toplevel_profiler.get_elapsed("inf-offline") / (double) INF_COUNT_TOTAL);
    double total_online_time = toplevel_profiler.get_elapsed("inf-online");
    printf("total Online time (ms),%f\n", total_online_time);
    printf("Amortized Online time for single instance (ms),%f\n", total_online_time / (double) INF_COUNT_TOTAL);
    printf("Amortized Online Communication time (ms),%f\n", comm_time / (double) INF_COUNT_TOTAL);
    printf("Amortized Online Computation time (ms),%f\n", (total_online_time - comm_time) / (double) INF_COUNT_TOTAL);
    printf("Offline tx comm (MB),%f\n", offline_total_comm_tx_mb);
    printf("Online tx comm (MB),%f\n", online_total_comm_tx_mb);
}

template<typename T, template<typename, typename...> typename Share>
void PrintDT(DecisionTree<T, Share> *dt)
{
    std::cout << "Begin to print DT: " << std::endl;
    std::cout << "tree_depth = " << dt->tree_level + 1 << std::endl;

    DeviceData<T> reconstructedNodeFlag(NODE_COUNT);
    reconstruct(dt->nodeFlag, reconstructedNodeFlag);
    std::vector<double> hostNodeFlag(NODE_COUNT);
    copyToHost(reconstructedNodeFlag, hostNodeFlag, false);

    DeviceData<T> reconstructedTree(NODE_COUNT);
    reconstruct(dt->tree, reconstructedTree);
    std::vector<double> hostTree(NODE_COUNT);
    copyToHost(reconstructedTree, hostTree, false);

    int num_L = 1 << dt->tree_level;
    DeviceData<T> reconstructedLeaf(num_L);
    reconstruct(dt->leafClass, reconstructedLeaf);
    std::vector<double> hostLeaf(num_L);
    copyToHost(reconstructedLeaf, hostLeaf, false);
    
    int num_node = 0;
    for (int i = 0; i < dt->tree_level + 1; i++)
    {
        num_node = 1 << i;
        if (i == dt->tree_level)
        {
            for (int j = 0; j < num_node; j++)
            {   
                std::cout << "(" << (int) hostNodeFlag[j + num_node - 1] << ", " << (int) hostTree[j + num_node - 1] << ", " << (int) hostLeaf[j] << ") ";
            }
            std::cout << std::endl;
        }
        else
        {
            for (int j = 0; j < num_node; j++)
            {   
                std::cout << "(" << (int) hostNodeFlag[j + num_node - 1] << ", " << (int) hostTree[j + num_node - 1] << ") ";   
            }
            std::cout << std::endl;
        }
    }
}

std::vector<std::string> split_attributes(std::string line, char delim) {
    std::vector<std::string> arr;
    const char *start = line.c_str();
    bool instring = false;

    for (const char* p = start; *p; p++) {
        if (*p == '"') {
            instring = !instring;
        } else if (*p == delim && !instring) {
            arr.push_back(std::string(start, p-start));
            start = p + 1;
        }
    }

    arr.push_back(std::string(start)); // last field delimited by end of line instead of comma
    return arr;
}

std::vector<std::string> split(std::string str, std::string delim) {
    char* cstr = const_cast<char*>(str.c_str());
    char* current;
    vector<std::string> arr;
    current = strtok(cstr, delim.c_str());

    while (current != NULL) {
        arr.push_back(current);
        current = strtok(NULL, delim.c_str());
    }

    return arr;
}

void deleteObjects() {
	//close connection
	for (int i = 0; i < gst_config["num_parties"]; i++) {
		if (i != partyNum) {
			delete communicationReceivers[i];
			delete communicationSenders[i];
		}
	}
	delete[] communicationReceivers;
	delete[] communicationSenders;
	delete[] addrs;
}

