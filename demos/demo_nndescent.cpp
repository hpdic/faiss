/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexNNDescent.h>

#include <iostream>

using namespace std::chrono;

int main(void) {

    printf("HPDIC MOD -- verified by Din Djarin \n");

    // dimension of the vectors to index
    int d = 64;
    int K = 64;
    K *= 10; // DFZ: let's enlarge the neighbors

    // size of the database we plan to index
    size_t nb = 10000;
    nb *= 10; // DFZ: make it larger

    std::mt19937 rng(12345);

    // make the index object and train it
    faiss::IndexNNDescentFlat index(d, K, faiss::METRIC_L2);
    index.nndescent.S = 10;
    index.nndescent.R = 32;
    index.nndescent.L = K;
    index.nndescent.iter = 10;
    index.verbose = true;

    // generate labels by IndexFlat
    faiss::IndexFlat bruteforce(d, faiss::METRIC_L2);

    std::vector<float> database(nb * d);
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = rng() % 1024;
    }

    { // populating the database
        index.add(nb, database.data());
        bruteforce.add(nb, database.data());
    }

    size_t nq = 1000;

    { // searching the database
        printf("Searching ...\n");
        index.nndescent.search_L = 50;

        std::vector<float> queries(nq * d);
        for (size_t i = 0; i < nq * d; i++) {
            queries[i] = rng() % 1024;
        }

        int k = 5;
        k *= 10; // DFZ: again let's enlarge the search scope
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<faiss::idx_t> gt_nns(k * nq);
        std::vector<float> dis(k * nq);

        auto start = high_resolution_clock::now();
        index.search(nq, queries.data(), k, dis.data(), nns.data());
        auto end = high_resolution_clock::now();
        auto t1 = duration_cast<microseconds>(end - start).count();

        // find exact kNNs by brute force search
        start = high_resolution_clock::now();
        bruteforce.search(nq, queries.data(), k, dis.data(), gt_nns.data());
        end = high_resolution_clock::now();
        auto t2 = duration_cast<microseconds>(end - start).count();

        int recalls = 0;
        for (size_t i = 0; i < nq; ++i) {
            for (int n = 0; n < k; n++) {
                for (int m = 0; m < k; m++) {
                    if (nns[i * k + n] == gt_nns[i * k + m]) {
                        recalls += 1;
                    }
                }
            }
        }
        float recall = 1.0f * recalls / (k * nq);
        int qps = nq * 1.0f * 1000 * 1000 / t1;

        printf("Recall@%d: %f, QPS: %d\n", k, recall, qps);

        std::cout << "ANN search time: " << t1 / 1000.0 << " ms" << std::endl;
        std::cout << "ENN search time: " << t2 / 1000.0 << " ms" << std::endl;
    }
}
