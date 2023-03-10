#pragma once

#include "distance.h"
#include "heap.h"
#include "system.h"
#include "utils.h"

#include <algorithm>
#include <omp.h>

// Data type: T1, T2
// Distance type: C::T
// ID type C::TI

template<class C, typename T1, typename T2>
void knn_1 (const T1 * x, // query_data
            const T2 * y, // centroids_data
            int64_t nx,  //query_number 
            int64_t ny,  //centroids_number 
            int64_t dim,
            int64_t k,
            typename C::T * value,  //dis
            typename C::TI * labels,  //ids
            Computer<T1, T2, typename C::T> comptuer  //dis metric
            )
{
    std::cout << "do knn_1 with nx = " << nx << ", ny = " << ny
              << ", k = " << k << std::endl;
#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
        auto *x_i = x + i * dim;
        auto *y_j = y;

        auto * __restrict val_ = value  + i * k;
        auto * __restrict ids_ = labels  + i * k;

        heap_heapify<C>(k, val_, ids_);

        for (int64_t j = 0; j < ny; j++) {
            auto disij = comptuer (x_i, y_j, dim);
            if (C::cmp(val_[0], disij)) {
                heap_swap_top<C>(k, val_, ids_, disij, j);
            }
            y_j += dim;
        }

        heap_reorder<C> (k, val_, ids_);
    }
}

template<class C, typename T1, typename T2>
void knn_2 (const T1 * x, // query
            const T2 * y, // base
            int64_t nx, int64_t ny, int64_t dim,
            int64_t k,
            typename C::T * value,
            typename C::TI * labels,
            Computer<T1, T2, typename C::T> comptuer)
{
    using DIS_TYPE = typename C::T;
    using ID_TYPE = typename C::TI;

    int64_t thread_max_num = omp_get_max_threads();
    int64_t l3_size = get_L3_Size();

    int64_t block_x = std::min(
        int64_t(l3_size / (dim * sizeof(T1) + thread_max_num * k * (sizeof(DIS_TYPE) + sizeof(ID_TYPE)))),
        nx);
    if (block_x == 0) {
        block_x = 1;
    }

    int64_t all_heap_size = block_x * k * thread_max_num;
    DIS_TYPE* value_global = new DIS_TYPE[all_heap_size];
    ID_TYPE* labels_global = new ID_TYPE[all_heap_size];

    for (int64_t x_from = 0, x_to; x_from < nx; x_from = x_to) {
        x_to = std::min(nx, x_from + block_x);
        int64_t size = x_to - x_from;
        int64_t thread_heap_size = size * k;

        // init heap
        heap_heapify<C>(all_heap_size, value_global, labels_global);

#pragma omp parallel for schedule(static)
        for (int j = 0; j < ny; j++) {
            int64_t thread_no = omp_get_thread_num();
            auto* y_j = y + j * dim;
            auto* x_i = x + x_from * dim;
            for (int64_t i = 0; i < size; i++) {
                DIS_TYPE disij = comptuer (x_i, y_j, dim);
                DIS_TYPE* val_ = value_global + thread_no * thread_heap_size + i * k;
                ID_TYPE* ids_ = labels_global + thread_no * thread_heap_size + i * k;
                if (C::cmp(val_[0], disij)) {
                    heap_swap_top<C> (k, val_, ids_, disij, j);
                }
                x_i += dim;
            }
        }

        // merge heap
        for (int64_t t = 1; t < thread_max_num; t++) {
            for (int64_t i = 0; i < size; i++) {
                DIS_TYPE* __restrict value_x = value_global + i * k;
                ID_TYPE* __restrict labels_x = labels_global + i * k;
                DIS_TYPE* value_x_t = value_x + t * thread_heap_size;
                ID_TYPE* labels_x_t = labels_x + t * thread_heap_size;
                for (int j = 0; j < k; j++) {
                    if (C::cmp(value_x[0], value_x_t[j])) {
                        heap_swap_top<C> (k, value_x, labels_x, value_x_t[j], labels_x_t[j]);
                    }
                }
            }
        }

        // sort
        for (int64_t i = 0; i < size; i++) {
            DIS_TYPE * value_x = value_global + i * k;
            ID_TYPE * labels_x = labels_global + i * k;
            heap_reorder<C> (k, value_x, labels_x);
        }

        // copy result
        memcpy(value + x_from * k, value_global, thread_heap_size * sizeof(DIS_TYPE));
        memcpy(labels + x_from * k, labels_global, thread_heap_size * sizeof(ID_TYPE));
    }

    delete[] value_global;
    delete[] labels_global;
}

