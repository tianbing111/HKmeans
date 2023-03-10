#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <functional>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <sys/stat.h>

#include "file_handler.h" 
#include "distance.h"
#include "defines.h"


template <typename T1, typename T2, typename R>
using Computer = std::function<R(const T1*, const T2*, int n)>;


template<typename T1, typename T2, typename R>
Computer<T1, T2, R> select_computer(MetricType metric_type) {
    switch (metric_type) {
        case MetricType::L2:
            return L2sqr<const T1, const T2, R>;
            break;
        case MetricType::IP:
            return IP<const T1, const T2, R>;
            break;
    }
}

inline void get_bin_metadata(const std::string& bin_file, uint32_t& nrows, uint32_t& ncols) {
    std::ifstream reader(bin_file, std::ios::binary);
    reader.read((char*) &nrows, sizeof(uint32_t));
    reader.read((char*) &ncols, sizeof(uint32_t));
    reader.close();
    std::cout << "get meta from " << bin_file << ", nrows = " << nrows << ", ncols = " << ncols << std::endl;
}

inline void set_bin_metadata(const std::string& bin_file, const uint32_t& nrows, const uint32_t& ncols) {
    std::ofstream writer(bin_file, std::ios::binary | std::ios::in);
    writer.seekp(0);
    writer.write((char*) &nrows, sizeof(uint32_t));
    writer.write((char*) &ncols, sizeof(uint32_t));
    writer.close();
    std::cout << "set meta to " << bin_file << ", nrows = " << nrows << ", ncols = " << ncols << std::endl;
}

inline void load_meta_impl(const std::string& index_path,
               std::vector<std::vector<uint32_t>>& meta,
               const int K1) {
    for (int i = 0; i < K1; i ++) {
        std::ifstream reader(index_path + CLUSTER + std::to_string(i) + META + BIN, std::ios::binary);
        uint32_t nmeta, dmeta;
        reader.read((char*)&nmeta, sizeof(uint32_t));
        reader.read((char*)&dmeta, sizeof(uint32_t));
        assert(1 == dmeta);
        meta[i].resize(nmeta);
        reader.read((char*)meta[i].data(), (uint64_t)nmeta * dmeta * sizeof(uint32_t));
        reader.close();
    }
}

template<typename T>
inline void write_bin_file(const std::string& file_name, T* data, uint32_t n,
                    uint32_t dim) {
    assert(data != nullptr);
    std::ofstream writer(file_name, std::ios::binary);

    writer.write((char*)&n, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    writer.write((char*)data, sizeof(T) * (uint64_t)n * dim);

    writer.close();
    std::cout << "write binary file to " << file_name << " done in ... seconds, n = "
              << n << ", dim = " << dim << std::endl;
}

template<typename T>
inline void read_bin_file(const std::string& file_name, T*& data, uint32_t& n,
                    uint32_t& dim) {
    std::ifstream reader(file_name, std::ios::binary);

    reader.read((char*)&n, sizeof(uint32_t));
    reader.read((char*)&dim, sizeof(uint32_t));
    if (data == nullptr) {
        data = new T[(uint64_t)n * (uint64_t)dim];
    }
    reader.read((char*)data, sizeof(T) * (uint64_t)n * dim);

    reader.close();
    std::cout << "read binary file from " << file_name << " done in ... seconds, n = "
              << n << ", dim = " << dim << std::endl;
}

template<typename T>
void reservoir_sampling(const std::string& data_file, const size_t sample_num, T* sample_data) {
    assert(sample_data != nullptr);
    std::random_device rd;
    auto x = rd();
    std::mt19937 generator((unsigned) x);
    uint32_t nb, dim;
    size_t ntotal, ndims;
    IOReader reader(data_file);
    reader.read((char*)&nb, sizeof(uint32_t));
    reader.read((char*)&dim, sizeof(uint32_t));
    ntotal = nb;
    ndims = dim;
    std::unique_ptr<T[]> tmp_buf = std::make_unique<T[]>(ndims);
    for (size_t i = 0; i < sample_num; i ++) {
        auto pi = sample_data + ndims * i;
        reader.read((char*) pi, ndims * sizeof(T));
    }
    for (size_t i = sample_num; i < ntotal; i ++) {
        reader.read((char*)tmp_buf.get(), ndims * sizeof(T));
        std::uniform_int_distribution<size_t> distribution(0, i);
        size_t rand = (size_t)distribution(generator);
        if (rand < sample_num) {
            memcpy((char*)(sample_data + ndims * rand), tmp_buf.get(), ndims * sizeof(T));
        }
    }
}

template<typename T>
void reservoir_sampling_residual(
        const std::string& output_path,
        const std::vector<std::vector<uint32_t> >& metas,
        const float* ivf_cen,
        const uint32_t dim, 
        const uint32_t sample_num,
        T* sample_data,
        float* sample_ivf_cen,
        const int K1) {
    assert(sample_ivf_cen != nullptr);
    assert(sample_data != nullptr);

    std::random_device rd;
    std::mt19937 generator((unsigned)(rd()));
    uint32_t cluster_size, cluster_dim, global_cnt = 0;
    std::vector<T> cluster_data;
    std::vector<uint32_t> ivf_cen_offsets(sample_num);

    uint32_t i = 0; // default choose

    std::string data_file = output_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;

    IOReader data_reader(data_file);
    data_reader.read((char*)&cluster_size, sizeof(uint32_t));
    data_reader.read((char*)&cluster_dim, sizeof(uint32_t));
    assert(cluster_dim == dim);

    // read vectors in each cluster
    const uint64_t total_size = cluster_size * cluster_dim;
    cluster_data.resize(total_size);
    data_reader.read((char*)(cluster_data.data()), total_size * sizeof(T));

    const T* vec = cluster_data.data();
    for (size_t j = 0; j < metas[i].size(); ++j) {
        for (uint32_t k = 0; k < metas[i][j]; ++k) {
            // deal with the situation when one bucket is not enough
            // for filling the resulting array
            if (global_cnt < sample_num) {
                memcpy(sample_data + cluster_dim * global_cnt, vec, cluster_dim * sizeof(T));
                ivf_cen_offsets[global_cnt] = j;
            } else {
                std::uniform_int_distribution<size_t> distribution(0, global_cnt);
                size_t rand = (size_t)distribution(generator);
                if (rand < sample_num) {
                    memcpy(sample_data + cluster_dim * rand, vec, cluster_dim * sizeof(T));
                    ivf_cen_offsets[rand] = j;
                }
            }

            vec += cluster_dim;
            ++global_cnt;
        }
    }

    for (uint32_t i = 0; i < sample_num; ++i) {
        memcpy(
            sample_ivf_cen + i * dim,
            ivf_cen + ivf_cen_offsets[i] * dim,
            dim * sizeof(float));
    }
}

template<typename T>
void random_sampling_k2(
        const T* data,
        const int64_t data_size,
        const int64_t dim,
        const int64_t sample_size,
        T* sample_data,
        int64_t seed = 1234
) {
    std::vector<int> perm(data_size);
    for (int64_t i = 0; i < data_size; i++) {
        perm[i] = i;
    }
    std::shuffle(perm.begin(), perm.end(), std::default_random_engine(seed));
    for (int64_t i = 0; i < sample_size; i++) {
        memcpy(sample_data + i * dim, data + perm[i] * dim,  dim * sizeof(T));
    }
    return ;
}

inline uint32_t gen_global_block_id(const uint32_t cid, const uint32_t bid) {
    uint32_t ret = 0;
    ret |= (cid & 0xff);
    ret <<= 24;
    ret |= (bid & 0xffffff);
    return ret;
}

inline void parse_global_block_id(uint32_t id, uint32_t& cid, uint32_t& bid) {
    bid = (id & 0xffffff);
    id >>= 24;
    cid = (id & 0xff);
    return ;
}


inline uint64_t gen_id(const uint32_t cid, const uint32_t bid, const uint32_t off) {
    uint64_t ret = 0;
    ret |= (cid & 0xff);
    ret <<= 24;
    ret |= (bid & 0xffffff);
    ret <<= 32;
    ret |= (off & 0xffffffff);
    return ret;
}

inline void parse_id(uint64_t id, uint32_t& cid, uint32_t& bid, uint32_t& off) {
    off = (id & 0xffffffff);
    id >>= 32;
    bid = (id & 0xffffff);
    id >>= 24;
    cid = (id & 0xff);
}

inline uint64_t gen_refine_id(const uint32_t cid, const uint32_t offset, const uint32_t queryid) {
    uint64_t ret = 0;
    ret |= (cid & 0x000000ff);
    ret <<= 32;
    ret |= (offset & 0xffffffff);
    ret <<= 24;
    ret |= (queryid & 0x00ffffff);
    return ret;
}

inline void parse_refine_id(uint64_t id, uint32_t& cid, uint32_t& offset, uint32_t& queryid) {
    queryid = (id & 0x00ffffff);
    id >>= 24;
    offset = (id & 0xffffffff);
    id >>= 32;
    cid = (id & 0x000000ff);
}

inline MetricType get_metric_type_by_name(const std::string& mt_name) {
    if (mt_name == std::string("L2"))
        return MetricType::L2;
    if (mt_name == std::string("IP"))
        return MetricType::IP;
    return MetricType::None;
}

inline QuantizerType get_quantizer_type_by_name(const std::string& s) {
    if (s == "PQ") {
        return QuantizerType::PQ;
    } else if (s == "PQRes") {
        return QuantizerType::PQRES;
    }
    return QuantizerType::None;
}

template<typename DISTT, typename IDT>
void print_vec_id_dis(std::vector<std::vector<std::pair<IDT, DISTT>>>& v, const std::string msg) {
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    std::cout << msg << std::endl;
    for (auto i = 0; i < v.size(); i ++) {
        for (auto j = 0; j < v[i].size(); j ++) {
            std::cout << "(" << v[i][j].first << ", " << v[i][j].second << ") ";
        }
        std::cout << std::endl;
    }
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
}

template<typename DISTT, typename IDT>
void read_sift(std::vector<std::vector<std::pair<IDT, DISTT>>>& v, const std::string& gt_file, uint32_t& nq, uint32_t& topk) {
    nq = 100; // default value 

    std::ifstream gin(gt_file, std::ios::binary);
    uint32_t sz;
    v.resize(nq);
    for (auto i = 0; i < nq; i ++) {
        gin.read((char*)&sz, sizeof(sz));
        v[i].resize(sz);
        for (auto j = 0; j < sz; j ++) {
            gin.read((char*)&v[i][j].first, sizeof(IDT));
            gin.read((char*)&v[i][j].second, sizeof(DISTT));
        }
    }
    topk = sz;
    gin.close();
}

template<typename FILE_DISTT, typename FILE_IDT, typename DISTT, typename IDT>
void read_comp(std::vector<std::vector<std::pair<IDT, DISTT>>>& v, const std::string& gt_file, uint32_t& nq, uint32_t& topk) {
    std::ifstream gin(gt_file, std::ios::binary);

    gin.read((char*)&nq, sizeof(uint32_t));
    gin.read((char*)&topk, sizeof(uint32_t));

    v.resize(nq);
    FILE_IDT t_id;
    for (uint32_t i = 0; i < nq; ++i) {
        v[i].resize(topk);
        for (uint32_t j = 0; j < topk; ++j) {
            gin.read((char*)&t_id, sizeof(FILE_IDT));
            v[i][j].first = static_cast<IDT>(t_id);
        }
    }

    FILE_DISTT t_dist;
    for (uint32_t i = 0; i < nq; ++i) {
        for (uint32_t j = 0; j < topk; ++j) {
            gin.read((char*)&t_dist, sizeof(FILE_DISTT));
            v[i][j].second = static_cast<DISTT>(t_dist);
        }
    }
    gin.close();
}

// TODO: need a is_range_search argument from top-level
template<typename FILE_IDT, typename IDT>
void read_comp_range_search(
        std::vector<std::vector<IDT>>& v,
        const std::string& gt_file,
        int32_t& nq,
        int32_t& total_res) {
    std::ifstream gin(gt_file, std::ios::binary);

    gin.read((char*)&nq, sizeof(int32_t));
    gin.read((char*)&total_res, sizeof(int32_t));
    
    v.resize(nq);
    int32_t n_results_per_query;
    for (int i = 0; i < nq; ++i) {
        gin.read((char*)&n_results_per_query, sizeof(int32_t));
        v[i].resize(n_results_per_query);
    }

    FILE_IDT t_id;
    for (uint32_t i = 0; i < nq; ++i) {
        for (uint32_t j = 0; j < v[i].size(); ++j) {
            gin.read((char*)&t_id, sizeof(FILE_IDT));
            v[i][j] = static_cast<IDT>(t_id);
        }
    }

    gin.close();
}

template<typename DISTT, typename IDT>
void recall(const std::string& groundtruth_file, const std::string& answer_file, MetricType metric_type, bool use_comp_format = true, bool cmp_id = false) {
    std::cout << "recall parammeters:" << std::endl;
    std::cout << " groundtruth_file: " << groundtruth_file
              << " answer_file: " << answer_file
              << " metric_type: " << (int)metric_type
              << " use_comp_format: " << use_comp_format
              << " cmp_id: " << cmp_id
              << std::endl;

    uint32_t gt_nq, gt_topk, answer_nq, answer_topk;

    std::vector<std::vector<std::pair<IDT, DISTT>>> groundtruth;
    if (use_comp_format) {
        read_comp<float, uint32_t, DISTT, IDT>(groundtruth, groundtruth_file, gt_nq, gt_topk);
    } else {
        read_sift<DISTT, IDT>(groundtruth, groundtruth_file, gt_nq, gt_topk);
    }

    // print_vec_id_dis<DISTT, IDT>(groundtruth, "show groundtruth:");

    std::vector<std::vector<std::pair<IDT, DISTT>>> resultset;
    if (use_comp_format) {
        read_comp<DISTT, IDT, DISTT, IDT>(resultset, answer_file, answer_nq, answer_topk);
    } else {
        read_sift<DISTT, IDT>(resultset, answer_file, answer_nq, answer_topk);
    }

    if (gt_nq != answer_nq || gt_topk < answer_topk) { // || gt_topk != answer_topk) {
        std::cerr << "Grountdtruth parammeters does not match. GT nq " << gt_nq
        << "(" << answer_nq << "), topk " << gt_topk << "(" << answer_topk << ")" << std::endl;
        return ;
    }

    // print_vec_id_dis<DISTT, IDT>(resultset, "show resultset:");

    int max_recall, min_recall;
    // recall statistics
    std::vector<int> border = {0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100};
    int recall0 = 0;
    int recall100 = 0;
    std::vector<int> recall_hist(border.size(), 0);
    std::vector<int> recalls(answer_nq);

    int tot_cnt = 0;
    std::cout << "recall@" << answer_topk << " between groundtruth file:"
              << groundtruth_file << " and answer file:"
              << answer_file << " is:" << std::endl;
    if (cmp_id) {
        for (auto i = 0; i < answer_nq; i ++) {
            std::unordered_map<IDT, bool> hash;
            int cnti = 0;
            for (auto j = 0; j < answer_topk; j ++) {
                hash[groundtruth[i][j].first] = true;
            }
            for (auto j = 0; j < answer_topk; j ++) {
                if (hash.find(resultset[i][j].first) != hash.end())
                    cnti ++;
            }
            recalls[i] = cnti;
            tot_cnt += cnti;
        }
        std::cout << "avg recall@" << answer_topk << " = " << ((double)(tot_cnt)) / answer_topk / answer_nq * 100 << "%." << std::endl;
    } else {
        if (MetricType::L2 == metric_type) {
            for (auto i = 0; i < answer_nq; i ++) {
                int cnti = 0;
                for (auto j = 0; j < answer_topk; j ++) {
                    if (resultset[i][j].second <= groundtruth[i][answer_topk - 1].second)
                        cnti ++;
                }
                recalls[i] = cnti;
                tot_cnt += cnti;
                // std::cout << "query " << i << " recall@" << answer_topk << " is: " << ((double)(cnti)) / answer_topk * 100 << "%." << std::endl;
            }
            std::cout << "avg recall@" << answer_topk << " = " << ((double)(tot_cnt)) / answer_topk / answer_nq * 100 << "%." << std::endl;
        } else if (MetricType::IP == metric_type) {
            for (auto i = 0; i < answer_nq; i ++) {
                int cnti = 0;
                for (auto j = 0; j < answer_topk; j ++) {
                    if (resultset[i][j].second >= groundtruth[i][answer_topk - 1].second)
                        cnti ++;
                }
                recalls[i] = cnti;
                tot_cnt += cnti;
                // std::cout << "query " << i << " recall@" << answer_topk << " is: " << ((double)(cnti)) / answer_topk * 100 << "%." << std::endl;
            }
            std::cout << "avg recall@" << answer_topk << " = " << ((double)(tot_cnt)) / answer_topk / answer_nq * 100 << "%." << std::endl;
        } else {
            std::cout << "invalid metric type: " << (int)metric_type << std::endl;
        }
    }

    for (auto i = 0; i < answer_nq; i ++) {
        if (recalls[i] == 0) {
            recall0 ++;
            continue;
        }
        if (recalls[i] == answer_topk) {
            recall100 ++;
            continue;
        }
        for (auto j = 0; j < border.size() - 1; j ++) {
            if (recalls[i] * 100 >= border[j] * answer_topk && recalls[i] * 100 < border[j + 1] * answer_topk) {
                recall_hist[j] ++;
                break;
            }
        }
    }
    int check_sum = recall0 + recall100;
    std::cout << "show more details about recall histogram:" << std::endl;
    std::cout << "recall@" << answer_topk << " in range [0, 0]: " << recall0 << std::endl;
    for (auto i = 0; i < border.size() - 1; i ++) {
        std::cout << "recall@" << answer_topk << " in range [" << border[i] << ", " <<  border[i + 1] << "): " << recall_hist[i] << std::endl;
        check_sum += recall_hist[i];
    }
    std::cout << "recall@" << answer_topk << " in range [100, 100]: " << recall100 << std::endl;
    std::cout << "check sum recall: " << check_sum << ", which should equal nq: " << answer_nq << std::endl;
}

inline uint64_t fsize(std::string& filename) {
    struct stat st;
    stat(filename.c_str(), &st);
    return st.st_size;
}

template<typename T>
bool cmp_vec(const T* x, const T* y, uint32_t dim) {
    for (auto i = 0; i < dim; ++i) {
        if (x[i] != y[i]) return false;
    }
    return true;
}