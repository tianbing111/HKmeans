#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <unistd.h>
#include <stdio.h>
#include "util/file_handler.h"
#include "util/read_file.h"
#include "util/utils.h"
#include "util/flat.h"
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
using namespace std;
void analyze()
{
    const char *Query_Path = "/home/tianbin/dataset/query.public.10K.u8bin";
    //const char *Centroids_Path = "/home/tianbin/smartann/HKmeans/result/centroids_100M_1GB";
    uint32_t number_centroids = 174;
    uint32_t temp;
    uint32_t cdim = 128;
    
    IOReader data_reader("/home/tianbin/smartann/HKmeans/result/centroids_100M_1GB");
    data_reader.read((char*)&temp, sizeof(uint32_t));
    data_reader.read((char*)&temp, sizeof(uint32_t));

    float* centroids_data = new float[number_centroids * cdim];
    data_reader.read((char*)centroids_data, number_centroids * cdim * sizeof(float));
    uint32_t number_query,  qdim;
    uint8_t *query_data = NULL;
    //read query
    get_bin_metadata(Query_Path, number_query, qdim);
    read_bin_file(Query_Path, query_data, number_query, qdim);
    //read centroids
    //get_bin_metadata(Centroids_Path, number_centroids, cdim);
    //read_bin_file(Centroids_Path, centroids_data, number_centroids, cdim);

    int nprobe = 20;
    std::unique_ptr<uint32_t[]> idx(new uint32_t[number_query * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[number_query * nprobe]);
    knn_1<CMax<float, uint32_t>, uint8_t, float> (
        query_data, 
        centroids_data, 
        number_query, 
        number_centroids, 
        qdim, nprobe, 
        coarse_dis.get(), 
        idx.get(), 
        L2sqr<const uint8_t, const float, float>);
    coarse_dis = nullptr;
    
    for(int i=0; i < nprobe; i++)
    {
        cout<<idx[i]<<'\t';
    }
    cout<<'\n';
    for(int i=0; i < nprobe; i++)
    {
        cout<<idx[i+20]<<'\t';
    }
    delete [] centroids_data;
    delete [] query_data;
}

int main()
{
    analyze();
    return 0;
}