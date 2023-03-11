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
#include <unordered_set>
using namespace std;
void analyze_power_law()
{
    const char *Query_Path = "/home/tianbin/dataset/query.public.10K.u8bin";
    //const char *Centroids_Path = "/home/tianbin/smartann/HKmeans/result/centroids_100M_1GB";
    uint32_t number_centroids = 90;
    uint32_t temp;
    uint32_t cdim = 128;
    
    IOReader data_reader("/home/tianbin/smartann/HKmeans/result/centroids_100M_2GB");
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

    int nprobe = 30;
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
    
    //record count of centroids
    /*
    int *count = new int[number_centroids]();
    int win_size = 100;
    for(int i=0; i < win_size*nprobe; i++)
    {
        count[idx[i]]++;
    }
    for(int j = 0; j < number_centroids; j++)
    {
        cout<<count[j]<<'\t';
    }
    */
   
    ofstream query_file("query_to_centroids.txt");
    int temp1 = 0;
    for(int i = 0; i < number_query * nprobe;i++)
    {
        query_file << idx[i] << "\t";
        temp1++;
        while(temp1 == nprobe)
        {
            query_file << "\n";
            temp1 = 0;
        }
    }
    query_file.close();
    
    int *count = new int[number_centroids]();
    int win_size = 10000;
    for(int i=0; i < win_size*nprobe; i++)
    {
        count[idx[i]]++;
    }
    ofstream OutFile("centroids_count.txt");
    for(int i = 0; i < number_centroids; i++)
    {
        OutFile << count[i] <<"\n";
    }
    OutFile.close();
    delete [] centroids_data;
    delete [] query_data;
    delete [] count;
}

bool hasDuplicate(int arr1[], int len1, int arr2[], int len2)
{
    unordered_set<int> set1(arr1, arr1 + len1);
    for(int i = 0; i < len2; i++)
    {
        if(set1.find(arr2[i]) != set1.end())
        {
            return true;
        }
    }
    return false;
}
void analy_query_locality()
{
    uint32_t number_centroids = 174;
    int nprobe = 34;
    uint32_t temp;
    uint32_t cdim = 128;
    uint32_t number_query,  qdim;
    uint8_t *query_data = NULL;
    float* centroids_data = new float[number_centroids * cdim];
    const char *Query_Path = "/home/tianbin/dataset/query.public.10K.u8bin";
    get_bin_metadata(Query_Path, number_query, qdim);
    read_bin_file(Query_Path, query_data, number_query, qdim);
    IOReader data_reader("/home/tianbin/smartann/HKmeans/result/centroids_100M_1GB");
    data_reader.read((char*)&temp, sizeof(uint32_t));
    data_reader.read((char*)&temp, sizeof(uint32_t));
    data_reader.read((char*)centroids_data, number_centroids * cdim * sizeof(float));

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

    int windows = 2500;

    //std::vector<std::vector<int>> query_to_centroids(windows,std::vector<int>(nprobe));
    int query_to_centroids[windows][nprobe];
    //get query_to_centroids within windows
    for(int i = 0; i < windows; i++)
    {
        for(int j = 0; j < nprobe; j++)
        {
            query_to_centroids[i][j] = idx[i*nprobe + j];
        }
    }
    int *count = new int[windows]();
    int record = 0;
    for(int i = 0; i < windows; i++)
    {
        for(int j = 0; j < windows; j++)
        {
            if(hasDuplicate(query_to_centroids[i], nprobe, query_to_centroids[j], nprobe) && (i != j))
            {
                count[i]++;
            }
        }
    }
    double sum = 0;
    for(int i = 0; i < windows; i++)
    {
        sum += count[i] / (double)windows;
    }
    double avg = sum/windows;
    cout<<"avgery value:"<<avg;
    delete [] count;
    delete [] centroids_data;
    delete [] query_data;

}

int main()
{
    analy_query_locality();
    return 0;
}