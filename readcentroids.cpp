#include <iostream>
#include <fstream>
using namespace std;
int main(int argc, char** argv)
{
    ifstream inF;
    inF.open("/home/tianbin/smartann/BBAnn/release/test/centroids", std::ifstream::binary);
    float centroids;
    inF.read(reinterpret_cast<char *>(&centroids),sizeof(float));
    inF.read(reinterpret_cast<char *>(&centroids),sizeof(float));
    int count=0;
    int sum = 0;
    while ((inF.read(reinterpret_cast<char *>(&centroids),sizeof(float))))
    {
        cout<<centroids<<" ";
        count++;
        if(count==128)
        {
            cout<<'\n';
            count = 0;
            sum++;
        }
    }
    cout<<sum;
    inF.close();

    return 0;  
}