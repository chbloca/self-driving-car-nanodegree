#include <iostream>
#include <vector>
#include <string>

using namespace std;

vector<double> p (5, 0.2);
vector<string> world = {"green", "red", "red", "green", "green"};
vector<string> measurements = {"red", "green"};
vector<int> motions (2, 1);
double pHit = 0.6;
double pMiss = 0.2;
double pExact = 0.8;
double pOvershoot = 0.1;
double pUndershoot = 0.1;

vector<double> sense(vector<double> p, string Z);
vector<double> move(vector<double> p, int U);
void print_probability(vector<double> p);

int main()
{
    for(int k = 0; k < measurements.size(); ++k){
        p = sense(p, measurements[k]);
        p = move(p, motions[k]);
    }
    print_probability(p);
    return 0;
}

vector<double> sense(vector<double> p, string Z){
    vector<double> q;
    bool hit;
    double s = 0;
    for(int i = 0; i < p.size(); ++i){
        hit = (Z == world[i]);
        q[i] = p[i] * pHit + (1 - hit) * pMiss;
        s += q[i];
    }
    for(int i = 0; i < q.size(); ++i)
        q[i] /= s;
    return q;
}


vector<double> move(vector<double> p, int U){
    vector<double> q;
    double s = 0;
    for(int i = 0; i < p.size(); ++i){
        if(i-U < 0)
            s = pExact * p[(i-U) + p.size()];
        else
            s = pExact * p[(i-U) % p.size()];
        if(i-U-1 < 0)
            s += pOvershoot * p[(i-U-1) + p.size()];
        else
            s += pOvershoot * p[(i-U-1) % p.size()];
        if(i-U+1)
            s += pUndershoot * p[((i-U+1) + p.size())];
        else
            s += pUndershoot * p[((i-U+1) % p.size())];
        q[i] = s;
    }
    return q;
}

void print_probability(vector<double> p){
    for(int i = 0; i < p.size(); ++i)
        cout << p[i] << " ";
    cout << endl;
}
