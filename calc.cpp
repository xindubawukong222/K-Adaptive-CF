

#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;


class Worker {
public:

    double* result;
    double time;
    vector<pair<int, double> > aa[1400000];

    Worker() {
        result = NULL;
        time = 0;
    }

    void init(int len, int* row, int* col, double* val) {
        for (int i = 0; i < len; i++) {
            aa[col[i]].push_back(make_pair(row[i], val[i]));
        }
    }

    void work(int m, int cnt, double* relation_, int* row, int* col, double* tmp, int A_len, int* A_row, int* A_col, double* A_val, bool reset) {
        if (reset) {
            init(A_len, A_row, A_col, A_val);
        }
        if (result != NULL) {
            delete[] result;
        }
        result = new double[cnt];
        for (int jj = 0; jj < cnt; jj++) {
            result[jj] = -1;
            int movie = row[jj];
            if (tmp[movie] < 1e-9) continue;
            int user = col[jj];
            vector<pair<int, double> >& not0 = aa[user];
            int len = not0.size();
            if (len == 0) continue;
            double sim_min = 1e18;
            double sim_max = -1e18;
            double* rrr = relation_ + movie * m;
            for (int i = 0; i < len; i++) {
                double tmp = rrr[not0[i].first];
                if (tmp > sim_max) sim_max = tmp;
                if (tmp < sim_min) sim_min = tmp;
            }
            double sum_sim = 0;
            double sum_score = 0;
            for (int i = 0; i < len; i++) {
                double tmp = rrr[not0[i].first];
                if (len > 1) {
                    tmp = (tmp - sim_min) / (sim_max - sim_min);
                    if (tmp < 0.5) tmp = 0;
                }
                sum_sim += tmp;
                sum_score += tmp * not0[i].second;
            }
            result[jj] = sum_score / sum_sim;
        }
    }

    double* get_result() {
        return result;
    }

    static Worker* get_worker() {
        static Worker worker;
        return &worker;
    }
};


extern "C" {

double* get_result_(int m, int cnt, double* relation_, int* row, int* col, double* tmp, int A_len, int* A_row, int* A_col, double* A_val, bool reset) {
    Worker* worker = Worker::get_worker();
    worker->work(m, cnt, relation_, row, col, tmp, A_len, A_row, A_col, A_val, reset);
    return worker->get_result();
}

}