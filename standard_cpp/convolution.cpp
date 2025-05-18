#include <vector>
#include <iostream>
using namespace std;

vector<vector<double>> convolution(const vector<vector<double>>& matrix, const vector<vector<double>>& kernel) {
    int ksize = kernel.size();
    int msize = matrix.size();
    vector<vector<double>> result(msize - ksize + 1, vector<double>(msize - ksize + 1, 0));
    for (int r = 0; r < msize - ksize + 1; r++) {
        for (int c = 0; c < msize - ksize + 1; c++) {
            for (int k = 0; k < ksize; k++) {
                for (int l = 0; l < ksize; l++) {
                    result[r][c] += matrix[r + k][c + l] * kernel[k][l];
                }
            }
        }
    }
    return result;
}

int main() {
    vector<vector<double>> matrix =  {
        {1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2},
        {3, 3, 3, 3, 3},
        {4, 4, 4, 4, 4},
        {5, 5, 5, 5, 5}
    };
    vector<vector<double>> kernel =  {
        {0, 0.2, 0},
        {0.2, 0.2, 0.2},
        {0, 0.2, 0}
    };

    vector<vector<double>> result = convolution(matrix, kernel);
    for (int r = 0; r < result.size(); r++) {
        for (int c = 0; c < result.size(); c++) {
            cout << result[r][c] << " ";
        }
        cout << endl;
    }
    return 0;
}
