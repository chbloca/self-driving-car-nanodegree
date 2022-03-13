#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

vector< vector<int> > input_get_matrix(int a, int b);
vector< vector<int> > multiply(vector< vector<int> > matrix_1, vector< vector<int> > matrix_2);
void print_matrix(vector< vector<int> > matrix);
vector<int> get_row(vector< vector<int> > matrix, int row);
vector<int> get_column(vector< vector<int> > matrix, int column);
int dot_product(vector<int> vector_one, vector<int> vector_two);


int main(){
    int m, n, w, z;
    vector< vector<int> > result;
    cout << "Enter dimensions of matrix_1: " << endl;
    cout << "m: ";
    cin >> m;
    cout << "n: ";
    cin >> n;

    cout << "Enter dimensions of matrix_2: " << endl;
    cout << "w: ";
    cin >> w;
    cout << "z: ";
    cin >> z;

    vector< vector<int> > matrix_1;
    vector< vector<int> > matrix_2;

    matrix_1 = input_get_matrix(m, n);
    print_matrix(matrix_1);
    matrix_2 = input_get_matrix(w, z);
    print_matrix(matrix_1);

    result = multiply(matrix_1, matrix_2);
    print_matrix(result);


    return 0;
}

vector< vector<int> > input_get_matrix(int a, int b){
    int input;
    vector<int> row;
    vector < vector<int> > matrix;

    for(int i = 0; i < a; ++i){
        for(int j = 0; j < b; ++j){
            cout << "Elemento: [" << j << "," << i << "]: ";
            cin >> input;
            row.push_back(input);
        }
        matrix.push_back(row);
        row.clear();
    }
    return matrix;
}

vector<int> get_row(vector< vector<int> > matrix, int r){
    return matrix[r];
}

vector<int> get_column(vector< vector<int> > matrix, int c){
    vector<int> column;
    for(int i = 0; i < c; ++c)
        column.push_back(matrix[i][c]);
    return column;
}

int dot_product(vector<int> vector_one, vector<int> vector_two){
    int sum = 0;
    for(int i = 0; i < vector_one.size(); ++i)
        sum += vector_one[i] * vector_two[i];
    return  sum;
}

void print_matrix(vector< vector<int> > matrix){
    for(int i = 0; i < matrix.size(); i++){
        for(int j = 0; j < matrix[0].size(); j++)
            cout << matrix[i][j] << " ";
        cout << endl;
    }
}

vector< vector<int> > multiply(vector< vector<int> > matrix_1, vector< vector<int> > matrix_2){
    vector< vector<int> > result;
    vector<int> row_result;
    vector<int> row_A;
    vector<int> column_B;
    for(int i = 0; i < matrix_1.size(); ++i){
        for(int j = 0; j < matrix_2[0].size(); ++j){
            row_A = get_row(matrix_1, i);
            column_B = get_column(matrix_2, j);
            row_result.push_back(dot_product(row_A, column_B));
        }
        result.push_back(row_result);
        row_result.clear();
    }
}
