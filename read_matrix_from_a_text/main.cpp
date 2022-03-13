#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

int main() {

    // initialize string variables for reading in text file lines
    string line;
    stringstream ss;

    // initialize variables to hold the matrix
    vector < vector <float> > matrix;
    vector<float> row;

    // counter for characters in a text file line
    float i;

    // read in the file
    ifstream matrixfile ("matrix.txt");

    // read in the matrix file line by line
    // parse the file

    if (matrixfile.is_open()) {
        while (getline (matrixfile, line)) {

            // parse the text line with a stringstream
            // clear the string stream to hold the next line
            ss.clear();
            ss.str("");
            ss.str(line);
            row.clear();

            // parse each line and push to the end of the row vector
            // the ss variable holds a line of text
            // ss >> i puts the next character into the i variable.
            // the >> syntax is like cin >> some_value or cout << some_value
            // ss >> i is false when the end of the line is reached

            while(ss >> i) {
                row.push_back(i);

                if (ss.peek() == ',' || ss.peek() == ' ') {
                    ss.ignore();
                }
            }

            // push the row to the end of the matrix
            matrix.push_back(row);
        }

        matrixfile.close();

        // print out the matrix
        for (unsigned int row = 0; row < matrix.size(); row++) {
            for (unsigned int column = 0; column < matrix[row].size(); column++) {
                cout << matrix[row][column] << " " ;
            }
            cout << endl;
        }
    }

    else cout << "Unable to open file";

    return 0;
}
