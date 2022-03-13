#include "headers/normalize.h"
using namespace std;

// OPTIMIZATION: Pass variable by reference
vector< vector<float> > normalize(vector< vector <float> > &grid) {

  	// OPTIMIZATION: Avoid declaring and defining 				// intermediate variables that are not needed.
	float total = 0.0;
	vector<float> newRow;

    for (int i = 0; i < grid.size(); i++)
        for (int j=0; j< grid[0].size(); j++)
            total += grid[i][j];

	vector< vector<float> > newGrid;

    for (int i = 0; i < grid.size(); i++) {
		newRow.clear();
        for (int j=0; j< grid[0].size(); j++) {
            newRow.push_back(grid[i][j] / total);
		}
		newGrid.push_back(newRow);
	}

	return newGrid;
}
