// Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Mesh_Processor.h"
#include "Header.h"




int main0()
{
	string graph_path = "E:\\VS_Projects\\Mesh\\Test\\feature_back.pb";
	string python_path = "D:/Python";
	int coarsen_times = 2;
	int coarsen_level = 3;
	const int pt_num = 2000;
	const int K = 13; // 需要调整

	int* adj = new int[pt_num * K];
	memset(adj, 0, sizeof(int) * pt_num * K);
	int actual_pt_num = load_file("E:\\VS_Projects\\Mesh\\Test\\adj_f.txt", adj, K);

	float* x = new float[pt_num * 3];
	load_file("E:\\VS_Projects\\Mesh\\Test\\vertice_f.txt", x, 3);

	Mesh_Processor* mp = new Mesh_Processor(graph_path, Mesh_Processor::FEAT, false, python_path, "coarsening",
		coarsen_times, coarsen_level);



	float** output = new float* [8];
	for (int i = 0; i < 8; i++) {
		output[i] = new float[3];
	}

	mp->predict_feature(x, adj, actual_pt_num, 13, Mesh_Processor::LU,output);


	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 3; j++) {
			cout << output[i][j] << "  ";
		}
		cout << endl;
	}
	getchar();
	return 0;
}

