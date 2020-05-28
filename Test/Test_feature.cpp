// Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Mesh_Processor.h"
#include "Header.h"
#include "TeethDetect.h"
#include <time.h>




int main()
{
	string graph_path = "E:\\VS_Projects\\Mesh\\Test\\feature_back_res.pb";
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
	for (int i = 0; i < 5; i++) {
		clock_t start = clock();

		mp->predict_feature(x, adj, actual_pt_num, 13, Mesh_Processor::LU, output);
	
		clock_t end = clock();
		cout << i << "  th run time is: " <<end - start << endl<<endl;
}



	getchar();


///////  TeethDetect
//	int num, w, h;
//	float** coord = new float* [16];
//	for (int i = 0; i < 16; i++) {
//		coord[i] = new float[7];
//	}
//
//	TeethDetector* td = new TeethDetector(
//		"E:/TensorFlowCplusplus/TeethDetect/x64/Release/output_graph.pb");
//
//	td->detect("E:/TensorFlowCplusplus/TeethDetect/x64/Release/low.png", num, coord, w, h);
//	for (int i = 0; i < num; ++i) {
//	for (int j = 0; j < 7; ++j)
//		cout << coord[i][j] << "\t";
//	std::cout << std::endl;
//}
//
//	getchar();
	///////////////
	return 0;
}

