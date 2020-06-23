// Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Mesh_Processor.h"
#include "Header.h"
#include <time.h>



int main()
{
	string graph_path = "E:/VS_Projects/Mesh/Test/feature_incisor.pb";
	string python_path = "D:/Python";
	const int pt_num = 2000;
	const int K = 13; // ��Ҫ����
	const int feat_num = 4;
	int* adj = new int[pt_num * K];
	memset(adj, 0, sizeof(int) * pt_num * K);
	int actual_pt_num = load_file(
		"F:/ProjectData/mesh_feature/Case_debug/Case/low/TestPatientL 00/DownArchR/tooth7/adj.txt", 
		adj, K);

	float* x = new float[pt_num * 3];
	load_file(
		"F:/ProjectData/mesh_feature/Case_debug/Case/low/TestPatientL 00/DownArchR/tooth7/vertice.txt",
		x, 3);

	Feature_Processor* mp = new Feature_Processor(graph_path, false, python_path, "coarsening");

	const char label_path[] =
		"F:/ProjectData/mesh_feature/Case_debug/Case/low/TestPatientL 00/DownArchR/tooth7/feature.txt";


	float** output = new float* [feat_num];
	for (int i = 0; i < feat_num; i++) {
		output[i] = new float[3];
	}


	float output_loss[feat_num * 3];

	//for (int i = 0; i < 5; i++) {
		clock_t start = clock();

		mp->predict(x, adj, actual_pt_num, K, Feature_Processor::RD, output);


#if defined(DEBUG)

		for (int m = 0; m < feat_num; m++) {
			for (int n = 0; n < 3; n++) {
				output_loss[3*m+n] = output[m][n];
				cout << output[m][n] << "  ";
			}
			cout << endl;
		}
		float loss = mp->get_loss(label_path, feat_num, output_loss);
		cout << "loss =" << loss << endl;
#endif

		clock_t end = clock();


		//cout << i << "  th run time is: " <<end - start << endl<<endl;
//}



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

