// Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Mesh_Processor.h"
#include "Header.h"
#include <time.h>



int main()
{
	string graph_path = "E:/VS_Projects/Mesh/Test/area.pb";
	string python_path = "D:/Python";
	const int pt_num = 20000;
	const int K = 30; // ��Ҫ����
	int* adj = new int[pt_num * K];
	memset(adj, 0, sizeof(int) * pt_num * K);
	int actual_pt_num = load_file(
		"adj.txt",
		adj, K);

	//int actual_pt_num = load_file(
	//	"F:/ProjectData/mesh_feature/Case_debug/Case/low/ChristianLagos-8toothL 00/DownArchL/tooth4/adj.txt",
	//	adj, K);

	float* x = new float[pt_num * 3];
	load_file(
		"vertice.txt",
		x, 3);

	//load_file(
	//	"F:/ProjectData/mesh_feature/Case_debug/Case/low/ChristianLagos-8toothL 00/DownArchL/tooth4/vertice.txt",
	//	x, 3);

	int c_levels[] = { 2,2,3,3 };
	Area_Processor* mp = new Area_Processor(graph_path, python_path, "coarsening", 4, c_levels);


	for (int i = 0; i < 3; i++) {
		clock_t end0 = clock();
		int size;
		int* output = mp->predict(x, adj, actual_pt_num, K, PartID::LU, size);
		clock_t end1 = clock();
		cout << "once time: " << end1 - end0 << "\n\n\n";
		if (size < 100) {
			cout << "error!!!!!!!!!!!!\n";
			getchar();
		}
	}
#if defined(AI_DEBUG)

		float loss = mp->get_loss(label_path, feat_num, output_loss);
		cout << "loss =" << loss << endl;
#endif

	getchar();


	return 0;
}

