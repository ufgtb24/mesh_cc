// Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Mesh_Processor.h"
#include "Header.h"
#include "TeethDetect.h"
#include <time.h>




int main()
{
	string graph_path = "E:/VS_Projects/Mesh/pb/feature_premollar.pb";
	const int feature_num = 6;
	string case_root = 
		"E:/VS_Projects/Mesh/Case/ChristianLagos-8toothU 00/UpArchR/tooth5/";
	const char label_path[] = 
		"E:/VS_Projects/Mesh/Case/ChristianLagos-8toothU 00/UpArchR/tooth5/feature.txt"; 

	string python_path = "D:/Python";
	int coarsen_times = 2;
	int coarsen_level = 3;
	const int pt_num = 2000;
	const int K = 13; // 需要调整
	float x[pt_num * 3];
	load_file(case_root+"vertice.txt", x, 3);

	int adj[pt_num * K];
	//int* adj = new int[pt_num * K];
	//memset(adj, 0, sizeof(int) * pt_num * K);
	int actual_pt_num = load_file(case_root + "adj.txt", adj, K);


	Mesh_Processor* mp = new Mesh_Processor(graph_path, Mesh_Processor::FEAT, false, python_path, "coarsening",
		coarsen_times, coarsen_level);
	//string label_path_str = case_root + "feature.txt";

	float output[feature_num*3];
	int exceed = 0;
	float sum = 0;
	for (int t = 0; t < 100; t++)
	{
		clock_t start = clock();

		mp->predict_feature(x, adj, actual_pt_num, 13, Mesh_Processor::RU, output);
		clock_t end = clock();
		cout <<t<<"    "<< end - start << endl;

		//float loss=mp->get_loss(label_path, feature_num, output);
		//sum += loss;
		//cout <<t <<"    "<<loss<< endl;
		//if (loss > 2) {
		//	for (int i = 0; i < feature_num * 3; i++)
		//		cout << output[i] << "  ";
		//	cout << endl;
		//	exceed++;
		//}
		//getchar();

	}

	cout << "exceed times = "<<exceed << endl;
	cout << sum / 100;
	getchar();

	return 0;
}

