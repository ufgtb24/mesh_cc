// Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "Mesh_Processor.h"
#include <time.h>
using namespace std;


int load_file(string file_name, int* adj, int K) {
	ifstream Input2(file_name);
	const int LINE_LENGTH = 100;
	char str[LINE_LENGTH];
	int i = 0;
	string s;
	string c = " ";

	while (Input2.getline(str, LINE_LENGTH))
	{
		vector<string> v;
		//cout << "Read from file: " << str << endl;
		s = str;
		int j = 0;
		//按空格分割
		string::size_type pos1, pos2;
		pos2 = s.find(c);
		pos1 = 0;
		//cout << "pos2 " << pos2 << endl;
		while (string::npos != pos2)
		{
			//v.push_back(s.substr(pos1, pos2 - pos1));
			adj[i * K + j] = atoi(s.substr(pos1, pos2 - pos1).c_str());
			j++;
			pos1 = pos2 + c.size();
			pos2 = s.find(c, pos1);
		}
		if (pos1 != s.length())
			//v.push_back(s.substr(pos1));
			adj[i * K + j] = atoi(s.substr(pos1).c_str());
		i++;
	}
	return i;
}
int load_file(string file_name, float* adj, int K) {
	ifstream Input2(file_name);
	const int LINE_LENGTH = 100;
	char str[LINE_LENGTH];
	int i = 0;
	string s;
	string c = " ";

	while (Input2.getline(str, LINE_LENGTH))
	{
		vector<string> v;
		//cout << "Read from file: " << str << endl;
		s = str;
		int j = 0;
		//按空格分割
		string::size_type pos1, pos2;
		pos2 = s.find(c);
		pos1 = 0;
		//cout << "pos2 " << pos2 << endl;
		while (string::npos != pos2)
		{
			//v.push_back(s.substr(pos1, pos2 - pos1));
			adj[i * K + j] = atof(s.substr(pos1, pos2 - pos1).c_str());
			j++;
			pos1 = pos2 + c.size();
			pos2 = s.find(c, pos1);
		}
		if (pos1 != s.length())
			//v.push_back(s.substr(pos1));
			adj[i * K + j] = atof(s.substr(pos1).c_str());
		i++;
	}
	return i;
}

int main()
{
	string graph_path = "E:\\VS_Projects\\Mesh\\Test\\output_graph.pb";
	string python_path = "D:/Python";
	int coarsen_times = 2;
	int coarsen_level = 3;
	const int pt_num = 1000;
	const int K = 20;
	Mesh_Processor* mp = new Mesh_Processor(graph_path, python_path, "coarsening",
		coarsen_times, coarsen_level, K);
	int* adj = new int[pt_num * K];
	float* x = new float[pt_num * 3];
	memset(adj, 0, sizeof(int) * pt_num * K);
	int actual_pt_num = load_file("E:\\VS_Projects\\Mesh\\Test\\adj.txt", adj, K);
	load_file("E:\\VS_Projects\\Mesh\\Test\\x.txt", x, 3);
	//for (int i = 0; i < actual_pt_num; i++) {
	//	for(int j = 0; j < K; j++)
	//	  cout<< adj[i * K + j] <<" ";
	//	cout << endl;
	//}
	//getchar();
	float output[4];
	clock_t  start, stop;
	mp->predict(x, adj, actual_pt_num, output);
	for (int j = 0; j < 4; j++) {
		cout << output[j] << "  ";
	}
	cout << endl << "*********** time monitor" << endl;

	start = clock();

	for (int i = 0; i < 10; i++) {
		float outputn[4];
		mp->predict(x, adj, actual_pt_num, outputn);
		for (int j = 0; j < 4; j++) {
			cout << outputn[j] << "  ";
		}
		cout << endl;
	}
	stop = clock();

	cout<<"time cost: "<<stop - start<<" ms";
	getchar();

	//cout << output[0] << "  " << output[1] << "  "\
	//	<< output[2] << "  " << output[3] << "  ";
	return 0;
}

