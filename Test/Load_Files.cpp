#include "Header.h"

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
int load_file(string file_name, float* x, int K) {
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
			x[i * K + j] = atof(s.substr(pos1, pos2 - pos1).c_str());
			j++;
			pos1 = pos2 + c.size();
			pos2 = s.find(c, pos1);
		}
		if (pos1 != s.length())
			//v.push_back(s.substr(pos1));
			x[i * K + j] = atof(s.substr(pos1).c_str());
		i++;
	}
	return i;
}
