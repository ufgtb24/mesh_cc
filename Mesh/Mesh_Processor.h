#pragma once

#ifdef MESH_EXPORTS
#define MESH_API __declspec(dllexport)
#else
#define MESH_API __declspec(dllimport)
#endif

#include<string>
#include <memory>

namespace tensorflow {
	class Tensor;
	class Session;
	class Status;
	class Env;
}

struct _object;
typedef _object PyObject;

using namespace std;
using namespace tensorflow;

class MESH_API Mesh_Processor
{
public:
	enum Usage { FEAT, ORIEN };
	enum PartID { LD, RD,LU, RU };
	Mesh_Processor(string graph_path, Usage usage,bool use_GPU, 
		string python_path, string script_name,
		int coarsen_times, int coarsen_level);
	~Mesh_Processor();
	void predict_orientation(float* vertice, int* adj, int pt_num, int init_K, float** output);
	void predict_feature(float* vertice, int* adj, int pt_num, int init_K, PartID part_id, float** output);

private:
	PyObject* pFunc_Coarsen;
	PyObject* pFunc_Normal;
	PyObject* pFunc_iNormal;
	unique_ptr<Session> sess;

	const int c_times;
	const int c_level;

	wchar_t* GetWC(string str);
	int init_numpy();
	void init_python(string python_path, string script_name, Usage usage);
	float* predict(float* vertice, int* adj, int pt_num, int init_K, int& out_size);

	Status LoadGraph  (const string& graph_file_name,
		unique_ptr<tensorflow::Session>* session, bool use_GPU);

	PyObject* coarsen(int* adj, int pt_num, int init_K);
	PyObject* normalize(float* x, int pt_num, int part_id);
	PyObject* ivs_normalize(float* feature, float* center, int feature_num, int part_id);


};

