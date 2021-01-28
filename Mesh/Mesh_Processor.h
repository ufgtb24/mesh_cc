#pragma once

#ifdef MESH_EXPORTS
#define MESH_API __declspec(dllexport)
#else
#define MESH_API __declspec(dllimport)
#endif
//#define AI_DEBUG


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

enum class PartID { LD, RD, LU, RU };
class MESH_API Mesh_Processor
{
public:
	Mesh_Processor(string graph_path);
	~Mesh_Processor();
	//enum PartID { LD, RD, LU, RU };

protected:
	PyObject* pFunc_Coarsen;
	PyObject* pFunc_Normal;
	PyObject* pFunc_iNormal;
	int init_numpy();
	unique_ptr<Session> sess;
	wchar_t* GetWC(string str);
	PyObject* init_python(string python_path, string script_name);


	//virtual void init_python(string python_path, string script_name) = 0;
	//virtual float* run_graph(float* vertice, int* adj, int pt_num, int init_K, int& out_size) = 0;

	Status LoadGraph(const string& graph_file_name,
		unique_ptr<tensorflow::Session>* session);
	PyObject* normalize(float* x, int pt_num, PartID part_id);
	PyObject* ivs_normalize(float* feature, float* center, int feature_num, PartID part_id);


};


class MESH_API Feature_Processor :public Mesh_Processor
{

public:
	Feature_Processor(string graph_path,
		string python_path, string script_name, int coasen_times, int coarsen_levels[]);
	~Feature_Processor();
	void predict(float* vertice, int* adj, int pt_num, int init_K, PartID part_id, float** output);

#if defined(AI_DEBUG)
	float get_loss(const char* label_path, int f_num, float* predict);
#endif

private:
	int c_levels[10];
	int c_times;
#if defined(AI_DEBUG)
	PyObject* pFunc_Loss;
#endif

	int init_numpy();

	void init_python(string python_path, string script_name);


	PyObject* coarsen(int* adj, int pt_num, int init_K);
	float* run_graph(float* vertice, int pt_num, PyObject* perm_adj_map, int& out_size);


};


class MESH_API Area_Processor :public Mesh_Processor
{

public:
	Area_Processor(string graph_path,
		string python_path, string script_name, int coasen_times, int coarsen_levels[]);
	~Area_Processor();
	int* predict(float* vertice, int* adj, int pt_num, int init_K, PartID part_id, int& num);

#if defined(AI_DEBUG)
	float get_loss(const char* label_path, int f_num, float* predict);
#endif

private:
	PyObject* pFunc_Preprocess;
	PyObject* pFunc_Postprocess;
	int c_levels[10];
	int c_times;
#if defined(AI_DEBUG)
	PyObject* pFunc_Loss;
#endif

	int init_numpy();

	void init_python(string python_path, string script_name);


	PyObject* preprocess(PyObject* X, PyObject* Adj, 
		PartID part_id,int target_num );

	PyObject* postprocess(
		PyObject* Dec,
		PyObject* Adj,
		PyObject* Blur,
		PyObject* Map
	);

	auto run_graph(float* vertice, int pt_num, PyObject* perm_adj_map);

};


class MESH_API Orientation_Processor :public Mesh_Processor
{
public:
	Orientation_Processor(string graph_path, 
		string python_path, string script_name,
		int coarsen_times, int coarsen_level);
	~Orientation_Processor();
	void predict(float* vertice, int* adj, int pt_num, int init_K, float** output);

private:

	const int c_times;
	const int c_level;

	int init_numpy();

	void init_python(string python_path, string script_name);
	float* run_graph(float* vertice, int* adj, int pt_num, int init_K, int& out_size);


	PyObject* coarsen(int* adj, int pt_num, int init_K);


};

