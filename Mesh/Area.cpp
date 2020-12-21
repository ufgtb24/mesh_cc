#include "pch.h"
#include "Mesh_Processor.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>

Area_Processor::Area_Processor(string graph_path, string python_path,
	string script_name, int coasen_times, int coarsen_levels[]) :Mesh_Processor(graph_path)

{
	init_python(python_path, script_name);
	c_times = coasen_times;
	for (int i = 0; i < coasen_times; i++) {
		c_levels[i] = coarsen_levels[i];
	}
}
Area_Processor::~Area_Processor()
{

}
int Area_Processor::init_numpy()
{
	import_array();

}
void Area_Processor::init_python(string python_path, string script_name)
{
	PyObject* pModule = Mesh_Processor::init_python(python_path, script_name);
	init_numpy();
	//天坑！！！！！！！
	//脚本无论有多少返回值，一定要放到一个列表里，即使只有一个！！！！！！
	//编辑 脚本 一定要用 Pycharm 打开，不然会有看不出来的格式错误！！！！！
	//Python中的返回数据一定要显式指定类型为32bit，因为默认是64bit，而
	//C++ 中，即使是64位机器 float 和 int 也是32 位的！！！！！！

#if defined(AI_DEBUG)

	pFunc_Loss = PyObject_GetAttrString(pModule, "loss_debug");//multi_coarsen
	if (pFunc_Loss == nullptr)
		cout << "no pFuncLoss is load" << endl;
#endif

	pFunc_Preprocess = PyObject_GetAttrString(pModule, "area_preprocess");//multi_coarsen
	if (pFunc_Coarsen == nullptr)
		cout << "no pFunc_Preprocess is load" << endl;

	pFunc_Postprocess = PyObject_GetAttrString(pModule, "recover_area");
	if (pFunc_Normal == nullptr)
		cout << "no pFunc_Postprocess is load" << endl;


}


PyObject* Area_Processor::preprocess(float* vertice_ori, int* adj, int pt_num, 
	int init_K, PartID part_id, int target_num)
{
	PyObject* ArgArray = PyTuple_New(5);

	npy_intp Dims[2] = { pt_num, 3 };
	PyObject* X = PyArray_SimpleNewFromData(2, Dims, NPY_FLOAT, vertice_ori);

	npy_intp Dims1[2] = { pt_num, init_K };
	PyObject* Adj = PyArray_SimpleNewFromData(2, Dims1, NPY_INT, adj);

	PyObject* coarsen_levels = PyList_New(c_times);
	for (int i = 0; i < c_times; i++)
		PyList_SetItem(coarsen_levels, i, Py_BuildValue("i", c_levels[i]));

	PyTuple_SetItem(ArgArray, 0, X);
	PyTuple_SetItem(ArgArray, 1, Adj);
	PyTuple_SetItem(ArgArray, 2, coarsen_levels);
	PyTuple_SetItem(ArgArray, 3, Py_BuildValue("i", target_num));
	PyTuple_SetItem(ArgArray, 4, Py_BuildValue("i", part_id));

	PyObject* FuncOneBack = PyObject_CallObject(pFunc_Preprocess, ArgArray);
	return FuncOneBack;
}


PyObject* Area_Processor::postprocess(int* dec_id, int* dec_map,int pt_num,int dec_num)
{
	PyObject* ArgArray = PyTuple_New(3);

	npy_intp Dims[1] = { dec_num };
	PyObject* DEC = PyArray_SimpleNewFromData(1, Dims, NPY_INT, dec_id);

	npy_intp Dims1[1] = { pt_num };
	PyObject* MAP = PyArray_SimpleNewFromData(1, Dims1, NPY_INT, dec_map);


	PyTuple_SetItem(ArgArray, 0, DEC);
	PyTuple_SetItem(ArgArray, 1, MAP);
	PyTuple_SetItem(ArgArray, 2, Py_BuildValue("i", pt_num));

	PyObject* FuncOneBack = PyObject_CallObject(pFunc_Postprocess, ArgArray);
	return FuncOneBack;
}


static void DeallocateTensor(void* data, std::size_t, void*) {
	//std::free(data);
}



int* Area_Processor::predict(float* vertice_ori, int* adj, int pt_num,
	int init_K, PartID part_id)
{

	PyObject* perm_adj_pmap_dmap = preprocess(vertice_ori,adj, pt_num, init_K, part_id,2000);

	PyArrayObject* vertice_dec = (PyArrayObject*)PyList_GetItem(perm_adj_pmap_dmap, 3 * c_times + 1);
	PyArrayObject* dec_map = (PyArrayObject*)PyList_GetItem(perm_adj_pmap_dmap, 3 * c_times + 2);

	cout << "dec pt_num=" << vertice_dec->dimensions[0] << endl;
	int ouput_size;
	int* area_id = run_graph((float*)(vertice_dec->data), vertice_dec->dimensions[0], 
		perm_adj_pmap_dmap, ouput_size);

	cout << "area num dec = " << ouput_size << endl;

	PyObject* recover_id=postprocess(area_id, (int*)(dec_map->data), pt_num, vertice_dec->dimensions[0]);
	PyArrayObject* id_np = (PyArrayObject*)PyList_GetItem(recover_id, 0);
	int rec_size = id_np->dimensions[0];
	int* output = (int*)(id_np->data);

	cout << "area num = " << rec_size << endl;

	return output;

}

int* Area_Processor::run_graph(float* vertice, int pt_num, PyObject* perm_adj_map, int& out_size) {


	//int* imNumPt = new int(1);
	vector<pair<string, Tensor>> inputs;



	const int64_t tensorDims[3] = { 1,pt_num ,3 };
	TF_Tensor* v_tensor = TF_NewTensor(TF_FLOAT, tensorDims, 3,
		vertice, ((size_t)pt_num * 3) * sizeof(float),
		DeallocateTensor, NULL);


	Tensor vertice_tensor = TensorCApi::MakeTensor(v_tensor->dtype, v_tensor->shape, v_tensor->buffer);
	inputs.push_back({ "vertice",vertice_tensor });
	//////////////
	for (int i = 0; i < c_times; i++) {
		PyArrayObject* perm = (PyArrayObject*)PyList_GetItem(perm_adj_map, i);//TODO delete perm

		const int64_t perm_dims[1] = { perm->dimensions[0] };

		TF_Tensor* tftensor = TF_NewTensor(TF_INT32, perm_dims, 1,
			(int*)(perm->data), (perm->dimensions[0]) * sizeof(int),
			DeallocateTensor, NULL);

		Tensor perm_tensor = TensorCApi::MakeTensor(tftensor->dtype, tftensor->shape, tftensor->buffer);

		ostringstream ostr;
		ostr << "perm_" << i;
		string node_name = ostr.str();

		inputs.push_back({ node_name,perm_tensor });


	}

	for (int i = 0; i < c_times + 1; i++) {
		PyArrayObject* adj = (PyArrayObject*)PyList_GetItem(perm_adj_map, c_times + i);//TODO delete adj

		const int64_t adj_dims[2] = { adj->dimensions[0],adj->dimensions[1] };

		TF_Tensor* tftensor = TF_NewTensor(TF_INT32, adj_dims, 2,
			(int*)(adj->data), (adj->dimensions[0] * adj->dimensions[1]) * sizeof(int),
			DeallocateTensor, NULL);

		Tensor adj_tensor = TensorCApi::MakeTensor(tftensor->dtype, tftensor->shape, tftensor->buffer);

		ostringstream ostr;
		ostr << "adj_" << i;
		string node_name = ostr.str();

		inputs.push_back({ node_name,adj_tensor });


	}
	for (int i = 0; i < c_times; i++) {
		PyArrayObject* map = (PyArrayObject*)PyList_GetItem(perm_adj_map, 2 * c_times + 1 + i);//TODO delete perm

		const int64_t map_dims[1] = { map->dimensions[0] };

		TF_Tensor* tftensor = TF_NewTensor(TF_INT32, map_dims, 1,
			(int*)(map->data), (map->dimensions[0]) * sizeof(int),
			DeallocateTensor, NULL);

		Tensor map_tensor = TensorCApi::MakeTensor(tftensor->dtype, tftensor->shape, tftensor->buffer);

		ostringstream ostr;
		ostr << "poolmap_" << i;
		string node_name = ostr.str();

		inputs.push_back({ node_name,map_tensor });


	}


	////////////////
		//const int64_t tensorDims2[2] = { pt_num ,init_K };
		//TF_Tensor* a_tensor = TF_NewTensor(TF_INT32, tensorDims2, 2,
		//	adj, ((size_t)pt_num * init_K) * sizeof(int),
		//	DeallocateTensor, NULL);


		//Tensor adj_tensor = TensorCApi::MakeTensor(a_tensor->dtype, a_tensor->shape, a_tensor->buffer);
		//inputs.push_back({ "adj",adj_tensor });
		//input_names.push_back("adj");


	vector<Tensor> outputs;
	clock_t start0 = clock();
	Status status = sess->Run(inputs, { "output_node" }, {}, &outputs);
	if (!status.ok()) {
		cout<<"run error !!  "<<endl;
	}

	
	clock_t end0 = clock();
	cout << "sess->Run time: " << end0 - start0 << endl;

	auto output_c = outputs[0].flat<int>();
	int* output_test = (int*)(output_c.data());
	out_size = output_c.size();

	return output_test;
}


#if defined(AI_DEBUG)
float Area_Processor::get_loss(const char* label_path, int f_num, float* predict)
{
	npy_intp Dims[2] = { f_num, 3 };

	PyObject* Predict = PyArray_SimpleNewFromData(2, Dims, NPY_FLOAT, predict);
	PyObject* ArgArray = PyTuple_New(2);
	PyTuple_SetItem(ArgArray, 0, Predict);
	PyTuple_SetItem(ArgArray, 1, Py_BuildValue("s", label_path));
	PyObject* FuncOneBack = PyObject_CallObject(pFunc_Loss, ArgArray);


	PyArrayObject* loss_np = (PyArrayObject*)PyList_GetItem(FuncOneBack, 0);
	//cout<< *(float*)(loss_np->data)<<endl;
	float loss = *(float*)(loss_np->data);
	return loss;
}

#endif