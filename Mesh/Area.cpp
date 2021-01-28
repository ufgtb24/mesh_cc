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
	//��ӣ�������������
	//�ű������ж��ٷ���ֵ��һ��Ҫ�ŵ�һ���б����ʹֻ��һ��������������
	//�༭ �ű� һ��Ҫ�� Pycharm �򿪣���Ȼ���п��������ĸ�ʽ���󣡣�������
	//Python�еķ�������һ��Ҫ��ʽָ������Ϊ32bit����ΪĬ����64bit����
	//C++ �У���ʹ��64λ���� float �� int Ҳ��32 λ�ģ�����������

#if defined(AI_DEBUG)

	pFunc_Loss = PyObject_GetAttrString(pModule, "loss_debug");//multi_coarsen
	if (pFunc_Loss == nullptr)
		cout << "no pFuncLoss is load" << endl;
#endif

	pFunc_Preprocess = PyObject_GetAttrString(pModule, "area_preprocess");//multi_coarsen
	if (pFunc_Preprocess == nullptr)
		cout << "no pFunc_Preprocess is load" << endl;

	pFunc_Postprocess = PyObject_GetAttrString(pModule, "area_postprocess");
	if (pFunc_Postprocess == nullptr)
		cout << "no pFunc_Postprocess is load" << endl;


}


PyObject* Area_Processor::preprocess(PyObject* X, PyObject* Adj, 
	PartID part_id, int target_num)
{
	PyObject* ArgArray = PyTuple_New(5);


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


PyObject* Area_Processor::postprocess(
	PyObject* Dec,
	PyObject* Adj, 
	PyObject* Blur,
	PyObject* Map
)
{
	PyObject* ArgArray = PyTuple_New(4);
	PyTuple_SetItem(ArgArray, 0, Dec);
	PyTuple_SetItem(ArgArray, 1, Adj);
	PyTuple_SetItem(ArgArray, 2, Blur);
	PyTuple_SetItem(ArgArray, 3, Map);

	PyObject* FuncOneBack = PyObject_CallObject(pFunc_Postprocess, ArgArray);
	return FuncOneBack;
}


static void DeallocateTensor(void* data, std::size_t, void*) {
	//std::free(data);
}




auto Area_Processor::run_graph(float* vertice, int pt_num, PyObject* perm_adj_map
	//int* output,
	//int* blur,
	//int& out_size

) {


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
	Status status = sess->Run(inputs, { "output_node","blur_node" }, {}, &outputs);
	if (!status.ok()) {
		cout<<"run error !!  "<<endl;
	}

	
	clock_t end0 = clock();
	cout << "sess->Run time: " << end0 - start0 << endl;

	auto output_c = outputs[0].flat<int>();
	int* output = (int*)(output_c.data());
	int out_size = output_c.size();

	cout << "out_size =" << out_size << endl;

	auto blur_c = outputs[1].flat<int>();
	int* blur = (int*)(blur_c.data());
	int blur_size = blur_c.size();

	cout << "blur_size =" << blur_size << endl;



	return make_tuple(output, out_size, blur, blur_size);
}


int* Area_Processor::predict(float* vertice_ori, int* adj, int pt_num,
	int init_K, PartID part_id, int& num)
{
	clock_t start0 = clock();


	npy_intp Dims0[2] = { pt_num, 3 };
	PyObject* X = PyArray_SimpleNewFromData(2, Dims0, NPY_FLOAT, vertice_ori);

	npy_intp Dims1[2] = { pt_num, init_K };
	PyObject* Adj = PyArray_SimpleNewFromData(2, Dims1, NPY_INT, adj);



	PyObject* perm_adj_pmap_dmap = preprocess(X, Adj, part_id, 2000);
	clock_t end0 = clock();
	cout << "preprocess time: " << end0 - start0 << endl;

	PyArrayObject* vertice_dec = (PyArrayObject*)PyList_GetItem(perm_adj_pmap_dmap, 3 * c_times + 1);
	PyArrayObject* dec_map = (PyArrayObject*)PyList_GetItem(perm_adj_pmap_dmap, 3 * c_times + 2);

	int* dec_area_id;
	int* blur_area_id;
	int dec_area_num;
	int blur_num;
	tie(dec_area_id, dec_area_num,blur_area_id,blur_num) = run_graph((float*)(vertice_dec->data), vertice_dec->dimensions[0],
		perm_adj_pmap_dmap);

	//cout << "pt_num  " << pt_num << endl;
	//cout << "dec_map size = " << dec_map->dimensions[0] << endl;
	//cout << "dec_pts_num = " << vertice_dec->dimensions[0] << endl;
	cout << "dec_area_num = " << dec_area_num << endl;

	PyArrayObject* dec_adj = (PyArrayObject*)PyList_GetItem(perm_adj_pmap_dmap, c_times);//TODO delete adj

	npy_intp Dims2[1] = { dec_area_num };
	PyObject* Dec = PyArray_SimpleNewFromData(1, Dims2, NPY_INT, dec_area_id);

	cout << "blur_num = " << blur_num << endl;
	npy_intp Dims3[1] = { blur_num };
	PyObject* Blur = PyArray_SimpleNewFromData(1, Dims3, NPY_INT, blur_area_id);



	PyObject* recover_id = postprocess(
		Dec, (PyObject*)dec_adj, Blur, (PyObject*)dec_map);

	PyArrayObject* id_np = (PyArrayObject*)PyList_GetItem(recover_id, 0);
	int* output = (int*)(id_np->data);
	num = id_np->dimensions[0];
	return output;

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