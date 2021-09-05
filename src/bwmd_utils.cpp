#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>

namespace py = pybind11;


float hamdist(std::vector<int> a, std::vector<int> b){
	float result = 0;
	for (int i = 0; i < a.size(); i++){
		result += a[i] != b[i];
	}
	return result / a.size();
}


PYBIND11_MODULE(bwmd_utils, m){
	m.def("hamdist", &hamdist, "Compute hamming distance.");
}