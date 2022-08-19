#ifndef __NUMPY_HELPER_H__
#define __NUMPY_HELPER_H__

// #define BOOSTPYTHON

#include <vector>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

#ifdef BOOSTPYTHON
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
namespace p = boost::python;
namespace np = boost::python::numpy;
np::ndarray toNumPyArray(const std::vector<float>& val);
np::ndarray toNumPyArray(const std::vector<double>& val);
np::ndarray toNumPyArray(const std::vector<Eigen::VectorXd>& val);
np::ndarray toNumPyArray(const std::vector<Eigen::MatrixXd>& val);
np::ndarray toNumPyArray(const std::vector<std::vector<float>>& val);
np::ndarray toNumPyArray(const std::vector<std::vector<double>>& val);
np::ndarray toNumPyArray(const std::vector<bool>& val);
np::ndarray toNumPyArray(const Eigen::VectorXd& vec);
np::ndarray toNumPyArray(const Eigen::MatrixXd& matrix);
np::ndarray toNumPyArray(const Eigen::Isometry3d& T);
Eigen::VectorXd toEigenVector(const np::ndarray& array);
std::vector<Eigen::VectorXd> toEigenVectorVector(const np::ndarray& array);
Eigen::MatrixXd toEigenMatrix(const np::ndarray& array);
std::vector<bool> toStdVector(const p::list& list);
#else
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
py::array_t<float> toNumPyArray(const std::vector<float>& val);
py::array_t<float> toNumPyArray(const std::vector<double>& val);
py::array_t<float> toNumPyArray(const std::vector<Eigen::VectorXd>& val);
py::array_t<float> toNumPyArray(const std::vector<Eigen::MatrixXd>& val);
py::array_t<float> toNumPyArray(const std::vector<std::vector<float>>& val);
py::array_t<float> toNumPyArray(const std::vector<std::vector<double>>& val);
py::array_t<float> toNumPyArray(const std::vector<bool>& val);
py::array_t<float> toNumPyArray(const Eigen::VectorXd& vec);
py::array_t<float> toNumPyArray(const Eigen::MatrixXd& matrix);
py::array_t<float> toNumPyArray(const Eigen::Isometry3d& T);
Eigen::VectorXd toEigenVector(const py::array_t<float>& array);
std::vector<Eigen::VectorXd> toEigenVectorVector(const py::array_t<float>& array);
Eigen::MatrixXd toEigenMatrix(const py::array_t<float>& array);
std::vector<bool> toStdVector(const py::list& list);
#endif

#endif