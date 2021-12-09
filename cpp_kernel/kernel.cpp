#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>

namespace py = pybind11;

py::array product(py::array_t<double> m1, py::array_t<double> m2)
{
    py::buffer_info info1 = m1.request();
    double * ptr1 = static_cast<double *>(info1.ptr);

    py::buffer_info info2 = m2.request();
    double * ptr2 = static_cast<double *>(info2.ptr);

    unsigned int nbRows1 = info1.shape[0];
    unsigned int nbCols1 = info1.shape[1];

    unsigned int nbRows2 = info2.shape[0];
    unsigned int nbCols2 = info2.shape[1];

    int resDim = nbRows1 * nbCols2;

    double * ptr = new double[resDim];

    double localSum = 0.0;
    for (int i = 0 ; i < nbRows1; ++i)
    {
        for (int j = 0 ; j < nbCols2; ++j)
        {
            for (int l = 0; l < nbCols1; ++l)
            {
                localSum += ptr1[nbCols1 * i + l] * ptr2[nbCols2 * l + j];
            }
            ptr[nbCols2 * i + j] = localSum;
            localSum = 0.0;
        }
    }
    py::array_t<double> mRes = py::array_t<double>
                                (
                                    py::buffer_info
                                    (
                                        ptr,
                                        sizeof(double), //itemsize
                                        py::format_descriptor<double>::format(),
                                        2, // ndim
                                        std::vector<size_t> { nbRows1, nbCols2 }, // shape
                                        std::vector<size_t> {nbRows1 * sizeof(double), sizeof(double)} // strides
                                    )
                                );
    delete[] ptr;
    return mRes;
}


PYBIND11_MODULE(kernel, m) {
        m.doc() = "Multiply two matrices using pybind11"; // optional module docstring

        m.def("product", &product, "Product of two Numpy arrays");
        
}