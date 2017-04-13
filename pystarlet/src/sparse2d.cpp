#include "starlet2D.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(sparse2d)
{
	np::initialize();
	
	class_< Starlet2D >("Starlet2D", init<int, int, int >())
    .def("transform", &Starlet2D::transform_numpy)
    .def("transform_gen1", &Starlet2D::transform_gen1_numpy)
    .def("reconstruct", &Starlet2D::reconstruct_numpy)
    .def("rec_adjoint", &Starlet2D::rec_adjoint_numpy)
    .def("trans_adjoint", &Starlet2D::trans_adjoint_numpy)
    .def("trans_adjoint_gen1", &Starlet2D::trans_adjoint_gen1_numpy);
}