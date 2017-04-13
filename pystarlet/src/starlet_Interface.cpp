#include "starlet.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(starlet)
{
	np::initialize();
	
	class_< Starlet1D >("Starlet1D", init<int, int >())
    .def("transform", &Starlet1D::transform_numpy)
    .def("transform_gen1", &Starlet1D::transform_gen1_numpy)
    .def("reconstruct", &Starlet1D::reconstruct_numpy)
    .def("reconstruct_gen1", &Starlet1D::reconstruct_gen1_numpy)
    .def("trans_adjoint", &Starlet1D::trans_adjoint_numpy)
    .def("trans_adjoint_gen1", &Starlet1D::trans_adjoint_gen1_numpy)
    .def("stack_transform", &Starlet1D::stack_transform_numpy)
    .def("stack_transform_gen1", &Starlet1D::stack_transform_gen1_numpy)
    .def("stack_reconstruct", &Starlet1D::stack_reconstruct_numpy)
    .def("stack_reconstruct_gen1", &Starlet1D::stack_reconstruct_gen1_numpy)
    .def("stack_trans_adjoint", &Starlet1D::stack_trans_adjoint_numpy)
    .def("stack_trans_adjoint_gen1", &Starlet1D::stack_trans_adjoint_gen1_numpy);
    
//    class_< Starlet1DStack >("Starlet1DStack", init<int, int, int >())
//    .def("stack_transform", &Starlet1DStack::stack_transform_numpy)
//    .def("stack_transform_gen1", &Starlet1DStack::stack_transform_gen1_numpy)
//    .def("stack_reconstruct", &Starlet1DStack::stack_reconstruct_numpy)
//    .def("stack_reconstruct_gen1", &Starlet1DStack::stack_reconstruct_gen1_numpy)
//    .def("stack_trans_adjoint", &Starlet1DStack::stack_trans_adjoint_numpy)
//    .def("stack_trans_adjoint_gen1", &Starlet1DStack::stack_trans_adjoint_gen1_numpy);
    
	class_< Starlet2D >("Starlet2D", init<int, int, int >())
    .def("transform", &Starlet2D::transform_numpy)
    .def("transform_gen1", &Starlet2D::transform_gen1_numpy)
    .def("reconstruct", &Starlet2D::reconstruct_numpy)
    .def("reconstruct_gen1", &Starlet2D::reconstruct_gen1_numpy)
    .def("trans_adjoint", &Starlet2D::trans_adjoint_numpy)
    .def("trans_adjoint_gen1", &Starlet2D::trans_adjoint_gen1_numpy)
    .def("stack_transform", &Starlet2D::stack_transform_numpy)
    .def("stack_transform_gen1", &Starlet2D::stack_transform_gen1_numpy)
    .def("stack_reconstruct", &Starlet2D::stack_reconstruct_numpy)
    .def("stack_reconstruct_gen1", &Starlet2D::stack_reconstruct_gen1_numpy)
    .def("stack_trans_adjoint", &Starlet2D::stack_trans_adjoint_numpy)
    .def("stack_trans_adjoint_gen1", &Starlet2D::stack_trans_adjoint_gen1_numpy);
}