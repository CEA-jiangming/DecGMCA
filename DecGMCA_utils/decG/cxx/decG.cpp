#include "decG_utils.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(decG)
{
	np::initialize();
	
	class_< PLK >("PLK",init<>())
    .def("applyH_Pinv_S",&PLK::applyHt_PInv_S_numpy)
    .def("applyH_Pinv_A",&PLK::applyHt_PInv_A_numpy);
    
}