/*
 * cln.h - This file is part of MRS3D
 * Created on 16/05/11
 * Contributor : François Lanusse (francois.lanusse@gmail.com)
 *
 * Copyright 2012 CEA
 *
 * This software is a computer program whose purpose is to apply mutli-
 * resolution signal processing algorithms on spherical 3D data.
 *
 * This software is governed by the CeCILL  license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 */

#ifndef STARLET2D_H
#define STARLET2D_H

#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include "NumPyArrayData.h"

namespace bp = boost::python;
namespace np = boost::numpy;

class Starlet2D
{
    
public:
    Starlet2D(int Nx, int Ny, int nscales);
    ~Starlet2D();
    
    void transform(double *In, double *AlphaOut, bool normalized=false);
    np::ndarray transform_numpy(np::ndarray &In, bool normalized=false){
        
        // Objects to easily access the data in the arrays
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(nscales,Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc(sizeof(double)*Nx*Ny);
        double *pt_Out = (double *) malloc(sizeof(double)*Nx*Ny*nscales);
        for (int x=0; x < Nx; x++) {
            for (int y =0; y< Ny; y++) {
                pt_In[y*Nx +x] = In_data(x,y);
            }
        }
    
        transform(pt_In,pt_Out,normalized);
        
        for(int n = 0; n < nscales ; n++){
        for (int x=0; x < Nx; x++) {
            for (int y =0; y< Ny; y++) {
                Out_data(n,x,y) = pt_Out[n*Nx*Ny + y*Nx +x];
            }
        }
        }
        
        free(pt_In);
        free(pt_Out);
        
        return Out;
    }
    
    void transform_gen1(double* In, double* AlphaOut, bool normalized = false);
    np::ndarray transform_gen1_numpy(np::ndarray &In, bool normalized = false){
    
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(nscales,Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc(sizeof(double)*Nx*Ny);
        double *pt_Out = (double *) malloc(sizeof(double)*Nx*Ny*nscales);
        for (int x=0; x < Nx; x++) {
            for (int y =0; y< Ny; y++) {
                pt_In[y*Nx +x] = In_data(x,y);
            }
        }
        
        transform_gen1(pt_In,pt_Out,normalized);
        
        for(int n = 0; n < nscales ; n++){
            for (int x=0; x < Nx; x++) {
                for (int y =0; y< Ny; y++) {
                    Out_data(n,x,y) = pt_Out[n*Nx*Ny + y*Nx +x];
                }
            }
        }
        
        free(pt_In);
        free(pt_Out);
        
        return Out;
        
    }
//    void transform_gen1(dblarray &In, dblarray &AlphaOut, bool normalised = false){
//        transform_gen1(In.buffer(),AlphaOut.buffer(),normalised);
//    }
    
    void reconstruct(double *AlphaIn, double *Out, bool normalized = false);
    np::ndarray reconstruct_numpy(np::ndarray &AlphaIn, bool normalized = false){
        
        NumPyArrayData<double> In_data(AlphaIn);
        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc (sizeof(double)*Nx*Ny*nscales);
        double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
        
        for (int n=0; n < nscales; n++) {
            for (int x=0; x< Nx; x++){
                for (int y = 0; y < Ny; y++) {
                    pt_In[n*Nx*Ny + y*Nx + x] = In_data(n,x,y);
                }
            }
        }
        
        reconstruct(pt_In,pt_Out,normalized);
        
        for (int x = 0; x < Nx; x++) {
            for (int y = 0; y < Ny; y++) {
                Out_data(x,y) = pt_Out[y*Nx+x];
            }
        }
        free(pt_In);
        free(pt_Out);
        
        
        return Out;
    }
    
//    void reconstruct(dblarray &AlphaIn, dblarray &Out, bool normalized = false){
//        reconstruct(AlphaIn.buffer(),Out.buffer(),normalized);
//    }
    
    void rec_adjoint(double *In, double *AlphaOut, bool normalized = false);
    np::ndarray rec_adjoint_numpy(np::ndarray &In, bool normalized = false){
        
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc (sizeof(double)*Nx*Ny*nscales);
        double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
        
        for (int n=0; n < nscales; n++) {
            for (int x=0; x< Nx; x++){
                for (int y = 0; y < Ny; y++) {
                    pt_In[n*Nx*Ny + y*Nx + x] = In_data(n,x,y);
                }
            }
        }
        
        reconstruct(pt_In,pt_Out,normalized);
        
        for (int x = 0; x < Nx; x++) {
            for (int y = 0; y < Ny; y++) {
                Out_data(x,y) = pt_Out[y*Nx+x];
            }
        }
        free(pt_In);
        free(pt_Out);
        
        return Out;
    }

//    void rec_adjoint(dblarray &In, dblarray &AlphaOut, bool normalized = false){
//        rec_adjoint(In.buffer(),AlphaOut.buffer(),normalized);
//    }
    
    void trans_adjoint_gen1(double* AlphaIn, double* Out, bool normalised = false);
    np::ndarray trans_adjoint_gen1_numpy(np::ndarray &In, bool normalized = false){
        
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc (sizeof(double)*Nx*Ny*nscales);
        double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
        
        for (int n=0; n < nscales; n++) {
            for (int x=0; x< Nx; x++){
                for (int y = 0; y < Ny; y++) {
                    pt_In[n*Nx*Ny + y*Nx + x] = In_data(n,x,y);
                }
            }
        }
        
        trans_adjoint_gen1(pt_In,pt_Out,normalized);
        
        for (int x = 0; x < Nx; x++) {
            for (int y = 0; y < Ny; y++) {
                Out_data(x,y) = pt_Out[y*Nx+x];
            }
        }
        free(pt_In);
        free(pt_Out);
        
        return Out;
    }
//    void trans_adjoint_gen1(dblarray &AlphaIn, dblarray &Out, bool normalised = false){
//        trans_adjoint_gen1(AlphaIn.buffer(),Out.buffer(),normalised);
//    }
    
    void trans_adjoint(double *AlphaIn, double *Out, bool normalized = false);
    np::ndarray trans_adjoint_numpy(np::ndarray &In, bool normalized = false){
        
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc (sizeof(double)*Nx*Ny*nscales);
        double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
        
        for (int n=0; n < nscales; n++) {
            for (int x=0; x< Nx; x++){
                for (int y = 0; y < Ny; y++) {
                    pt_In[n*Nx*Ny + y*Nx + x] = In_data(n,x,y);
                }
            }
        }
        
        trans_adjoint(pt_In,pt_Out,normalized);
        
        for (int x = 0; x < Nx; x++) {
            for (int y = 0; y < Ny; y++) {
                Out_data(x,y) = pt_Out[y*Nx+x];
            }
        }
        free(pt_In);
        free(pt_Out);
        
        return Out;
    }
//    void trans_adjoint(dblarray &AlphaIn, dblarray &Out, bool normalized = false){
//        trans_adjoint(AlphaIn.buffer(),Out.buffer(),normalized);
//    }
    
    void checkAdjoint_gen1();
    void checkAdjoint();
    //double get_spectral_norm(int niter);
    
private:
    
    double * norm2D;
    double * norm2D_gen1;
    
    int Nx, Ny, nscales;
    
    
    double * tmpWavelet;
    double * tmpConvol;
    
    double Coeff_h0;
    double Coeff_h1;
    double Coeff_h2;
};

#endif // STARLET2D_H