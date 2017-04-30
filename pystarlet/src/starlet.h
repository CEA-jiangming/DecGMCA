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

#ifndef STARLET_H
#define STARLET_H

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <iostream>

#include "NumPyArrayData.h"
#include "omp.h"

#define NUM_THREADS 24

namespace bp = boost::python;
namespace np = boost::numpy;

using namespace std;

class Starlet1D
{
    
public:
    Starlet1D(int Ny, int nscales);
    ~Starlet1D();
    
    void transform(double *In, double *AlphaOut, bool normalized=false);
    np::ndarray transform_numpy(np::ndarray &In, bool normalized=false){
        
        // Objects to easily access the data in the arrays
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(nscales,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc(sizeof(double)*Ny);
        double *pt_Out = (double *) malloc(sizeof(double)*Ny*nscales);

        for (int y =0; y< Ny; y++) {
            pt_In[y] = In_data(y);
        }
    
        transform(pt_In,pt_Out,normalized);
        
        for(int n = 0; n < nscales ; n++){
            
            for (int y =0; y< Ny; y++) {
                Out_data(n,y) = pt_Out[n*Ny + y];
            }
        }
        
        free(pt_In);
        free(pt_Out);
        
        return Out;
    }
    
    void transform_gen1(double* In, double* AlphaOut, bool normalized = false);
    np::ndarray transform_gen1_numpy(np::ndarray &In, bool normalized = false){
    
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(nscales,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc(sizeof(double)*Ny);
        double *pt_Out = (double *) malloc(sizeof(double)*Ny*nscales);
        for (int y =0; y< Ny; y++) {
            pt_In[y] = In_data(y);
        }
        
        transform_gen1(pt_In,pt_Out,normalized);
        
        for(int n = 0; n < nscales ; n++){
            for (int y =0; y< Ny; y++) {
                Out_data(n,y) = pt_Out[n*Ny + y];
            }
        }
        
        free(pt_In);
        free(pt_Out);
        
        return Out;
        
    }
    
    void reconstruct(double *AlphaIn, double *Out, bool normalized = false);
    np::ndarray reconstruct_numpy(np::ndarray &AlphaIn, bool normalized = false){
        
        NumPyArrayData<double> In_data(AlphaIn);
        np::ndarray Out = np::zeros(bp::make_tuple(Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc (sizeof(double)*Ny*nscales);
        double *pt_Out = (double *) malloc (sizeof(double)*Ny);
        
        for (int n=0; n < nscales; n++) {
            for (int y = 0; y < Ny; y++) {
                pt_In[n*Ny + y] = In_data(n,y);
            }
        }
        
        reconstruct(pt_In,pt_Out,normalized);
        
        for (int y = 0; y < Ny; y++) {
            Out_data(y) = pt_Out[y];
        }
        
        free(pt_In);
        free(pt_Out);
        
        
        return Out;
    }
    
    void reconstruct_gen1(double *AlphaIn, double *Out, bool normalized = false);
    np::ndarray reconstruct_gen1_numpy(np::ndarray &AlphaIn, bool normalized = false){
        
        NumPyArrayData<double> In_data(AlphaIn);
        np::ndarray Out = np::zeros(bp::make_tuple(Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc (sizeof(double)*Ny*nscales);
        double *pt_Out = (double *) malloc (sizeof(double)*Ny);
        
        for (int n=0; n < nscales; n++) {
            for (int y = 0; y < Ny; y++) {
                pt_In[n*Ny + y] = In_data(n,y);
            }
        }
        
        reconstruct_gen1(pt_In,pt_Out,normalized);
        
        for (int y = 0; y < Ny; y++) {
            Out_data(y) = pt_Out[y];
        }
        
        free(pt_In);
        free(pt_Out);
        
        
        return Out;
    }
    
    void trans_adjoint_gen1(double* AlphaIn, double* Out, bool normalised = false);
    np::ndarray trans_adjoint_gen1_numpy(np::ndarray &In, bool normalized = false){
        
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc (sizeof(double)*Ny*nscales);
        double *pt_Out = (double *) malloc (sizeof(double)*Ny);
        
        for (int n=0; n < nscales; n++) {
            for (int y = 0; y < Ny; y++) {
                pt_In[n*Ny + y] = In_data(n,y);
            }
        }
        
        trans_adjoint_gen1(pt_In,pt_Out,normalized);
        
        for (int y = 0; y < Ny; y++) {
            Out_data(y) = pt_Out[y];
        }
        free(pt_In);
        free(pt_Out);
        
        return Out;
    }
    
    void trans_adjoint(double *AlphaIn, double *Out, bool normalized = false);
    np::ndarray trans_adjoint_numpy(np::ndarray &In, bool normalized = false){
        
        NumPyArrayData<double> In_data(In);
        np::ndarray Out = np::zeros(bp::make_tuple(Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        double *pt_In = (double *) malloc (sizeof(double)*Ny*nscales);
        double *pt_Out = (double *) malloc (sizeof(double)*Ny);
        
        for (int n=0; n < nscales; n++) {
            for (int y = 0; y < Ny; y++) {
                pt_In[n*Ny + y] = In_data(n,y);
            }
        }
        
        trans_adjoint(pt_In,pt_Out,normalized);
        
        for (int y = 0; y < Ny; y++) {
            Out_data(y) = pt_Out[y];
        }
        
        free(pt_In);
        free(pt_Out);
        
        return Out;
    }
    
    np::ndarray stack_transform_numpy(np::ndarray nSr, np::ndarray &In, bool normalized = false){
        
        NumPyArrayData<double> In_data(In);
        NumPyArrayData<double> nx(nSr);
        int Nx = nx(0);
        np::ndarray Out = np::zeros(bp::make_tuple(Nx,nscales,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        #pragma omp parallel for num_threads(NUM_THREADS)
//        #pragma omp parallel for
        for (int x =0; x< Nx; x++){
            
            double *pt_In = (double *) malloc(sizeof(double)*Ny);
            double *pt_Out = (double *) malloc(sizeof(double)*nscales*Ny);
        
            for (int y =0; y< Ny; y++) {
                pt_In[y] = In_data(x,y);
            }
            
            transform(pt_In,pt_Out,normalized);
            
            for(int n = 0; n < nscales ; n++){
                for (int y =0; y< Ny; y++) {
                    Out_data(x,n,y) = pt_Out[n*Ny + y];
                }
            }
            free(pt_In);
            free(pt_Out);
        }
        
        return Out;
        
    }
    
    np::ndarray stack_transform_gen1_numpy(np::ndarray nSr, np::ndarray &In, bool normalized = false){
        
        NumPyArrayData<double> In_data(In);
        NumPyArrayData<double> nx(nSr);
        int Nx = nx(0);
        np::ndarray Out = np::zeros(bp::make_tuple(Nx,nscales,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        #pragma omp parallel for num_threads(NUM_THREADS)
//        #pragma omp parallel for
        for (int x =0; x< Nx; x++){
            
            double *pt_In = (double *) malloc(sizeof(double)*Ny);
            double *pt_Out = (double *) malloc(sizeof(double)*nscales*Ny);
            
            for (int y =0; y< Ny; y++) {
                pt_In[y] = In_data(x,y);
            }
            
            transform_gen1(pt_In,pt_Out,normalized);
            
            for(int n = 0; n < nscales ; n++){
                for (int y =0; y< Ny; y++) {
                    Out_data(x,n,y) = pt_Out[n*Ny + y];
                }
            }
            free(pt_In);
            free(pt_Out);
        }
        
        return Out;
    }
    
    np::ndarray stack_reconstruct_numpy(np::ndarray nSr, np::ndarray &AlphaIn, bool normalized = false){
        
        NumPyArrayData<double> In_data(AlphaIn);
        NumPyArrayData<double> nx(nSr);
        int Nx = nx(0);
        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        #pragma omp parallel for num_threads(NUM_THREADS)
//        #pragma omp parallel for
        for (int x=0; x < Nx; x++) {
            
            double *pt_In = (double *) malloc (sizeof(double)*nscales*Ny);
            double *pt_Out = (double *) malloc (sizeof(double)*Ny);
            
            for (int n=0; n < nscales; n++) {
                for (int y = 0; y < Ny; y++) {
                    pt_In[n*Ny + y] = In_data(x,n,y);
                }
            }
        
            reconstruct(pt_In,pt_Out,normalized);
        
            for (int y = 0; y < Ny; y++) {
                Out_data(x,y) = pt_Out[y];
            }
            free(pt_In);
            free(pt_Out);
        }
        
        return Out;
    }
    
    np::ndarray stack_reconstruct_gen1_numpy(np::ndarray nSr, np::ndarray &AlphaIn, bool normalized = false){
        
        NumPyArrayData<double> In_data(AlphaIn);
        NumPyArrayData<double> nx(nSr);
        int Nx = nx(0);
        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        #pragma omp parallel for num_threads(NUM_THREADS)
//        #pragma omp parallel for
        for (int x=0; x < Nx; x++) {
            
            double *pt_In = (double *) malloc (sizeof(double)*nscales*Ny);
            double *pt_Out = (double *) malloc (sizeof(double)*Ny);

            for (int n=0; n < nscales; n++) {
                for (int y = 0; y < Ny; y++) {
                    pt_In[n*Ny + y] = In_data(x,n,y);
                }
            }
            
            reconstruct_gen1(pt_In,pt_Out,normalized);
            
            for (int y = 0; y < Ny; y++) {
                Out_data(x,y) = pt_Out[y];
            }
            
            free(pt_In);
            free(pt_Out);
        }
        
        return Out;
    }
    
    np::ndarray stack_trans_adjoint_numpy(np::ndarray nSr, np::ndarray &AlphaIn, bool normalized = false){
        
        NumPyArrayData<double> In_data(AlphaIn);
        NumPyArrayData<double> nx(nSr);
        int Nx = nx(0);
        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        #pragma omp parallel for num_threads(NUM_THREADS)
//        #pragma omp parallel for
        for (int x=0; x < Nx; x++) {
            
            double *pt_In = (double *) malloc (sizeof(double)*nscales*Ny);
            double *pt_Out = (double *) malloc (sizeof(double)*Ny);
            
            for (int n=0; n < nscales; n++) {
                for (int y = 0; y < Ny; y++) {
                    pt_In[n*Ny + y] = In_data(x,n,y);
                }
            }
            
            trans_adjoint(pt_In,pt_Out,normalized);
            
            for (int y = 0; y < Ny; y++) {
                Out_data(x,y) = pt_Out[y];
            }
            free(pt_In);
            free(pt_Out);
        }
        
        return Out;
    }
    
    np::ndarray stack_trans_adjoint_gen1_numpy(np::ndarray nSr, np::ndarray &AlphaIn, bool normalized = false){
        
        NumPyArrayData<double> In_data(AlphaIn);
        NumPyArrayData<double> nx(nSr);
        int Nx = nx(0);
        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        #pragma omp parallel for num_threads(NUM_THREADS)
//        #pragma omp parallel for
        for (int x=0; x < Nx; x++) {
            
            double *pt_In = (double *) malloc (sizeof(double)*nscales*Ny);
            double *pt_Out = (double *) malloc (sizeof(double)*Ny);

            for (int n=0; n < nscales; n++) {
                for (int y = 0; y < Ny; y++) {
                    pt_In[n*Ny + y] = In_data(x,n,y);
                }
            }
            
            trans_adjoint_gen1(pt_In,pt_Out,normalized);
            
            for (int y = 0; y < Ny; y++) {
                Out_data(x,y) = pt_Out[y];
            }
            free(pt_In);
            free(pt_Out);
        }
        
        return Out;
    }
    
private:
    
    double * norm1D;
    double * norm1D_gen1;
    
    int Ny, nscales;
    
//    double * tmpConvol;
    
    double Coeff_h0;
    double Coeff_h1;
    double Coeff_h2;
};


//class Starlet1DStack
//{
//public:
//    
//    Starlet1DStack(int Nx, int Ny, int nscales);
//    ~Starlet1DStack();
//    
//    void stack_transform(double* In, double* AlphaOut, bool normalized = false);
//    np::ndarray stack_transform_numpy(np::ndarray &In, bool normalized = false){
//        
//        NumPyArrayData<double> In_data(In);
//        np::ndarray Out = np::zeros(bp::make_tuple(Nx,nscales,Ny), np::dtype::get_builtin<double>());
//        NumPyArrayData<double> Out_data(Out);
//        
//        double *pt_In = (double *) malloc(sizeof(double)*Nx*Ny);
//        double *pt_Out = (double *) malloc(sizeof(double)*Nx*nscales*Ny);
//        for (int x =0; x< Nx; x++){
//            for (int y =0; y< Ny; y++) {
//                pt_In[x*Ny+y] = In_data(x,y);
//            }
//        }
//        
//        stack_transform(pt_In,pt_Out,normalized);
//        
//        for (int x =0; x< Nx; x++){
//            for(int n = 0; n < nscales ; n++){
//                for (int y =0; y< Ny; y++) {
//                    Out_data(x,n,y) = pt_Out[x*nscales*Ny + n*Ny + y];
//                }
//            }
//        }
//        
//        free(pt_In);
//        free(pt_Out);
//        
//        return Out;
//        
//    }
//
//    
//    void stack_transform_gen1(double* In, double* AlphaOut, bool normalized = false);
//    np::ndarray stack_transform_gen1_numpy(np::ndarray &In, bool normalized = false){
//        
//        NumPyArrayData<double> In_data(In);
//        np::ndarray Out = np::zeros(bp::make_tuple(Nx,nscales,Ny), np::dtype::get_builtin<double>());
//        NumPyArrayData<double> Out_data(Out);
//        
//        double *pt_In = (double *) malloc(sizeof(double)*Nx*Ny);
//        double *pt_Out = (double *) malloc(sizeof(double)*Nx*nscales*Ny);
//        for (int x =0; x< Nx; x++){
//            for (int y =0; y< Ny; y++) {
//                pt_In[x*Ny+y] = In_data(x,y);
//            }
//        }
//        
//        stack_transform_gen1(pt_In,pt_Out,normalized);
//        
//        for (int x =0; x< Nx; x++){
//            for(int n = 0; n < nscales ; n++){
//                for (int y =0; y< Ny; y++) {
//                    Out_data(x,n,y) = pt_Out[x*nscales*Ny + n*Ny + y];
//                }
//            }
//        }
//        
//        free(pt_In);
//        free(pt_Out);
//        
//        return Out;
//        
//    }
//    
//    void stack_reconstruct(double *AlphaIn, double *Out, bool normalized = false);
//    np::ndarray stack_reconstruct_numpy(np::ndarray &AlphaIn, bool normalized = false){
//        
//        NumPyArrayData<double> In_data(AlphaIn);
//        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
//        NumPyArrayData<double> Out_data(Out);
//        
//        double *pt_In = (double *) malloc (sizeof(double)*Nx*nscales*Ny);
//        double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
//        
//        for (int x=0; x < Nx; x++) {
//            for (int n=0; n < nscales; n++) {
//                for (int y = 0; y < Ny; y++) {
//                    pt_In[x*nscales*Ny + n*Ny + y] = In_data(x,n,y);
//                }
//            }
//        }
//        
//        stack_reconstruct(pt_In,pt_Out,normalized);
//        
//        for (int x=0; x < Nx; x++) {
//            for (int y = 0; y < Ny; y++) {
//                Out_data(x,y) = pt_Out[x*Ny + y];
//            }
//        }
//        
//        free(pt_In);
//        free(pt_Out);
//        
//        
//        return Out;
//    }
//    
//    void stack_reconstruct_gen1(double *AlphaIn, double *Out, bool normalized = false);
//    np::ndarray stack_reconstruct_gen1_numpy(np::ndarray &AlphaIn, bool normalized = false){
//        
//        NumPyArrayData<double> In_data(AlphaIn);
//        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
//        NumPyArrayData<double> Out_data(Out);
//        
//        double *pt_In = (double *) malloc (sizeof(double)*Nx*nscales*Ny);
//        double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
//        
//        for (int x=0; x < Nx; x++) {
//            for (int n=0; n < nscales; n++) {
//                for (int y = 0; y < Ny; y++) {
//                    pt_In[x*nscales*Ny + n*Ny + y] = In_data(x,n,y);
//                }
//            }
//        }
//        
//        stack_reconstruct_gen1(pt_In,pt_Out,normalized);
//        
//        for (int x=0; x < Nx; x++) {
//            for (int y = 0; y < Ny; y++) {
//                Out_data(x,y) = pt_Out[x*Ny + y];
//            }
//        }
//        
//        free(pt_In);
//        free(pt_Out);
//        
//        
//        return Out;
//    }
//    
//    void stack_trans_adjoint(double *AlphaIn, double *Out, bool normalized = false);
//    np::ndarray stack_trans_adjoint_numpy(np::ndarray &In, bool normalized = false){
//        
//        NumPyArrayData<double> In_data(In);
//        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
//        NumPyArrayData<double> Out_data(Out);
//        
//        double *pt_In = (double *) malloc (sizeof(double)*Nx*nscales*Ny);
//        double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
//        
//        for (int x=0; x < Nx; x++) {
//            for (int n=0; n < nscales; n++) {
//                for (int y = 0; y < Ny; y++) {
//                    pt_In[x*nscales*Ny + n*Ny + y] = In_data(x,n,y);
//                }
//            }
//        }
//        
//        stack_trans_adjoint(pt_In,pt_Out,normalized);
//        
//        for (int x=0; x < Nx; x++) {
//            for (int y = 0; y < Ny; y++) {
//                Out_data(x,y) = pt_Out[x*Ny + y];
//            }
//        }
//        
//        free(pt_In);
//        free(pt_Out);
//        
//        return Out;
//    }
//    
//    void stack_trans_adjoint_gen1(double *AlphaIn, double *Out, bool normalized = false);
//    np::ndarray stack_trans_adjoint_gen1_numpy(np::ndarray &In, bool normalized = false){
//        
//        NumPyArrayData<double> In_data(In);
//        np::ndarray Out = np::zeros(bp::make_tuple(Nx,Ny), np::dtype::get_builtin<double>());
//        NumPyArrayData<double> Out_data(Out);
//        
//        double *pt_In = (double *) malloc (sizeof(double)*Nx*nscales*Ny);
//        double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
//        
//        for (int x=0; x < Nx; x++) {
//            for (int n=0; n < nscales; n++) {
//                for (int y = 0; y < Ny; y++) {
//                    pt_In[x*nscales*Ny + n*Ny + y] = In_data(x,n,y);
//                }
//            }
//        }
//        
//        stack_trans_adjoint_gen1(pt_In,pt_Out,normalized);
//        
//        for (int x=0; x < Nx; x++) {
//            for (int y = 0; y < Ny; y++) {
//                Out_data(x,y) = pt_Out[x*Ny + y];
//            }
//        }
//        
//        free(pt_In);
//        free(pt_Out);
//        
//        return Out;
//    }
//    
//private:
//    
//    double * norm1D;
//    double * norm1D_gen1;
//    int Nx, Ny, nscales;
//    Starlet1D *objTrans;
//    
//};

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
    
    void reconstruct_gen1(double *AlphaIn, double *Out, bool normalized = false);
    np::ndarray reconstruct_gen1_numpy(np::ndarray &AlphaIn, bool normalized = false){
        
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
        
        reconstruct_gen1(pt_In,pt_Out,normalized);
        
        for (int x = 0; x < Nx; x++) {
            for (int y = 0; y < Ny; y++) {
                Out_data(x,y) = pt_Out[y*Nx+x];
            }
        }
        free(pt_In);
        free(pt_Out);
        
        
        return Out;
    }
    
    
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
    
    np::ndarray stack_transform_numpy(np::ndarray nSr, np::ndarray &In, bool normalized=false){
        
        // Objects to easily access the data in the arrays
        NumPyArrayData<double> In_data(In);
        NumPyArrayData<double> n_Sr(nSr);
        int NSr = n_Sr(0);
        np::ndarray Out = np::zeros(bp::make_tuple(NSr,nscales,Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);

        #pragma omp parallel for num_threads(NUM_THREADS)
//        # pragma omp parallel for
        for (int s=0; s<NSr; s++) {
            
            double *pt_In = (double *) malloc(sizeof(double)*Nx*Ny);
            double *pt_Out = (double *) malloc(sizeof(double)*Nx*Ny*nscales);
            
            for (int x=0; x < Nx; x++) {
                for (int y =0; y< Ny; y++) {
                    pt_In[y*Nx +x] = In_data(s,x,y);
                }
            }
        
        
            transform(pt_In,pt_Out,normalized);
        
            for(int n = 0; n < nscales ; n++){
                for (int x=0; x < Nx; x++) {
                    for (int y =0; y< Ny; y++) {
                        Out_data(s,n,x,y) = pt_Out[n*Nx*Ny + y*Nx +x];
                    }
                }
            }
            free(pt_In);
            free(pt_Out);
        }

        return Out;
    }
    
    np::ndarray stack_transform_gen1_numpy(np::ndarray nSr, np::ndarray &In, bool normalized=false){
        
        // Objects to easily access the data in the arrays
        NumPyArrayData<double> In_data(In);
        NumPyArrayData<double> n_Sr(nSr);
        int NSr = n_Sr(0);
        np::ndarray Out = np::zeros(bp::make_tuple(NSr,nscales,Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        #pragma omp parallel for num_threads(NUM_THREADS)
//        # pragma omp parallel for
        for (int s=0; s<NSr; s++) {
            
            double *pt_In = (double *) malloc(sizeof(double)*Nx*Ny);
            double *pt_Out = (double *) malloc(sizeof(double)*Nx*Ny*nscales);
            
            for (int x=0; x < Nx; x++) {
                for (int y =0; y< Ny; y++) {
                    pt_In[y*Nx +x] = In_data(s,x,y);
                }
            }
            
            transform_gen1(pt_In,pt_Out,normalized);
            
            for(int n = 0; n < nscales ; n++){
                for (int x=0; x < Nx; x++) {
                    for (int y =0; y< Ny; y++) {
                        Out_data(s,n,x,y) = pt_Out[n*Nx*Ny + y*Nx +x];
                    }
                }
            }
            free(pt_In);
            free(pt_Out);
        }
        
        return Out;
    }
    
    np::ndarray stack_reconstruct_numpy(np::ndarray nSr, np::ndarray &AlphaIn, bool normalized = false){
        
        NumPyArrayData<double> In_data(AlphaIn);
        NumPyArrayData<double> n_Sr(nSr);
        int NSr = n_Sr(0);
        np::ndarray Out = np::zeros(bp::make_tuple(NSr,Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        #pragma omp parallel for num_threads(NUM_THREADS)
//        # pragma omp parallel for
        for (int s=0; s<NSr; s++) {
            
            double *pt_In = (double *) malloc (sizeof(double)*Nx*Ny*nscales);
            double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);

            for (int n=0; n < nscales; n++) {
                for (int x=0; x< Nx; x++){
                    for (int y = 0; y < Ny; y++) {
                        pt_In[n*Nx*Ny + y*Nx + x] = In_data(s,n,x,y);
                    }
                }
            }
        
            reconstruct(pt_In,pt_Out,normalized);
        
            for (int x = 0; x < Nx; x++) {
                for (int y = 0; y < Ny; y++) {
                    Out_data(s,x,y) = pt_Out[y*Nx+x];
                }
            }
            free(pt_In);
            free(pt_Out);
        }
        
        return Out;
    }
    
    np::ndarray stack_reconstruct_gen1_numpy(np::ndarray nSr, np::ndarray &AlphaIn, bool normalized = false){
        
        NumPyArrayData<double> In_data(AlphaIn);
        NumPyArrayData<double> n_Sr(nSr);
        int NSr = n_Sr(0);
        np::ndarray Out = np::zeros(bp::make_tuple(NSr,Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        #pragma omp parallel for num_threads(NUM_THREADS)
//        # pragma omp parallel for
        for (int s=0; s<NSr; s++) {
            
            double *pt_In = (double *) malloc (sizeof(double)*Nx*Ny*nscales);
            double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
            
            for (int n=0; n < nscales; n++) {
                for (int x=0; x< Nx; x++){
                    for (int y = 0; y < Ny; y++) {
                        pt_In[n*Nx*Ny + y*Nx + x] = In_data(s,n,x,y);
                    }
                }
            }
            
            reconstruct_gen1(pt_In,pt_Out,normalized);
            
            for (int x = 0; x < Nx; x++) {
                for (int y = 0; y < Ny; y++) {
                    Out_data(s,x,y) = pt_Out[y*Nx+x];
                }
            }
            free(pt_In);
            free(pt_Out);
        }
        
        return Out;
    }
    
    np::ndarray stack_trans_adjoint_numpy(np::ndarray nSr, np::ndarray &AlphaIn, bool normalized = false){
        
        NumPyArrayData<double> In_data(AlphaIn);
        NumPyArrayData<double> n_Sr(nSr);
        int NSr = n_Sr(0);
        np::ndarray Out = np::zeros(bp::make_tuple(NSr,Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
        
        #pragma omp parallel for num_threads(NUM_THREADS)
//        # pragma omp parallel for
        for (int s=0; s<NSr; s++) {
            
            double *pt_In = (double *) malloc (sizeof(double)*Nx*Ny*nscales);
            double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
            
            for (int n=0; n < nscales; n++) {
                for (int x=0; x< Nx; x++){
                    for (int y = 0; y < Ny; y++) {
                        pt_In[n*Nx*Ny + y*Nx + x] = In_data(s,n,x,y);
                    }
                }
            }
            
            trans_adjoint(pt_In,pt_Out,normalized);
            
            for (int x = 0; x < Nx; x++) {
                for (int y = 0; y < Ny; y++) {
                    Out_data(s,x,y) = pt_Out[y*Nx+x];
                }
            }
            free(pt_In);
            free(pt_Out);
        }
        return Out;
    }
    
    np::ndarray stack_trans_adjoint_gen1_numpy(np::ndarray nSr, np::ndarray &AlphaIn, bool normalized = false){
        
        NumPyArrayData<double> In_data(AlphaIn);
        NumPyArrayData<double> n_Sr(nSr);
        int NSr = n_Sr(0);
        np::ndarray Out = np::zeros(bp::make_tuple(NSr,Nx,Ny), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);

        #pragma omp parallel for num_threads(NUM_THREADS)
//        # pragma omp parallel for
        for (int s=0; s<NSr; s++) {
            
            double *pt_In = (double *) malloc (sizeof(double)*Nx*Ny*nscales);
            double *pt_Out = (double *) malloc (sizeof(double)*Nx*Ny);
            
            for (int n=0; n < nscales; n++) {
                for (int x=0; x< Nx; x++){
                    for (int y = 0; y < Ny; y++) {
                        pt_In[n*Nx*Ny + y*Nx + x] = In_data(s,n,x,y);
                    }
                }
            }
            
            trans_adjoint_gen1(pt_In,pt_Out,normalized);
            
            for (int x = 0; x < Nx; x++) {
                for (int y = 0; y < Ny; y++) {
                    Out_data(s,x,y) = pt_Out[y*Nx+x];
                }
            }
            free(pt_In);
            free(pt_Out);
        }
        
        return Out;
    }
    
private:
    
    double * norm2D;
    double * norm2D_gen1;
    
    int Nx, Ny, nscales;
    
    double Coeff_h0;
    double Coeff_h1;
    double Coeff_h2;
};


#endif // STARLET_H