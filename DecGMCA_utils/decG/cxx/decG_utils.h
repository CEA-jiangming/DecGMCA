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

#ifndef DECG_UTILS_H
#define DECG_UTILS_H

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include "omp.h"

#include "NumPyArrayData.h" 

#define NUM_THREADS 24

namespace bp = boost::python;
namespace np = boost::numpy;

using namespace std;

class PLK
{
    
public:
    PLK();
    ~PLK();
            
    double PowerMethod(double* AtA,double* AtAx, double* x, int nr, int nc);
    void transpose(double* A, double* tA, int nr, int nc);
    void transpose_conjugate(double* A_re, double* A_im, double* tA_re, double* tA_im, int nr, int nc);
    void MM_multiply(double* A, double* B, double *AB, int nr, int nc, int ncB);
    void complex_multiply (double A_re,double A_im,double B_re,double B_im,double* C_re,double* C_im);
    void complex_divide (double A_re,double A_im,double B_re,double B_im,double* C_re,double* C_im);
    void MM_multiply_complex(double* A_re, double* A_im, double* B_re, double* B_im, double *AB_re, double *AB_im, int nr, int nc, int nc2);
    void MV_multiply(double* A, double* b, double *Ab, int nr, int nc);
    void MV_multiply_complex(double* A_re, double* A_im, double* b_re, double* b_im, double *Ab_re, double *Ab_im, int nr, int nc);
    void VM_multiply_complex(double* a_re, double* a_im, double* B_re, double* B_im, double *aB_re, double *aB_im, int nr, int nc);
    void Basic_CG(double* Z, double* X, double* S, int nc, int niter_max);
    void Basic_CG_complex(double* Z_re, double* Z_im, double* X_re, double* X_im, double* S_re, double* S_im, int nc, int niter_max);
    void Mcopy(double* A, double* cA, int nr, int nc);
    
       //####################################################
       // APPLY (Ht H + epsilon*L*eye(3))-1 Ht
       //####################################################
       
       np::ndarray applyHt_PInv_S_numpy(np::ndarray X_In,np::ndarray M,np::ndarray mixmat,np::ndarray epsilon,np::ndarray npar,np::ndarray nfreq,np::ndarray nx){
        
        NumPyArrayData<double> X_data(X_In);
        NumPyArrayData<double> B_data(M);
        NumPyArrayData<double> Epsi(epsilon);
        NumPyArrayData<double> Npar(npar);
        NumPyArrayData<double> n_x(nx);
        NumPyArrayData<double> Nfreq(nfreq);
        NumPyArrayData<double> MixMat(mixmat);
        
        
        double eps = Epsi(0);
        long NSources = Npar(0);  // Number of sources
        long NFreq = Nfreq(0);  // Number of observations
        long Nx = n_x(0);  // Number of elements in k-space
        
        np::ndarray Out = np::zeros(bp::make_tuple(NSources,Nx), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_data(Out);
                
        //
                
        int x=0;
        
        #pragma omp parallel for shared(x, X_data, B_data, Out_data, MixMat,eps, NSources,NFreq) num_threads(NUM_THREADS)
        for (x=0; x < Nx; x++) {
            
            double *pt_Ht = (double *) malloc(sizeof(double)*NFreq*NSources);  // This is the mixing matrix
            double *pt_D = (double *) malloc(sizeof(double)*NFreq); // This is the datum
            double *pt_Pout = (double *) malloc(sizeof(double)*NSources);  // Output sources
            double *pt_P = (double *) malloc(sizeof(double)*NSources); // These are the sources
            double Lip = 0;  // Lipschitz ct
            
            // Multiplying by Ht
            
            for (int y=0; y < NFreq; y++) {
                pt_D[y] = X_data(y,x);   // Store the data
                for (int z=0;z<NSources;z++){
                    pt_Ht[z*NFreq+y] = B_data(y,x)*MixMat(y,z);  // Already the transpose of A and apply the beam
                }
            }
            
            // Apply the matrix to the data
            
            MV_multiply(pt_Ht, pt_D, pt_P, NSources, NFreq);
            
            // Computing HtH, its norm and add epsilon*L*eye
            
            double *pt_HtH = (double *) malloc(sizeof(double)*NSources*NSources);
            double *pt_H = (double *) malloc(sizeof(double)*NSources* NFreq);
            double *pt_HtHx = (double *) malloc(sizeof(double)*NSources);
            double *pt_x = (double *) malloc(sizeof(double)*NSources);
            
            transpose(pt_Ht, pt_H, NSources, NFreq);
            MM_multiply(pt_Ht, pt_H, pt_HtH, NSources, NFreq,NSources);
            
            Lip = PowerMethod(pt_HtH,pt_HtHx,pt_x,NSources,NSources);
            
            for (int z=0;z<NSources;z++){
                   pt_HtH[z*NSources+z] += eps*Lip;  // Adding epsilon*Lip
            }
            
            // Apply the inverse applied to pt_D using CG
            
            Basic_CG(pt_HtH, pt_P, pt_Pout, NSources, NSources);
            
            //
            
            for (int y=0; y < NSources; y++) Out_data(y,x) = pt_Pout[y];
            
            free(pt_D);
            free(pt_Pout);
            free(pt_H);
            free(pt_Ht);
            free(pt_HtH);
            free(pt_HtHx);
            free(pt_x);
            free(pt_P);
        }
        
        return Out;       

        }
    
    
    //####################################################
    // APPLY Ht(H Ht)-1
    //####################################################
    
    np::ndarray applyHt_PInv_A_numpy(np::ndarray X_In_re,np::ndarray X_In_im,np::ndarray M,np::ndarray S_re,np::ndarray S_im,np::ndarray npar,np::ndarray nfreq,np::ndarray nx){
        
        NumPyArrayData<double> X_re_data(X_In_re);
        NumPyArrayData<double> X_im_data(X_In_im);
        NumPyArrayData<double> B_data(M);
        NumPyArrayData<double> Npar(npar);
        NumPyArrayData<double> n_x(nx);
        NumPyArrayData<double> Nfreq(nfreq);
        NumPyArrayData<double> S_re_data(S_re);
        NumPyArrayData<double> S_im_data(S_im);
        
        
        long NSources = Npar(0);  // Number of sources
        long NFreq = Nfreq(0);  // Number of observations
        long Nx = n_x(0);  // Number of elements in k-space
        
        np::ndarray Out_re = np::zeros(bp::make_tuple(NFreq,NSources), np::dtype::get_builtin<double>());
        NumPyArrayData<double> Out_re_data(Out_re);
        
        //
        
        int x=0;
        
        #pragma omp parallel for shared(x, X_re_data, X_im_data, B_data, Out_re_data, S_re_data, S_im_data, NSources,Nx) num_threads(NUM_THREADS)
        for (x=0; x < NFreq; x++) {
            
            double *pt_Ht_re = (double *) malloc(sizeof(double)*Nx*NSources);  // This is the source matrix
            double *pt_Ht_im = (double *) malloc(sizeof(double)*Nx*NSources);  // This is the source matrix
            double *pt_D_re = (double *) malloc(sizeof(double)*Nx); // This is the datum
            double *pt_D_im = (double *) malloc(sizeof(double)*Nx); // This is the datum
            double *pt_Pout_re = (double *) malloc(sizeof(double)*NSources);  // Output mixing matrix
            double *pt_Pout_im = (double *) malloc(sizeof(double)*NSources);  // Output mixing matrix
            double *pt_P_re = (double *) malloc(sizeof(double)*NSources); // These are the mixing matrix
            double *pt_P_im = (double *) malloc(sizeof(double)*NSources); // These are the mixing matrix
            
            // Multiplying by Ht
            
            for (int y=0; y < Nx; y++) {
                pt_D_re[y] = X_re_data(x,y);   // Store the data
                pt_D_im[y] = X_im_data(x,y);   // Store the data
                for (int z=0;z<NSources;z++){
                    pt_Ht_re[y*NSources+z] = B_data(x,y)*S_re_data(z,y);  // Already the conjugate transpose of S and apply the beam
                    pt_Ht_im[y*NSources+z] = -B_data(x,y)*S_im_data(z,y);  // Already the conjugate transpose of S and apply the beam
                }
            }
            
            // Apply the matrix to the data
            
            VM_multiply_complex(pt_D_re, pt_D_im, pt_Ht_re, pt_Ht_im, pt_P_re, pt_P_im, Nx, NSources);
            
            // Computing HHt
            
            double *pt_HHt_re = (double *) malloc(sizeof(double)*NSources*NSources);
            double *pt_HHt_im = (double *) malloc(sizeof(double)*NSources*NSources);
            double *pt_HtH_re = (double *) malloc(sizeof(double)*NSources*NSources);
            double *pt_HtH_im = (double *) malloc(sizeof(double)*NSources*NSources);
            double *pt_H_re = (double *) malloc(sizeof(double)*Nx* NSources);
            double *pt_H_im = (double *) malloc(sizeof(double)*Nx* NSources);
            
            transpose_conjugate(pt_Ht_re, pt_Ht_im, pt_H_re, pt_H_im, Nx, NSources);
            
            MM_multiply_complex(pt_H_re, pt_H_im, pt_Ht_re, pt_Ht_im, pt_HHt_re, pt_HHt_im, NSources, Nx,NSources);
            
            transpose(pt_HHt_re, pt_HtH_re, NSources, NSources);
            transpose(pt_HHt_im, pt_HtH_im, NSources, NSources);
            
            Basic_CG_complex(pt_HtH_re, pt_HtH_im, pt_P_re, pt_P_im, pt_Pout_re, pt_Pout_im, NSources, NSources);
            
            //
            
            for (int y=0; y < NSources; y++)
            {
                Out_re_data(x,y) = pt_Pout_re[y];
            }
            
            free(pt_D_re);
            free(pt_D_im);
            free(pt_Pout_re);
            free(pt_Pout_im);
            free(pt_H_re);
            free(pt_H_im);
            free(pt_Ht_re);
            free(pt_Ht_im);
            free(pt_HHt_re);
            free(pt_HHt_im);
            free(pt_HtH_re);
            free(pt_HtH_im);
            free(pt_P_re);
            free(pt_P_im);
        }
        
        return Out_re;
        
    }

    
private:
    
    int Nx, NFreq, Nf;  // Not used

};

#endif // PLANCK_UTILS_H