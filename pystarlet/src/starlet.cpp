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

//#define test_index_period(ind ,Nind) ( ind >= Nind ? ind - Nind: ( ind < 0 ? Nind-ind : ind) )
//#define test_index_period(ind ,Nind) ( ind >= Nind ? 2*Nind - ind - 2: ( ind < 0 ? -ind : ind) )
#include <iostream>
#include "starlet.h"
//#include <gsl/gsl_rng.h>
//#include <gsl/gsl_randist.h>
#include "omp.h"


int test_index_period(int ind ,int Nind)
{
    int res = ind;
    if (ind < 0)
    {
        res = -ind;
        if(res >= Nind)
        {
            res = 2 * Nind - 2 - ind;
        }
    }
    else if (ind >= Nind)
    {
        res = 2*Nind - 2 - ind;
        if(res < 0)
        {
            res = -ind;
        }
    }
    return res;
}

Starlet1D::Starlet1D(int ny, int Nscales)
{
    Ny = ny;
    nscales = Nscales;
    
    // Allocate temporary arrays
//    tmpConvol = new double[Ny];
    
    // Set the transform coefficients
    Coeff_h0 = 3. / 8.;
    Coeff_h1 = 1. / 4.;
    Coeff_h2 = 1. / 16.;
    
    
    //Computing normalization factor for the wavelet transform
    norm1D = new double[nscales];
    norm1D_gen1 = new double[nscales];

    
    double * frame = (double *) malloc(Ny*sizeof(double));
    double * wt = (double *) malloc(Ny*nscales*sizeof(double));

    for (int j=0; j < Ny; j++)
    {
        if (j == Ny/2) {
            frame[j] = 1.0;
        }
        else{ frame[j] = 0.0;}
    }
    transform(frame,wt);
    for (int b=0; b < nscales; b++)
    {
        double temp = 0.0;
        norm1D[b] = 1.0;
        for (int j=0; j < Ny; j++) temp += wt[b*Ny + j] *wt[b*Ny + j];
        norm1D[b] = sqrt(temp);
        //std::cout << norm2D(b) << std::endl;
    }
    transform_gen1(frame,wt);
    for (int b=0; b < nscales; b++)
    {
        double temp = 0.0;
        norm1D_gen1[b] = 1.0;
        for (int j=0; j < Ny; j++) temp += wt[b*Ny + j] *wt[b*Ny + j];
        norm1D_gen1[b] = sqrt(temp);
        //std::cout << norm2D(b) << std::endl;
    }
    free(frame);
    free(wt);
}

Starlet1D::~Starlet1D()
{
    delete norm1D;
    delete norm1D_gen1;
}

/*int Starlet2D::test_index_period(int ind ,int Nind)
{
    int res = ind;
    if (ind < 0)
    {
        res = -ind;
        if(res >= Nind)
        {
            res = 2 * Nind - 2 - ind;
        }
    }
    else if (ind >= Nind)
    {
        res = 2*Nind - 2 - ind;
        if(res < 0)
        {
            res = -ind;
        }
    }
    return res;
}*/

void Starlet1D::transform(double* In, double* AlphaOut, bool normalized)
{
    double *tmpConvol = new double[Ny];
    // copy delta into first wavelet scale
    for(long ind=0; ind < Ny; ind++) AlphaOut[ind] = In[ind];
    for (int s=0; s<nscales-1;  s++)
    {
        long step =  (int)(pow((double)2., (double) s) + 0.5);
	    ///////////////////////////////////////////////////////////
	    // b3spline convolution
	    for (int j = 0; j < Ny; j ++)
        {
            double Val = Coeff_h0 * (double) AlphaOut[s*Ny +j]
            + Coeff_h1 * (  AlphaOut[s*Ny + test_index_period(j-step,Ny)]
                            + AlphaOut[s*Ny + test_index_period(j+step,Ny)])
            + Coeff_h2 * (  AlphaOut[s*Ny + test_index_period(j-2*step,Ny)]
                            + AlphaOut[s*Ny + test_index_period(j+2*step,Ny)]);
            AlphaOut[(s+1)*Ny + j] = (double) Val;
        }
	    // End of convolution
	    //////////////////////////////////////////////////////////
	    
	    ///////////////////////////////////////////////////////////
	    //// SECOND Convolution
	    for (int j = 0; j < Ny; j ++)
        {
            double Val = Coeff_h0 * (double) AlphaOut[(s+1)*Ny +j]
            + Coeff_h1 * (  AlphaOut[(s+1)*Ny + test_index_period(j-step,Ny)]
                            + AlphaOut[(s+1)*Ny + test_index_period(j+step,Ny)])
            + Coeff_h2 * (  AlphaOut[(s+1)*Ny + test_index_period(j-2*step,Ny)]
                            + AlphaOut[(s+1)*Ny + test_index_period(j+2*step,Ny)]);
            tmpConvol[j] = (double) Val;
        }
	    ///// End of Convolution
	    ////////////////////////////////////////////////////////////
        for (int j = 0; j < Ny; j ++) AlphaOut[s*Ny + j] -= (double) tmpConvol[j];
    }
    
    if(normalized){
        for (int b=0; b < nscales; b++){
            for (long ind=0; ind < Ny; ind++)
                AlphaOut[b*Ny + ind] /= norm1D[b];
        }
    }
    free(tmpConvol);
}


void Starlet1D::transform_gen1(double* In, double* AlphaOut, bool normalized)
{
    double *tmpConvol = new double[Ny];
    // copy delta into first wavelet scale
    for(long ind=0; ind < Ny; ind++) AlphaOut[ind] = In[ind];
    for (int s=0; s<nscales-1;  s++)
    {
        long step =  (int)(pow((double)2., (double) s) + 0.5);
	    ///////////////////////////////////////////////////////////
	    // b3spline convolution
	    for (int j = 0; j < Ny; j ++)
        {
            double Val = Coeff_h0 * (double) AlphaOut[s*Ny +j]
            + Coeff_h1 * (  AlphaOut[s*Ny + test_index_period(j-step,Ny)]
                          + AlphaOut[s*Ny + test_index_period(j+step,Ny)])
            + Coeff_h2 * (  AlphaOut[s*Ny + test_index_period(j-2*step,Ny)]
                          + AlphaOut[s*Ny + test_index_period(j+2*step,Ny)]);
            AlphaOut[(s+1)*Ny + j] = (double) Val;
            tmpConvol[j] = (double) Val;
        }
	    // End of convolution
	    //////////////////////////////////////////////////////////
        for (int j = 0; j < Ny; j ++) AlphaOut[s*Ny + j] -= (double) tmpConvol[j];
    }
    
    if(normalized){
        for (int b=0; b < nscales; b++){
            for (long ind=0; ind < Ny; ind++)
                AlphaOut[b*Ny + ind] /= norm1D_gen1[b];
        }
    }
    free(tmpConvol);
    
}

void Starlet1D::reconstruct(double* AlphaIn, double* Out, bool normalized)
{
    double *tmpConvol = new double[Ny];
    // Add last scale
    //memcpy(pt_delta[z*Nx*Ny],AlphaIn[z][params.nscales2d -1],Nx*Ny*sizeof(double));
    // We are removing the last wavelet scale, so set delta to zero
    if(normalized) for(long ind=0;ind<Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Ny + ind]*norm1D[nscales-1];//0.0;
    else  for(long ind=0;ind<Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Ny + ind];
    for (int s=nscales-2; s>= 0 ; s--)
    {
	    ///////////////////////////////////////////////////////////
	    // b3spline convolution
	    int step =  (int)(pow((double)2., (double) s) + 0.5);
	    for (int j = 0; j < Ny; j ++)
        {
            double Val = Coeff_h0 * (double) Out[j]
            + Coeff_h1 * (  Out[test_index_period(j-step,Ny)]
                          + Out[test_index_period(j+step,Ny)])
            + Coeff_h2 * (  Out[test_index_period(j-2*step,Ny)]
                          + Out[test_index_period(j+2*step,Ny)]);
            tmpConvol[j] = (double) Val;
        }
	    // End of convolution
	    //////////////////////////////////////////////////////////
	    
	    if(normalized){
            for (int j=0; j < Ny; j++)
                Out[j] = tmpConvol[j] +  AlphaIn[s*Ny + j]*norm1D[s];
	    }else{
            for (int j=0; j < Ny; j++)
                Out[j] = tmpConvol[j] +  AlphaIn[s*Ny + j];
	    }
    }
    free(tmpConvol);
}

void Starlet1D::reconstruct_gen1(double* AlphaIn, double* Out, bool normalized)
{
    // Add last scale
    //memcpy(pt_delta[z*Nx*Ny],AlphaIn[z][params.nscales2d -1],Nx*Ny*sizeof(double));
    // We are removing the last wavelet scale, so set delta to zero
    if(normalized) for(long ind=0;ind<Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Ny + ind]*norm1D_gen1[nscales-1];//0.0;
    else  for(long ind=0;ind<Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Ny + ind];
    for (int s=nscales-2; s>= 0 ; s--)
    {
	    if(normalized){
            for (int j=0; j < Ny; j++)
                Out[j] += AlphaIn[s*Ny + j]*norm1D_gen1[s];
	    }else{
            for (int j=0; j < Ny; j++)
                Out[j] += AlphaIn[s*Ny + j];
	    }
    }
}

void Starlet1D::trans_adjoint(double* AlphaIn, double* Out, bool normalized)
{
    double *tmpConvol = new double[Ny];
    // Add last scale
    //memcpy(pt_delta[z*Nx*Ny],AlphaIn[z][params.nscales2d -1],Nx*Ny*sizeof(double));
    // We are removing the last wavelet scale, so set delta to zero
    if(normalized) for(long ind=0;ind<Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Ny + ind]*norm1D[nscales-1];//0.0;
    else  for(long ind=0;ind<Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Ny + ind];
    for (int s=nscales-2; s>= 0 ; s--)
    {
	    ///////////////////////////////////////////////////////////
	    // b3spline convolution
	    int step =  (int)(pow((double)2., (double) s) + 0.5);
	    for (int j = 0; j < Ny; j ++)
        {
            double Val = Coeff_h0 * (double) Out[j]
            + Coeff_h1 * (  Out[test_index_period(j-step,Ny)]
                          + Out[test_index_period(j+step,Ny)])
            + Coeff_h2 * (  Out[test_index_period(j-2*step,Ny)]
                          + Out[test_index_period(j+2*step,Ny)]);
            tmpConvol[j] = (double) Val;
        }
	    // End of convolution
	    //////////////////////////////////////////////////////////
	    for (int j = 0; j < Ny; j ++) Out[j] = tmpConvol[j];
        ///////////////////////////////////////////////////////////
	    //// SECOND Convolution
	    for (int j = 0; j < Ny; j ++)
        {
            double Val = Coeff_h0 * (double) AlphaIn[(s)*Ny +j]
            + Coeff_h1 * (  AlphaIn[(s)*Ny + test_index_period(j-step,Ny)]
                          + AlphaIn[(s)*Ny + test_index_period(j+step,Ny)])
            + Coeff_h2 * (  AlphaIn[(s)*Ny + test_index_period(j-2*step,Ny)]
                          + AlphaIn[(s)*Ny + test_index_period(j+2*step,Ny)]);
            tmpConvol[j] = (double) Val;
        }
	    ///// End of Convolution
	    ////////////////////////////////////////////////////////////
	    
	    
	    ///////////////////////////////////////////////////////////
	    //// THIRD Convolution
	    for (int j = 0; j < Ny; j ++)
        {
            double Val = Coeff_h0 * (double) tmpConvol[j]
            + Coeff_h1 * (  tmpConvol[test_index_period(j-step,Ny)]
                          + tmpConvol[test_index_period(j+step,Ny)])
            + Coeff_h2 * (  tmpConvol[test_index_period(j-2*step,Ny)]
                          + tmpConvol[test_index_period(j+2*step,Ny)]);
            Out[j] +=  normalized ?  (AlphaIn[s*Ny +j] - (double) Val)*norm1D[s] : AlphaIn[s*Ny +j]- (double) Val ;
//            Out[j] +=  AlphaIn[s*Ny +j]- (double) Val ;
        }
	    ///// End of Convolution
	    ////////////////////////////////////////////////////////////
    }
    free(tmpConvol);
}

void Starlet1D::trans_adjoint_gen1(double* AlphaIn, double* Out, bool normalised)
{
    double *tmpConvol = new double[Ny];
    if(normalised) for(long ind=0;ind<Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Ny + ind]*norm1D_gen1[nscales-1];
    else for(long ind=0;ind<Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Ny + ind];
    for (int s=nscales-2; s>= 0 ; s--)
    {
	    ///////////////////////////////////////////////////////////
	    // b3spline convolution
	    int step =  (int)(pow((double)2., (double) s) + 0.5);
	    for (int j = 0; j < Ny; j ++)
        {
            double Val = Coeff_h0 * (double) Out[j]
            + Coeff_h1 * (  Out[test_index_period(j-step,Ny)]
                          + Out[test_index_period(j+step,Ny)])
            + Coeff_h2 * (  Out[test_index_period(j-2*step,Ny)]
                          + Out[test_index_period(j+2*step,Ny)]);
            tmpConvol[j] = (double) Val;
        }
	    // End of convolution
	    //////////////////////////////////////////////////////////
	    for (int j = 0; j < Ny; j ++) Out[j] = tmpConvol[j];
        ///////////////////////////////////////////////////////////
	    //// SECOND Convolution
	    for (int j = 0; j < Ny; j ++)
        {
            double Val = Coeff_h0 * (double) AlphaIn[(s)*Ny +j]
            + Coeff_h1 * (  AlphaIn[(s)*Ny + test_index_period(j-step,Ny)]
                          + AlphaIn[(s)*Ny + test_index_period(j+step,Ny)])
            + Coeff_h2 * (  AlphaIn[(s)*Ny + test_index_period(j-2*step,Ny)]
                          + AlphaIn[(s)*Ny + test_index_period(j+2*step,Ny)]);
            Out[j] += normalised ?  (AlphaIn[(s)*Ny +j] - (double) Val)*norm1D_gen1[s] : AlphaIn[(s)*Ny +j] - (double) Val ;
        }
	    ///// End of Convolution
	    ////////////////////////////////////////////////////////////
    }
    free(tmpConvol);
}

//Starlet1DStack::Starlet1DStack(int nx, int ny, int Nscales)
//{
//    Nx = nx;
//    Ny = ny;
//    nscales = Nscales;
//    objTrans = new Starlet1D(ny,Nscales);
//}
//
//Starlet1DStack::~Starlet1DStack()
//{
//    delete objTrans;
//}
//
//void Starlet1DStack::stack_transform(double* In, double* AlphaOut, bool normalized)
//{
//    int x;
//    Starlet1D *pt;
////    #pragma omp parallel for private(pt) shared(In, AlphaOut)
//    for (x=0; x<Nx; x++) {
//        pt=new Starlet1D(Ny,nscales);
//        pt->transform(&In[x*Ny],&AlphaOut[x*nscales*Ny],normalized);
//    }
//}
//
//void Starlet1DStack::stack_transform_gen1(double* In, double* AlphaOut, bool normalized)
//{
//    int x;
//    Starlet1D *pt;
//    #pragma omp parallel for private(pt) shared(In, AlphaOut)
//    for (x=0; x<Nx; x++) {
//        pt=new Starlet1D(Ny,nscales);
//        objTrans->transform_gen1(&In[x*Ny],&AlphaOut[x*nscales*Ny],normalized);
//    }
//}
//
//void Starlet1DStack::stack_reconstruct(double* AlphaIn, double* Out, bool normalized)
//{
//    int x;
//    Starlet1D *pt;
//    #pragma omp parallel for private(pt) shared(AlphaIn, Out)
//    for (x=0; x<Nx; x++) {
//        pt=new Starlet1D(Ny,nscales);
//        pt->reconstruct(&AlphaIn[x*nscales*Ny],&Out[x*Ny],normalized);
//    }
//}
//
//void Starlet1DStack::stack_reconstruct_gen1(double* AlphaIn, double* Out, bool normalized)
//{
//    int x;
//    Starlet1D *pt;
//    #pragma omp parallel for private(pt) shared(AlphaIn, Out)
//    for (x=0; x<Nx; x++) {
//        pt=new Starlet1D(Ny,nscales);
//        pt->reconstruct_gen1(&AlphaIn[x*nscales*Ny],&Out[x*Ny],normalized);
//    }
//}
//
//void Starlet1DStack::stack_trans_adjoint(double* AlphaIn, double* Out, bool normalized)
//{
//    int x;
//    Starlet1D *pt;
//    #pragma omp parallel for private(pt) shared(AlphaIn, Out)
//    for (x=0; x<Nx; x++) {
//        pt=new Starlet1D(Ny,nscales);
//        pt->trans_adjoint(&AlphaIn[x*nscales*Ny],&Out[x*Ny],normalized);
//    }
//}
//
//void Starlet1DStack::stack_trans_adjoint_gen1(double* AlphaIn, double* Out, bool normalized)
//{
//    int x;
//    Starlet1D *pt;
//    #pragma omp parallel for private(pt) shared(AlphaIn, Out)
//    for (x=0; x<Nx; x++) {
//        pt=new Starlet1D(Ny,nscales);
//        pt->trans_adjoint_gen1(&AlphaIn[x*nscales*Ny],&Out[x*Ny],normalized);
//    }
//}

Starlet2D::Starlet2D(int nx, int ny, int Nscales)
{
    Nx = nx;
    Ny = ny;
    nscales = Nscales;
    
    // Allocate temporary arrays
//    tmpWavelet = new double[Nx*Ny];
//    tmpConvol  = new double[Nx*Ny];
    
    // Set the transform coefficients
    Coeff_h0 = 3. / 8.;
    Coeff_h1 = 1. / 4.;
    Coeff_h2 = 1. / 16.;
    
    
    //Computing normalization factor for the wavelet transform
    norm2D = new double[nscales];
    norm2D_gen1 = new double[nscales];
    
    
    double * frame = (double *) malloc(Nx*Ny*sizeof(double));
    double * wt = (double *) malloc(Nx*Ny*nscales*sizeof(double));
    for (int i=0; i < Nx; i++){
        for (int j=0; j < Ny; j++)
        {
            if (i == Nx/2 && j == Ny/2) {
                frame[i*Ny + j] = 1.0;
            }else{ frame[i*Ny + j] = 0.0;}
        }
    }
    transform(frame,wt);
    for (int b=0; b < nscales; b++)
    {
        double temp = 0.0;
        norm2D[b] = 1.0;
        for (int i=0; i < Nx; i++)
            for (int j=0; j < Ny; j++) temp += wt[b*Nx*Ny +i*Ny + j] *wt[b*Nx*Ny +i*Ny + j];
        norm2D[b] = sqrt(temp);
        //std::cout << norm2D(b) << std::endl;
    }
    transform_gen1(frame,wt);
    for (int b=0; b < nscales; b++)
    {
        double temp = 0.0;
        norm2D_gen1[b] = 1.0;
        for (int i=0; i < Nx; i++)
            for (int j=0; j < Ny; j++) temp += wt[b*Nx*Ny +i*Ny + j] *wt[b*Nx*Ny +i*Ny + j];
        norm2D_gen1[b] = sqrt(temp);
        //std::cout << norm2D(b) << std::endl;
    }
    free(frame);
    free(wt);
}

Starlet2D::~Starlet2D()
{
    delete norm2D;
    delete norm2D_gen1;
}


void Starlet2D::transform(double* In, double* AlphaOut, bool normalized)
{
    double *tmpConvol  = new double[Nx*Ny];
    // copy delta into first wavelet scale
    for(long ind=0; ind < Nx*Ny; ind++) AlphaOut[ind] = In[ind];
    for (int s=0; s<nscales-1;  s++)
    {
        long step =  (int)(pow((double)2., (double) s) + 0.5);
	    ///////////////////////////////////////////////////////////
	    // b3spline convolution
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) AlphaOut[s*Nx*Ny +j*Nx+ i]
                + Coeff_h1 * (  AlphaOut[s*Nx*Ny + test_index_period(j-step,Ny)*Nx+ i]
                              + AlphaOut[s*Nx*Ny + test_index_period(j+step,Ny)*Nx+ i])
                + Coeff_h2 * (  AlphaOut[s*Nx*Ny + test_index_period(j-2*step,Ny)*Nx+ i]
                              + AlphaOut[s*Nx*Ny + test_index_period(j+2*step,Ny)*Nx+ i]);
                tmpConvol[j*Nx + i] = (double) Val;
            }
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) tmpConvol[j*Nx+ i]
                + Coeff_h1 * (  tmpConvol[j*Nx + test_index_period(i-step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+step,Nx)])
                + Coeff_h2 * (  tmpConvol[j*Nx + test_index_period(i-2*step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+2*step,Nx)]);
                AlphaOut[(s+1)*Nx*Ny + j*Nx + i] = (double) Val;
            }
	    // End of convolution
	    //////////////////////////////////////////////////////////
	    
	    ///////////////////////////////////////////////////////////
	    //// SECOND Convolution
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) AlphaOut[(s+1)*Nx*Ny +j*Nx+ i]
                + Coeff_h1 * (  AlphaOut[(s+1)*Nx*Ny + test_index_period(j-step,Ny)*Nx+ i]
                              + AlphaOut[(s+1)*Nx*Ny + test_index_period(j+step,Ny)*Nx+ i])
                + Coeff_h2 * (  AlphaOut[(s+1)*Nx*Ny + test_index_period(j-2*step,Ny)*Nx+ i]
                              + AlphaOut[(s+1)*Nx*Ny + test_index_period(j+2*step,Ny)*Nx+ i]);
                tmpConvol[j*Nx + i] = (double) Val;
            }
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) tmpConvol[j*Nx+ i]
                + Coeff_h1 * (  tmpConvol[j*Nx + test_index_period(i-step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+step,Nx)])
                + Coeff_h2 * (  tmpConvol[j*Nx + test_index_period(i-2*step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+2*step,Nx)]);
                AlphaOut[s*Nx*Ny + j*Nx + i] -= (double) Val;
            }
	    ///// End of Convolution
	    ////////////////////////////////////////////////////////////
    }
    
    if(normalized){
        for (int b=0; b < nscales; b++){
            for (long ind=0; ind < Nx*Ny; ind++)
                AlphaOut[b*Nx*Ny + ind] /= norm2D[b];
        }
    }
    free(tmpConvol);
}


void Starlet2D::transform_gen1(double* In, double* AlphaOut, bool normalized)
{
    double *tmpConvol  = new double[Nx*Ny];
    // copy delta into first wavelet scale
    for(long ind=0; ind < Nx*Ny; ind++) AlphaOut[ind] = In[ind];
    for (int s=0; s<nscales-1;  s++)
    {
        long step =  (int)(pow((double)2., (double) s) + 0.5);
	    ///////////////////////////////////////////////////////////
	    // b3spline convolution
        for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) AlphaOut[s*Nx*Ny +j*Nx+ i]
                + Coeff_h1 * (  AlphaOut[s*Nx*Ny + test_index_period(j-step,Ny)*Nx+ i]
                              + AlphaOut[s*Nx*Ny + test_index_period(j+step,Ny)*Nx+ i])
                + Coeff_h2 * (  AlphaOut[s*Nx*Ny + test_index_period(j-2*step,Ny)*Nx+ i]
                              + AlphaOut[s*Nx*Ny + test_index_period(j+2*step,Ny)*Nx+ i]);
                tmpConvol[j*Nx + i] = (double) Val;
            }
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) tmpConvol[j*Nx+ i]
                + Coeff_h1 * (  tmpConvol[j*Nx + test_index_period(i-step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+step,Nx)])
                + Coeff_h2 * (  tmpConvol[j*Nx + test_index_period(i-2*step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+2*step,Nx)]);
                AlphaOut[(s+1)*Nx*Ny + j*Nx + i] = (double) Val;
                AlphaOut[s*Nx*Ny + j*Nx + i] -= (double) Val;
            }
	    // End of convolution
	    //////////////////////////////////////////////////////////
    }
    
    if(normalized){
        for (int b=0; b < nscales; b++){
            for (long ind=0; ind < Nx*Ny; ind++)
                AlphaOut[b*Nx*Ny + ind] /= norm2D_gen1[b];
        }
    }
    free(tmpConvol);
}

void Starlet2D::reconstruct(double* AlphaIn, double* Out, bool normalized)
{
    double *tmpConvol  = new double[Nx*Ny];
    double *tmpWavelet  = new double[Nx*Ny];
    // Add last scale
    //memcpy(pt_delta[z*Nx*Ny],AlphaIn[z][params.nscales2d -1],Nx*Ny*sizeof(double));
    // We are removing the last wavelet scale, so set delta to zero
    if(normalized) for(long ind=0;ind<Nx*Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Nx*Ny + ind]*norm2D[nscales-1];//0.0;
    else  for(long ind=0;ind<Nx*Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Nx*Ny + ind];
    for (int s=nscales-2; s>= 0 ; s--)
    {
	    ///////////////////////////////////////////////////////////
	    // b3spline convolution
	    int step =  (int)(pow((double)2., (double) s) + 0.5);
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) Out[j*Nx+ i]
                + Coeff_h1 * (  Out[test_index_period(j-step,Ny)*Nx+ i]
                              + Out[test_index_period(j+step,Ny)*Nx+ i])
                + Coeff_h2 * (  Out[test_index_period(j-2*step,Ny)*Nx+ i]
                              + Out[test_index_period(j+2*step,Ny)*Nx+ i]);
                tmpConvol[j*Nx + i] = (double) Val;
            }
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) tmpConvol[j*Nx+ i]
                + Coeff_h1 * (  tmpConvol[j*Nx + test_index_period(i-step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+step,Nx)])
                + Coeff_h2 * (  tmpConvol[j*Nx + test_index_period(i-2*step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+2*step,Nx)]);
                tmpWavelet[j*Nx + i] = (double) Val;
            }
	    // End of convolution
	    //////////////////////////////////////////////////////////
	    
	    if(normalized){
            for (int j=0; j < Ny; j++)
                for (int i=0; i < Nx; i++) Out[j*Nx + i] = tmpWavelet[j*Nx + i] +  AlphaIn[s*Nx*Ny + j*Nx + i]*norm2D[s];
	    }else{
            for (int j=0; j < Ny; j++)
                for (int i=0; i < Nx; i++) Out[j*Nx + i] = tmpWavelet[j*Nx + i] +  AlphaIn[s*Nx*Ny + j*Nx + i];
	    }
    }
    free(tmpConvol);
    free(tmpWavelet);
}

void Starlet2D::reconstruct_gen1(double* AlphaIn, double* Out, bool normalized)
{
    // Add last scale
    //memcpy(pt_delta[z*Nx*Ny],AlphaIn[z][params.nscales2d -1],Nx*Ny*sizeof(double));
    // We are removing the last wavelet scale, so set delta to zero
    if(normalized) for(long ind=0;ind<Nx*Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Nx*Ny + ind]*norm2D_gen1[nscales-1];//0.0;
    else  for(long ind=0;ind<Nx*Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Nx*Ny + ind];
    for (int s=nscales-2; s>= 0 ; s--)
    {
	    if(normalized){
            for (int j=0; j < Ny; j++)
                for (int i=0; i < Nx; i++)
                    Out[j*Nx + i] += AlphaIn[s*Nx*Ny + j*Nx + i]*norm2D_gen1[s];
	    }else{
            for (int j=0; j < Ny; j++)
                for (int i=0; i < Nx; i++)
                    Out[j*Nx + i] += AlphaIn[s*Nx*Ny + j*Nx + i];
	    }
        
        
    }
}

void Starlet2D::trans_adjoint(double* AlphaIn, double* Out, bool normalized)
{
    double *tmpConvol  = new double[Nx*Ny];
    double *tmpWavelet  = new double[Nx*Ny];
    // Add last scale
    //memcpy(pt_delta[z*Nx*Ny],AlphaIn[z][params.nscales2d -1],Nx*Ny*sizeof(double));
    // We are removing the last wavelet scale, so set delta to zero
    if(normalized) for(long ind=0;ind<Nx*Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Nx*Ny + ind]*norm2D[nscales-1];//0.0;
    else  for(long ind=0;ind<Nx*Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Nx*Ny + ind];
    for (int s=nscales-2; s>= 0 ; s--)
    {
	    ///////////////////////////////////////////////////////////
	    // b3spline convolution
	    int step =  (int)(pow((double)2., (double) s) + 0.5);
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) Out[j*Nx+ i]
                + Coeff_h1 * (  Out[test_index_period(j-step,Ny)*Nx+ i]
                              + Out[test_index_period(j+step,Ny)*Nx+ i])
                + Coeff_h2 * (  Out[test_index_period(j-2*step,Ny)*Nx+ i]
                              + Out[test_index_period(j+2*step,Ny)*Nx+ i]);
                tmpConvol[j*Nx + i] = (double) Val;
            }
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) tmpConvol[j*Nx+ i]
                + Coeff_h1 * (  tmpConvol[j*Nx + test_index_period(i-step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+step,Nx)])
                + Coeff_h2 * (  tmpConvol[j*Nx + test_index_period(i-2*step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+2*step,Nx)]);
                Out[j*Nx + i] = (double) Val;
            }
	    // End of convolution
	    //////////////////////////////////////////////////////////
	    
        ///////////////////////////////////////////////////////////
	    //// SECOND Convolution
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) AlphaIn[(s)*Nx*Ny +j*Nx+ i]
                + Coeff_h1 * (  AlphaIn[(s)*Nx*Ny + test_index_period(j-step,Ny)*Nx+ i]
                              + AlphaIn[(s)*Nx*Ny + test_index_period(j+step,Ny)*Nx+ i])
                + Coeff_h2 * (  AlphaIn[(s)*Nx*Ny + test_index_period(j-2*step,Ny)*Nx+ i]
                              + AlphaIn[(s)*Nx*Ny + test_index_period(j+2*step,Ny)*Nx+ i]);
                tmpConvol[j*Nx + i] = (double) Val;
            }
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) tmpConvol[j*Nx+ i]
                + Coeff_h1 * (  tmpConvol[j*Nx + test_index_period(i-step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+step,Nx)])
                + Coeff_h2 * (  tmpConvol[j*Nx + test_index_period(i-2*step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+2*step,Nx)]);
                tmpWavelet[j*Nx + i] = (double) Val;
            }
	    ///// End of Convolution
	    ////////////////////////////////////////////////////////////
	    
	    
	    ///////////////////////////////////////////////////////////
	    //// THIRD Convolution
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) tmpWavelet[j*Nx+ i]
                + Coeff_h1 * (  tmpWavelet[test_index_period(j-step,Ny)*Nx+ i]
                              + tmpWavelet[test_index_period(j+step,Ny)*Nx+ i])
                + Coeff_h2 * (  tmpWavelet[test_index_period(j-2*step,Ny)*Nx+ i]
                              + tmpWavelet[test_index_period(j+2*step,Ny)*Nx+ i]);
                tmpConvol[j*Nx + i] = (double) Val;
            }
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) tmpConvol[j*Nx+ i]
                + Coeff_h1 * (  tmpConvol[j*Nx + test_index_period(i-step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+step,Nx)])
                + Coeff_h2 * (  tmpConvol[j*Nx + test_index_period(i-2*step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+2*step,Nx)]);
                Out[j*Nx + i] +=  normalized ? (AlphaIn[s*Nx*Ny +j*Nx+ i]- (double) Val)*norm2D[s] : AlphaIn[s*Nx*Ny +j*Nx+ i]- (double) Val ;
            }
	    ///// End of Convolution
	    ////////////////////////////////////////////////////////////
    }
    free(tmpConvol);
    free(tmpWavelet);
}

void Starlet2D::trans_adjoint_gen1(double* AlphaIn, double* Out, bool normalised)
{
    double *tmpConvol  = new double[Nx*Ny];
    if(normalised) for(long ind=0;ind<Nx*Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Nx*Ny + ind]*norm2D_gen1[nscales-1];
    else for(long ind=0;ind<Nx*Ny;ind++) Out[ind] = AlphaIn[(nscales-1)*Nx*Ny + ind];
    for (int s=nscales-2; s>= 0 ; s--)
    {
	    ///////////////////////////////////////////////////////////
	    // b3spline convolution
	    int step =  (int)(pow((double)2., (double) s) + 0.5);
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) Out[j*Nx+ i]
                + Coeff_h1 * (  Out[test_index_period(j-step,Ny)*Nx+ i]
                              + Out[test_index_period(j+step,Ny)*Nx+ i])
                + Coeff_h2 * (  Out[test_index_period(j-2*step,Ny)*Nx+ i]
                              + Out[test_index_period(j+2*step,Ny)*Nx+ i]);
                tmpConvol[j*Nx + i] = (double) Val;
            }
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) tmpConvol[j*Nx+ i]
                + Coeff_h1 * (  tmpConvol[j*Nx + test_index_period(i-step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+step,Nx)])
                + Coeff_h2 * (  tmpConvol[j*Nx + test_index_period(i-2*step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+2*step,Nx)]);
                Out[j*Nx + i] = (double) Val;
            }
	    // End of convolution
	    //////////////////////////////////////////////////////////
	    
        ///////////////////////////////////////////////////////////
	    //// SECOND Convolution
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) AlphaIn[(s)*Nx*Ny +j*Nx+ i]
                + Coeff_h1 * (  AlphaIn[(s)*Nx*Ny + test_index_period(j-step,Ny)*Nx+ i]
                              + AlphaIn[(s)*Nx*Ny + test_index_period(j+step,Ny)*Nx+ i])
                + Coeff_h2 * (  AlphaIn[(s)*Nx*Ny + test_index_period(j-2*step,Ny)*Nx+ i]
                              + AlphaIn[(s)*Nx*Ny + test_index_period(j+2*step,Ny)*Nx+ i]);
                tmpConvol[j*Nx + i] = (double) Val;
            }
	    for (int j = 0; j < Ny; j ++)
            for (int i = 0; i < Nx; i ++)
            {
                double Val = Coeff_h0 * (double) tmpConvol[j*Nx+ i]
                + Coeff_h1 * (  tmpConvol[j*Nx + test_index_period(i-step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+step,Nx)])
                + Coeff_h2 * (  tmpConvol[j*Nx + test_index_period(i-2*step,Nx)]
                              + tmpConvol[j*Nx + test_index_period(i+2*step,Nx)]);
                Out[j*Nx + i] += normalised ?  (AlphaIn[(s)*Nx*Ny +j*Nx+ i] - (double) Val)*norm2D_gen1[s] : AlphaIn[(s)*Nx*Ny +j*Nx+ i] - (double) Val ;
            }
	    ///// End of Convolution
	    ////////////////////////////////////////////////////////////
    }
    free(tmpConvol);
}

