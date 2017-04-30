#include "decG_utils.h"
// #include <gsl/gsl_rng.h>
// #include <gsl/gsl_randist.h>
#include "omp.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>   
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;

PLK::PLK()
{
	// Nothing to be done
}

PLK::~PLK()
{
    // Nothing to be done
}


/****************************************************************************/
// Compute matrix transposition with nr rows and nc columns
/****************************************************************************/

void PLK::transpose(double* A, double* tA, int nr, int nc)
{
    for (int i=0;i<nr;i++)
    {
        for (int j=0;j<nc;j++)
        {
            tA[j*nr + i] = A[i*nc + j];
        }
    }
}

/****************************************************************************/
// Compute matrix conjugate transposition with nr rows and nc columns
/****************************************************************************/

void PLK::transpose_conjugate(double* A_re, double* A_im, double* tA_re, double* tA_im, int nr, int nc)
{
    for (int i=0;i<nr;i++)
    {
        for (int j=0;j<nc;j++)
        {
            tA_re[j*nr + i] = A_re[i*nc + j];
            tA_im[j*nr + i] = -A_im[i*nc + j];
        }
    }
}


/****************************************************************************/
// Compute matrix - matrix multiplication with nr rows and nc columns nc2 columns
/****************************************************************************/

void PLK::MM_multiply(double* A, double* B, double *AB, int nr, int nc, int nc2)
{ 
    double temp = 0;
    for (int i=0;i<nr;i++)
    {
        for (int j=0;j<nc2;j++)
        {
            temp = 0;
            for (int k=0;k<nc;k++)
            {
                temp += A[i*nc + k]*B[k*nc2 + j];
            }
            AB[i*nc2 + j] = temp;
        }
    }
    
}

void PLK::Mcopy(double* A, double* cA, int nr, int nc)
{
    for (int i=0;i<nr*nc;i++)
    {
        cA[i] = A[i];
    }
}


void PLK::complex_multiply (double A_re,double A_im,double B_re,double B_im,double* C_re,double* C_im)
{
    *C_re = A_re*B_re - A_im*B_im;
    *C_im = A_re*B_im + A_im*B_re;
}

void PLK::complex_divide (double A_re,double A_im,double B_re,double B_im,double* C_re,double* C_im)
{
    double denom;
    denom = B_re*B_re + B_im*B_im;
    
    *C_re = A_re*B_re + A_im*B_im;
    *C_re /= denom;
    *C_im = -A_re*B_im + A_im*B_re;
    *C_im /= denom;
}


/****************************************************************************/
// Compute complex matrix - complex matrix multiplication with nr rows and nc columns nc2 columns
/****************************************************************************/

void PLK::MM_multiply_complex(double* A_re, double* A_im, double* B_re, double* B_im, double *AB_re, double *AB_im, int nr, int nc, int nc2)
{
    double res_re = 0, res_im = 0;
    double temp_re,temp_im;
    for (int i=0;i<nr;i++)
    {
        for (int j=0;j<nc2;j++)
        {
            res_re = 0;
            res_im = 0;
            for (int k=0;k<nc;k++)
            {
                complex_multiply (A_re[i*nc + k],A_im[i*nc + k],B_re[k*nc2 + j],B_im[k*nc2 + j],&temp_re,&temp_im);
                res_re += temp_re;
                res_im += temp_im;
            }
            AB_re[i*nc2 + j] = res_re;
            AB_im[i*nc2 + j] = res_im;
        }
    }
    
}

/****************************************************************************/
// Compute matrix - vector multiplication with nr rows and nc columns
/****************************************************************************/

void PLK::MV_multiply(double* A, double* b, double *Ab, int nr, int nc)
{  
    double temp = 0;
    for (int i=0;i<nr;i++)
    {
        temp = 0;
        for (int k=0;k<nc;k++)
        {
            temp += A[i*nc + k]*b[k];
        }
        Ab[i] = temp;
    }
}

/****************************************************************************/
// Compute complex matrix - complex vector multiplication with nr rows and nc columns
/****************************************************************************/

void PLK::MV_multiply_complex(double* A_re, double* A_im, double* b_re, double* b_im, double *Ab_re, double *Ab_im, int nr, int nc)
{
    double temp_re = 0, c_re = 0;
    double temp_im = 0, c_im = 0;
    for (int i=0;i<nr;i++)
    {
        temp_re = 0;
        temp_im = 0;
        for (int k=0;k<nc;k++)
        {
            complex_multiply (A_re[i*nc + k],A_im[i*nc+k],b_re[k],b_im[k],&c_re,&c_im);
            temp_re += c_re;
            temp_im += c_im;
        }
        Ab_re[i] = temp_re;
        Ab_im[i] = temp_im;
    }
}

/****************************************************************************/
// Compute complex vector - complex matrix multiplication with nr rows and nc columns
/****************************************************************************/

void PLK::VM_multiply_complex(double* a_re, double* a_im, double* B_re, double* B_im, double *aB_re, double *aB_im, int nr, int nc)
{
    double* tB_re = (double *)malloc(sizeof(double)*nc*nr);
    double* tB_im = (double *)malloc(sizeof(double)*nc*nr);
    
    transpose(B_re, tB_re,  nr, nc);
    transpose(B_im, tB_im,  nr, nc);
    
    MV_multiply_complex(tB_re,tB_im, a_re, a_im, aB_re, aB_im, nc, nr);
    
    free(tB_re);
    free(tB_im);
}

void PLK::Basic_CG(double* Z, double* X, double* S, int nc, int niter_max)
{ 
   	double tol=1e-6;
	int i,j;
	bool Go_on = true;
	double Acum =0;
	double alpha,beta,deltaold;
    double* R = (double *) malloc(sizeof(double)*nc);
    double* D = (double *) malloc(sizeof(double)*nc);
    double* Q = (double *) malloc(sizeof(double)*nc); 
    double* tempS = (double *) malloc(sizeof(double)*nc); 

    int numiter = 0;
    
    double delta = 0;
     
    for (i=0;i<nc;i++) 
    {
        tempS[i] = 0; // Initialization (Could be different)
     	R[i] = X[i];
     	delta += R[i]*R[i];
     	//delta0 += R[i]*R[i];
     	D[i] = X[i];
    }
    
    if (delta == 0) {
        delta = numeric_limits<double>::epsilon();
    }
    double delta0 = delta;
    double bestres = sqrt(delta/delta0);
     
    while (Go_on == true)
    {    	
        MV_multiply(Z,D,Q,nc,nc);
        
        Acum =0;
        for (i=0;i<nc;i++) Acum += D[i]*Q[i];
        if (Acum == 0) {
//            Acum = pow(10,-100);
            Acum = numeric_limits<double>::epsilon();
        }
        alpha = delta/Acum;
                
        deltaold = delta;
        Acum =0;
             
        for (i=0;i<nc;i++) 
        {
            tempS[i] += alpha*D[i];
     	    R[i] -= alpha*Q[i];
        	Acum += R[i]*R[i];
        }
        
        delta = Acum;
                   
        beta = delta/deltaold;
        
        for (i=0;i<nc;i++) D[i] = R[i] + beta*D[i];
        
        numiter++;
        
        if (sqrt(delta/delta0) < bestres)
        {
            Mcopy(tempS,S,nc,1);
            bestres = sqrt(delta/delta0);
        }
            
        if (numiter > niter_max) Go_on = false;
        if (numiter > nc) Go_on = false;
        if (delta < tol*tol*delta0) Go_on == false;
        
    }
    
    free(R);
    free(D);
    free(Q);
    free(tempS);
}

void PLK::Basic_CG_complex(double* Z_re, double* Z_im, double* X_re, double* X_im, double* S_re, double* S_im, int nc, int niter_max)
{
   	double tol=1e-6;
	int i,j;
	bool Go_on = true;
	double Acum_re = 0, Acum_im = 0;
	double alpha_re,alpha_im,beta,deltaold;
    double* R_re = (double *) malloc(sizeof(double)*nc);
    double* D_re = (double *) malloc(sizeof(double)*nc);
    double* Q_re = (double *) malloc(sizeof(double)*nc);
    double* tempS_re = (double *) malloc(sizeof(double)*nc);
    double* R_im = (double *) malloc(sizeof(double)*nc);
    double* D_im = (double *) malloc(sizeof(double)*nc);
    double* Q_im = (double *) malloc(sizeof(double)*nc);
    double* tempS_im = (double *) malloc(sizeof(double)*nc);
    
    int numiter = 0;
    
    double delta = 0;
    
    for (i=0;i<nc;i++)
    {
        tempS_re[i] = 0; // Initialization (Could be different)
        tempS_im[i] = 0; // Initialization (Could be different)
     	R_re[i] = X_re[i];
        R_im[i] = X_im[i];
     	delta += R_re[i]*R_re[i] + R_im[i]*R_im[i];
     	//delta0 += R[i]*R[i];
     	D_re[i] = X_re[i];
        D_im[i] = X_im[i];
    }
    
    if (delta == 0) {
        delta = numeric_limits<double>::epsilon();
    }
    double delta0 = delta;
    double bestres = sqrt(delta/delta0);
    
    while (Go_on == true)
    {
        MV_multiply_complex(Z_re,Z_im,D_re,D_im,Q_re,Q_im,nc,nc);
        
        Acum_re = 0;
        Acum_im = 0;
        for (i=0;i<nc;i++)
        {
            Acum_re += D_re[i]*Q_re[i] + D_im[i]*Q_im[i];
            Acum_im += D_re[i]*Q_im[i] - D_im[i]*Q_re[i];
        }
        
        if (Acum_re == 0 && Acum_im == 0) {
            Acum_re = numeric_limits<double>::epsilon();
            Acum_re = numeric_limits<double>::epsilon();
        }
        
        complex_divide (delta,0,Acum_re,Acum_im,&alpha_re,&alpha_im);
        
        deltaold = delta;
        delta =0;
        
        for (i=0;i<nc;i++)
        {
            tempS_re[i] += alpha_re*D_re[i] - alpha_im*D_im[i];
            tempS_im[i] += alpha_re*D_im[i] + alpha_im*D_re[i];
     	    R_re[i] -= alpha_re*Q_re[i] - alpha_im*Q_im[i];
     	    R_im[i] -= alpha_re*Q_im[i] + alpha_im*Q_re[i];
        	delta += R_re[i]*R_re[i] + R_im[i]*R_im[i];
        }
        
        beta = delta/deltaold;
        
        for (i=0;i<nc;i++)
        {
            D_re[i] = R_re[i] + beta*D_re[i];
            D_im[i] = R_im[i] + beta*D_im[i];
        }
        
        numiter++;
        
        if (sqrt(delta/delta0) < bestres)
        {
            Mcopy(tempS_re,S_re,nc,1);
            Mcopy(tempS_im,S_im,nc,1);
            bestres = sqrt(delta/delta0);
        }
        
        if (numiter > niter_max) Go_on = false;
        if (numiter > nc) Go_on = false;
        if (delta < tol*tol*delta0) Go_on == false;
        
    }
    
    free(R_re);
    free(R_im);
    free(D_re);
    free(D_im);
    free(Q_re);
    free(Q_im);
    free(tempS_re);
    free(tempS_im);
}


/****************************************************************************/

double PLK::PowerMethod(double* AtA,double* AtAx, double* x, int nr, int nc)
{  
	double Acum=0;
	int i,k;
	int niter = 50;
	
	// Pick x at random
	
	srand( time(0));
	
	for (i=0;i < nc;i++) x[i] = std::rand() % 10-1;  //We  might need to check if it works or not
		
	// Compute the transpose of A
	
	for (k=0; k < niter ; k++) 
    {

      MV_multiply(AtA,x,AtAx,nc,nr);  

      Acum=0;
      for (i=0;i < nc;i++) Acum += AtAx[i]*AtAx[i];
      
      if (Acum < 1e-16)  // Re-Run
      {
           srand( time(0));
	       for (i=0;i < nc;i++) x[i] = std::rand() % 10 - 1; 
      }
      
      if (Acum > 1e-16)  // PK
      {
           for (i=0;i < nc;i++) x[i] = AtAx[i]/(sqrt(Acum)+1e-12);
      }
      	
    }
    
    return sqrt(Acum);
}


