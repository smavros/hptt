/*
 *   Copyright (C) 2017  Paul Springer (springer@aices.rwth-aachen.de)
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>

/**
 * \brief Computes the out-of-place tensor transposition of A into B
 *
 * A tensor transposition plan is a data structure that encodes the execution 
 *     of the tensor transposition.
 * HPTT supports tensor transpositions of the form: 
 * \f[ B_{\pi(i_0,i_1,...)} = \alpha * A_{i_0,i_1,...} + 
 *     \beta * B_{\pi(i_0,i_1,...)}. \f]
 * The plan can be reused over several transpositions.
 *
 * \param[in] perm dim-dimensional array representing the permutation 
 *     of the indices. 
 *               * For instance, perm[] = {1,0,2} denotes the following 
 *               transposition: \f$B_{i1,i0,i2} \gets A_{i0,i1,i2}\f$.
 * \param[in] dim Dimensionality of the tensors
 * \param[in] alpha scaling factor for A
 * \param[in] A Pointer to the raw-data of the input tensor A
 * \param[in] sizeA dim-dimensional array that stores the sizes of each 
 *     dimension of A 
 * \param[in] outerSizeA dim-dimensional array that stores the outer-sizes 
 *     of each dimension of A.
 *               * This parameter may be NULL, indicating that the outer-size 
 *               is equal to sizeA. 
 *               * If outerSizeA is not NULL, outerSizeA[i] >= sizeA[i] for 
 *               all 0 <= i < dim must hold.
 *               * This option enables HPTT to operate on sub-tensors.
 * \param[in] beta scaling factor for B
 * \param[inout] B Pointer to the raw-data of the output tensor B
 * \param[in] outerSizeB dim-dimensional array that stores the outer-sizes of 
 *     each dimension of B.
 *               * This parameter may be NULL, indicating that the outer-size
 *               is equal to the perm(sizeA). 
 *               * If outerSizeA is not NULL, outerSizeB[i] >= perm(sizeA)[i]
 *               for all 0 <= i < dim must hold.
 *               * This option enables HPTT to operate on sub-tensors.
 * \param[in] numThreads number of threads that participate in this tensor 
 *     transposition.
 * \param[in] useRowMajor This flag indicates whether a row-major memory 
 *     layout should be used (default: off = column-major).
 */

void sTensorTranspose( const int *perm, const int dim,
     const float alpha, 
     const float *A, const int *sizeA, const int *outerSizeA, 
     const float beta,
           float *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );

void dTensorTranspose( const int *perm, const int dim,
     const double alpha, 
     const double *A, const int *sizeA, const int *outerSizeA, 
     const double beta,
           double *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );

void cTensorTranspose( const int *perm, const int dim,
     const float _Complex alpha, 
     bool conjA, 
     const float _Complex *A, const int *sizeA, const int *outerSizeA, 
     const float _Complex beta,
           float _Complex *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );

void zTensorTranspose( const int *perm, const int dim,
     const double _Complex alpha, 
     bool conjA, 
     const double _Complex *A, const int *sizeA, const int *outerSizeA, 
     const double _Complex beta,                    
           double _Complex *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );

void sTensorTransposeAutoTuneMeasure( const int *perm, const int dim,
     const float alpha, 
     const float *A, const int *sizeA, const int *outerSizeA, 
     const float beta,
           float *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );

void dTensorTransposeAutoTuneMeasure( const int *perm, const int dim,
     const double alpha, 
     const double *A, const int *sizeA, const int *outerSizeA, 
     const double beta,
           double *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );

void cTensorTransposeAutoTuneMeasure( const int *perm, const int dim,
     const float _Complex alpha, 
     bool conjA, 
     const float _Complex *A, const int *sizeA, const int *outerSizeA, 
     const float _Complex beta,
           float _Complex *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );

void zTensorTransposeAutoTuneMeasure( const int *perm, const int dim,
     const double _Complex alpha, 
     bool conjA, 
     const double _Complex *A, const int *sizeA, const int *outerSizeA, 
     const double _Complex beta,                    
           double _Complex *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );

void sTensorTransposeAutoTunePatient( const int *perm, const int dim,
     const float alpha, 
     const float *A, const int *sizeA, const int *outerSizeA, 
     const float beta,
           float *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );

void dTensorTransposeAutoTunePatient( const int *perm, const int dim,
     const double alpha, 
     const double *A, const int *sizeA, const int *outerSizeA, 
     const double beta,
           double *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );

void cTensorTransposeAutoTunePatient( const int *perm, const int dim,
     const float _Complex alpha, 
     bool conjA, 
     const float _Complex *A, const int *sizeA, const int *outerSizeA, 
     const float _Complex beta,
           float _Complex *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );

void zTensorTransposeAutoTunePatient( const int *perm, const int dim,
     const double _Complex alpha, 
     bool conjA, 
     const double _Complex *A, const int *sizeA, const int *outerSizeA, 
     const double _Complex beta,                    
           double _Complex *B,                   const int *outerSizeB, 
     const int numThreads, const int useRowMajor );


#ifdef __cplusplus
}
#endif
