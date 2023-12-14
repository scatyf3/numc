#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    //edge case
    if (rows <=0 ){
        return -1;
    }
    if (cols <=0 ){
        return -1;
    }
    //给最初的mat分配空间
    (*mat) = (matrix*)malloc(sizeof(struct matrix));
    if ((*mat)==NULL){
        return -1;
    }
    //赋值
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL;
    (*mat)->data = malloc(rows*cols*sizeof(double));
    if (((*mat)->data)==NULL){
        return -1;
    }
    for(int i = 0;i< cols*rows ;i++){
        (*mat)->data[i] = 0;
    }
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
/*
Usage:
  CU_ASSERT_EQUAL(allocate_matrix_ref(&mat, from, 2, 2, 2), 0);
  CU_ASSERT_PTR_EQUAL(mat->data, from->data + 2);
  CU_ASSERT_PTR_EQUAL(mat->parent, from);
  CU_ASSERT_EQUAL(mat->parent->ref_cnt, 2);
  CU_ASSERT_EQUAL(mat->rows, 2);
  CU_ASSERT_EQUAL(mat->cols, 2);
*/
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    if (rows <=0 ){
        return -1;
    }
    if (cols <=0 ){
        return -1;
    }
    (*mat) = (matrix*)malloc(sizeof(struct matrix));
    if ((*mat)==NULL){
        return -1;
    }
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->ref_cnt = 1;
    (*mat)->parent = from;
    from->ref_cnt +=1;
    (*mat)->data = from->data + offset;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    if(mat==NULL){
        return;
    }
    if(mat->ref_cnt==1 && mat->parent==NULL){
        //`mat` is not a slice and has no existing slices
        free(mat->data);
    }
    else if(mat->ref_cnt==1 && mat->parent->ref_cnt==0){
        //`mat` is the last existing slice of its parent matrix and its parent matrix has no other references
        free(mat->data);
    }
    else if(mat->parent!=NULL) {
        mat->parent->ref_cnt--;
    }
    //free struct
    free(mat);
}

/*
@suggestion from CHATGPT
void deallocate_matrix(matrix *mat) {
    if (mat->parent != NULL) {
        // 若该矩阵不是根矩阵
        mat->parent->ref_cnt -= 1;
        if (mat->parent->ref_cnt == 0) {
            // 若父矩阵没有其他引用
            deallocate_matrix(mat->parent); // 递归释放父矩阵
        }
    }

    if (mat->ref_cnt == 1 && mat->parent == NULL) {
        // 若该矩阵没有子矩阵且没有其他引用
        free(mat->data);
    }

    free(mat);
}
*/

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return mat->data[row*mat->cols+col];

}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    mat->data[row*mat->cols+col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    if(mat == NULL){
        return ;
    }
    for(int i = 0;i<mat->rows*mat->cols;i++){
        mat->data[i]=val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if(mat1->rows!=mat2->rows){
        return -1;
    }
    if(mat1->cols!=mat2->cols){
        return -1;
    }
    if(mat1->rows!=result->rows){
        return -1;
    }
    if(mat1->cols!=result->cols){
        return -1;
    }
    for(int idx = 0;idx<(mat1->cols*mat1->rows);idx++){
        result->data[idx] = mat1->data[idx] + mat2->data[idx];
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if(mat1->rows!=mat2->rows){
        return -1;
    }
    if(mat1->cols!=mat2->cols){
        return -1;
    }
    if(mat1->rows!=result->rows){
        return -1;
    }
    if(mat1->cols!=result->cols){
        return -1;
    }
    for(int idx = 0;idx<(mat1->cols*mat1->rows);idx++){
        result->data[idx] = mat1->data[idx] - mat2->data[idx];
    }
    return 0;
}


/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if(mat1->rows!=mat2->rows){
        return -1;
    }
    if(mat1->cols!=mat2->cols){
        return -1;
    }
    if(mat1->rows!=result->rows){
        return -1;
    }
    if(mat1->cols!=result->cols){
        return -1;
    }
    //mat->data[row*mat->cols+col]，这个写法默认rows临近而cols不临近吧
    for(int i = 0;i<mat1->rows;i++){
        for(int j = 0;j<mat1->cols;j++){
            for(int k = 0;k<mat1->rows;k++){
                result->data[i*mat1->cols+j] += mat1->data[i*mat1->cols+k] * mat2->data[k*mat1->cols+j];
            }
        }
    }
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    int n = mat->rows;
    
    if (mat->rows != result->rows || mat->cols != result->cols || mat->rows != mat->cols) {
        return -1;
    }
    
    // 初始化 result 矩阵为对角线上元素为 1 的矩阵 
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result->data[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    if (pow == 0) {
        return 0;
    }
    
    matrix *tmp = NULL;
    int res = allocate_matrix(&tmp, n, n);
    if (res != 0) {
        return res;
    }
    
    // 逐次乘方计算
    for (int k = 0; k < pow; k++) {
        // 将 result 矩阵与输入矩阵相乘，结果存储在 tmp 矩阵中
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                tmp->data[i * n + j] = 0.0;
                for (int l = 0; l < n; l++) {
                    tmp->data[i * n + j] += result->data[i * n + l] * mat->data[l * n + j];
                }
            }
        }
        
        // 将 tmp 矩阵的值复制回 result 矩阵
        for (int i = 0; i < n * n; i++) {
            result->data[i] = tmp->data[i];
        }
    }
    
    deallocate_matrix(tmp);
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    for(int idx = 0;idx<(mat->cols*mat->rows);idx++){
        result->data[idx] = -mat->data[idx];
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    for(int idx = 0;idx<(mat->cols*mat->rows);idx++){
        result->data[idx] = abs(mat->data[idx]);
    }
    return 0;
}

