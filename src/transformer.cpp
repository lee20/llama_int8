#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <limits>
#include <string>
#include <cassert>
#include <chrono>
#include <cstring>
#include "transformer.h"

tensor3d g_kcache(32,tensor2d(4096,tensor1d(4096,0)));
tensor3d g_vcache(32,tensor2d(4096,tensor1d(4096,0)));

float EPS = 1e-5;
std::string filepath = "/home/liyanjun/llama/llama2/llama2chatweightfp32/";

void PrintTime(std::chrono::time_point<std::chrono::high_resolution_clock> start,std::chrono::time_point<std::chrono::high_resolution_clock> end, std::string name){
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << name<< " cost " << duration.count() << " us" << std::endl;
}


void ShowTensor2D(std::string variable_name, tensor2d mat){
    int print_length = (mat.size() < mat[0].size())?mat.size() : mat[0].size();
    std::cout<<"tensor[0][10:15] = {";
    for(int i=10;i<20;i++){
        std::cout<< mat[0][i] << ",";
    }
    std::cout << "}"<<std::endl;
}


tensor2d operator+(const tensor2d& mat1, const tensor2d& mat2) {
    // 检查行数是否相同
    if (mat1.size() != mat2.size()) {
        throw std::invalid_argument("Matrices must have the same number of rows.");
    }

    // 检查每一行的列数是否相同
    for (size_t i = 0; i < mat1.size(); ++i) {
        if (mat1[i].size() != mat2[i].size()) {
            throw std::invalid_argument("Matrices must have the same number of columns.");
        }
    }

    // 初始化结果矩阵，大小与mat1相同
    tensor2d result(mat1.size(), std::vector<float>(mat1[0].size()));

    // 逐元素相加
    for (size_t i = 0; i < mat1.size(); ++i) {
        for (size_t j = 0; j < mat1[i].size(); ++j) {
            result[i][j] = mat1[i][j] + mat2[i][j];
        }
    }

    return result;  // 返回相加后的结果矩阵
}


// 矩阵的对应位乘
tensor2d operator*(const tensor2d& mat1, const tensor2d& mat2) {
    // 检查行数是否相同
    if (mat1.size() != mat2.size()) {
        throw std::invalid_argument("Matrices must have the same number of rows.");
    }
    for (size_t i = 0; i < mat1.size(); ++i) {
        if (mat1[i].size() != mat2[i].size()) {
            throw std::invalid_argument("Matrices must have the same number of columns.");
        }
    }

    // 初始化结果矩阵，大小与mat1相同
    tensor2d result(mat1.size(), std::vector<float>(mat1[0].size()));

    // 逐元素相乘 
    for (size_t i = 0; i < mat1.size(); ++i) {
        for (size_t j = 0; j < mat1[i].size(); ++j) {
            result[i][j] = mat1[i][j] * mat2[i][j];
        }
    }
    return result;  // 返回相加后的结果矩阵
}

// // 矩阵向量乘
// void MatVecMul(tensor1d &output, const tensor2d &mat, const tensor1d &vec){
//     assert(mat[0].size() == vec.size());

//     int cols = mat[0].size();
//     int rows = mat.size();
//     float temp = 0;


//     for(int i=0;i<rows;i++){
//         temp = 0;
//         for(int j=0; j<cols; j++){
//             temp += mat[i][j]*vec[j];
//         }
//         output[i] = temp;
//     }
// }


void avx_matrix_vector_multiply(tensor1d &results, const tensor2d &A, const tensor1d &x) {
    int rows = A.size();
    int cols = A[0].size();
    int length = cols/8;
    assert(cols%8 == 0); //8对齐是后面注释掉无法被8整除的那部分的前提
    __m256 vecX[cols/8];
    float result[8];

    for(int i=0;i<length;i++){
        vecX[i] = _mm256_loadu_ps(&x[i*8]);
    }


    for (int i = 0; i < rows; ++i) {
        __m256 sum = _mm256_setzero_ps();  // 初始化累加器

        // 按 8 元素一组进行计算
        int j = 0;

        for (; j <length; j += 8) {
            
            __m256 vecA1 = _mm256_loadu_ps(&A[i][j << 3]);         // 加载第一个 256 位块
            __m256 vecA2 = _mm256_loadu_ps(&A[i][(j + 1) << 3]);   // 加载第二个 256 位块
            __m256 vecA3 = _mm256_loadu_ps(&A[i][(j + 2) << 3]);
            __m256 vecA4 = _mm256_loadu_ps(&A[i][(j + 3) << 3]);
            __m256 vecA5 = _mm256_loadu_ps(&A[i][(j + 4) << 3]);
            __m256 vecA6 = _mm256_loadu_ps(&A[i][(j + 5) << 3]);
            __m256 vecA7 = _mm256_loadu_ps(&A[i][(j + 6) << 3]);
            __m256 vecA8 = _mm256_loadu_ps(&A[i][(j + 7) << 3]);

            sum = _mm256_fmadd_ps(vecA1, vecX[j], sum);           // 执行第一个块的乘加操作



            sum = _mm256_fmadd_ps(vecA2, vecX[j + 1], sum);       // 执行第二个块的乘加操作
            sum = _mm256_fmadd_ps(vecA3, vecX[j + 2], sum);
            sum = _mm256_fmadd_ps(vecA4, vecX[j + 3], sum);
            if(i==0 && j == 0){
                 _mm256_storeu_ps(result, sum);
                // 累加 SIMD 的结果和尾部的部分结果
                float ff = result[0] + result[1] + result[2] + result[3] +  result[4] + result[5] + result[6] + result[7];
                printf("%f\n",ff);
            
            }

            sum = _mm256_fmadd_ps(vecA5, vecX[j + 4], sum);
            sum = _mm256_fmadd_ps(vecA6, vecX[j + 5], sum);
            sum = _mm256_fmadd_ps(vecA7, vecX[j + 6], sum);
            sum = _mm256_fmadd_ps(vecA8, vecX[j + 7], sum);

            // __builtin_prefetch(&A[i][(j + 8 + 8) << 3], 0, 1);
            // __builtin_prefetch(&A[i][(j + 8 + 10) << 3], 0, 1);
            // __builtin_prefetch(&A[i][(j + 8 + 12) << 3], 0, 1);
            // __builtin_prefetch(&A[i][(j + 8 + 14) << 3], 0, 1);
        }


        // 处理余下无法整除 8 的部分
        // float partial_sum = 0.0f;
        // j = j * 8;
        // for (; j < cols; ++j) {
        //     partial_sum += A[i][ j] * x[j];
        // }

        // 将 SIMD 寄存器的值存储到数组中
        
        _mm256_storeu_ps(result, sum);
        // 累加 SIMD 的结果和尾部的部分结果
        results[i] = result[0] + result[1] + result[2] + result[3] +  result[4] + result[5] + result[6] + result[7];
    }
}


float QuantizeX(const tensor1d & input, const tensor1d &input_scale, tensor8b1d &output){
    assert(input.size() == input_scale.size());
    tensor1d tempvector(4096,0);
    std::vector<int8_t> temp0(4096,0);
    float max_abs = 0;
    
    for(int i=0;i<input.size();i++){
        //printf("input_scale = %f,%f\n",input[i] , input_scale[i]);
        tempvector[i] = input[i] / input_scale[i];
        if(abs(tempvector[i]) > max_abs){
            max_abs = abs(tempvector[i]);
        }
    }
    
    float delta = max_abs/127.0;
    
    for(int i=0;i<input.size();i++){
        temp0[i] = static_cast<int8_t>(tempvector[i]/delta);
    }
    
    for(int i=0;i<input.size();i+=32){
        output[i/32] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(temp0.data() + i));
    }
    return delta;

}


void avx_matrix_vector_multiply8b(int layer_id, tensor1d &results, const TransformerWeights &weights, const tensor1d &x) {
    // 量化X
    tensor8b1d input8b(128,_mm256_setzero_si256());

    float deltax = QuantizeX(x, weights.scale_q[layer_id],input8b);

    float delatw = weights.delta_q[layer_id];

    // 实现计算
    for (int i = 0; i < 4096; ++i) {

        __m256i result1 = _mm256_setzero_si256();
        int q[8];
        int sum2 = 0;
        // 里面的数可以进一步循环展开
        for(int j=0; j<128;j++){
            __m256i w = weights.wq8[layer_id][i][j];
            __m256i x = input8b[j];

            __m256i x_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 0));
            __m256i x_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1));
            __m256i w_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 0));
            __m256i w_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
            auto val1 = _mm256_madd_epi16(x_low, w_low);
            auto val2 = _mm256_madd_epi16(x_high, w_high);
            
            result1 = _mm256_add_epi32(result1, val1);
            result1 = _mm256_add_epi32(result1, val2);

            if(i == 0 && j == 0){
                _mm256_store_si256((__m256i*)q, result1);
                sum2 = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];

                results[i] = deltax*delatw*sum2;

                printf("%d,%f\n",sum2,results[i]);
            }

        }
        _mm256_store_si256((__m256i*)q, result1);
        sum2 = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];

        results[i] = deltax*delatw*sum2;
    }

}




// n*4096
void MatMul(tensor2d &output, const tensor2d &input,const tensor2d & weight){
    // omp_set_num_threads(30);
    // #pragma omp parallel for
    for(int i=0;i<input.size();i++){
        avx_matrix_vector_multiply(output[i],weight,input[i]);
    }
}

// n*4096
void MatMul8bit(int layerid, tensor2d &output, const tensor2d &input,const TransformerWeights & weights){
    // omp_set_num_threads(30);
    // #pragma omp parallel for
    //std::cout<<"layer id"<<layerid<<std::endl;
    for(int i=0;i<input.size();i++){
        avx_matrix_vector_multiply8b(layerid,output[i],weights,input[i]);
    }
    //std::cout<<"end layer id"<<layerid<<std::endl;
}


void rmsnorm(tensor1d &output, const tensor1d &input, const tensor1d &weight) {
    float ss = 0.0;
    for (int i = 0; i < input.size(); i++)
        ss += input[i] * input[i];
    ss = ss / input.size() + EPS;
    float inv_ss = 1 / sqrt(ss);
    for (int i = 0; i < input.size(); i++)
        output[i] = input[i] * inv_ss * weight[i];
}

void softmax(tensor1d &output, tensor1d &input, int max_pos) {
    if (max_pos == -1)  max_pos = input.size();
    float max_val = input[0];
    for (int i = 1; i < max_pos; i++)
        if (input[i] > max_val)  max_val = input[i];
    
    // exp and sum
    float sum = 0;
    for (int i = 0; i < max_pos; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    // normalize
    for (int i = 0; i < max_pos; i++)
        output[i] /= sum;
}

void applyRoPE(int seq_id, tensor1d & input, const TransformerWeights & weights){
    //tensor1d weight = weights.freq_cis;
    int head_size = 128;
    tensor1d rope_result(input);
    for (int head = 0; head < 32; ++head) {
        int start = head * head_size;
        for (int i = 0; i < head_size; i += 2) {
            float q0 = input[start + i];
            float q1 = input[start + i + 1];
            float fcr = weights.freq_cis_real[seq_id][i / 2];
            float fci = weights.freq_cis_image[seq_id][i / 2];
            rope_result[start + i]     = q0 * fcr - q1 * fci;
            rope_result[start + i + 1] = q0 * fci + q1 * fcr;
        }
    }
    memcpy(input.data(), rope_result.data(), 4096*sizeof(float));
}


//这里q指的是本级大小，KV指的是全部KVCache的大小
tensor3d MultiheadQKV(const tensor2d & q_tensor, const tensor2d & k_tensor, const tensor2d &v_tensor, int startpos, int endpos){
    
    //注意这里用q_tensor而不是用start和end中间的数据是否合适?
    int sizeq = q_tensor.size();
    tensor3d scores(32,tensor2d(sizeq,tensor1d(endpos,0)));
    float mul_coef = 1/sqrt(128.0);

    omp_set_num_threads(100);
    #pragma omp parallel for
    for(int i=0;i<32;i++){
        int ff = i*128;
        for(int j=0;j<sizeq;j++){
            for(int k=0;k<endpos;k++){
                float temp = 0;
                for(int m=0;m<128;m++){
                    temp += q_tensor[j][ff+m]*k_tensor[k][ff+m];
                }
                scores[i][j][k] = temp*mul_coef;
            }
        }
    }

    
    return scores;

}


tensor2d Attention(int layer_id, const tensor2d & input,const TransformerWeights &weights, int start_pos, int end_pos){
    

    auto start = std::chrono::high_resolution_clock::now();
    
    
    tensor2d input_rms(input);
    //ShowTensor2D("input",input);

    for(int i=0;i<input.size();i++){
        rmsnorm(input_rms[i],input[i],weights.rms_att_weight[layer_id]);
    }
    if(layer_id == 27)
        ShowTensor2D("inputrms",input_rms);


    tensor2d q(input.size(),tensor1d(input[0].size(),0));
    tensor2d k(input.size(),tensor1d(input[0].size(),0));
    tensor2d v(input.size(),tensor1d(input[0].size(),0));
    tensor2d out(input.size(),tensor1d(input[0].size(),0));

    auto init = std::chrono::high_resolution_clock::now();
    
    //std::cout<<"attention matmul started"<<std::endl;
    MatMul(q, input_rms, weights.wq[layer_id]);
    auto mm1 = std::chrono::high_resolution_clock::now();
    ShowTensor2D("q",q);
    MatMul8bit(layer_id, q, input_rms, weights);
    auto mm2 = std::chrono::high_resolution_clock::now();
    ShowTensor2D("q",q);

    PrintTime(init,mm1,"matmul");
    PrintTime(mm1,mm2,"matmul2");
    exit(0);

    MatMul(k, input_rms, weights.wk[layer_id]);
    MatMul(v, input_rms, weights.wv[layer_id]);
    //std::cout<<"attention matmul finished"<<std::endl;
    auto qkv = std::chrono::high_resolution_clock::now();


    for(int i=0;i<q.size();i++){
        applyRoPE(start_pos + i,q[i],weights);
        applyRoPE(start_pos + i,k[i],weights);
    }
    // ShowTensor2D("q",q);
    // ShowTensor2D("k",k);
    // ShowTensor2D("v",v);

    //存储kvcache
    for(int i=0;i<q.size();i++){
        g_kcache[layer_id][start_pos + i] = k[i];
        g_vcache[layer_id][start_pos + i] = v[i];
    }

    auto scores = MultiheadQKV(q,g_kcache[layer_id],g_vcache[layer_id],start_pos,end_pos);
    //ShowTensor2D("score1",scores[0]);

    for(int i=0;i<32;i++){
        if(q.size() > 1){
            for(int j=0;j<q.size();j++){
                for(int k=0;k<end_pos;k++){
                    if(k>j) scores[i][j][k] = -std::numeric_limits<float>::infinity();
                }
            }
        }
        for(int j=0;j<q.size();j++){
            tensor1d temp_softmax(scores[i][j]);
            softmax(temp_softmax,scores[i][j]);
            scores[i][j] = temp_softmax;
        }
    }
    //ShowTensor2D("score2",scores[0]);
    auto multihead = std::chrono::high_resolution_clock::now();

    tensor2d output1(input);
    for(int m=0;m<q.size();m++){
        for(int i=0;i<32;i++){
            for(int j=0;j<128;j++){
                float temp = 0;
                for(int k=0;k<end_pos;k++){
                    temp += scores[i][m][k]*g_vcache[layer_id][k][i*128+j];
                }
                output1[m][i*128+j] = temp;
            }
        }
    }
    auto output1_after = std::chrono::high_resolution_clock::now();

    //ShowTensor2D("o1",output1);

    MatMul(out, output1, weights.wo[layer_id]);

    auto last_mul = std::chrono::high_resolution_clock::now();
    
        // PrintTime(start,init,"init");
        // PrintTime(init,qkv,"qkv");
        // PrintTime(qkv,multihead,"multihead");
        // PrintTime(multihead,last_mul,"last_mul");
        // PrintTime(start,last_mul,"attention");
    
    
    return out;
}

tensor2d Feedforward(int layer_id, tensor2d input,const TransformerWeights &weights){
    auto start1 = std::chrono::high_resolution_clock::now();
    
    tensor2d input_rms(input);
    for(int i=0;i<input.size();i++){
        rmsnorm(input_rms[i],input[i],weights.rms_ffn_weight[layer_id]);
    }
    //ShowTensor2D("ffn inputrms",input_rms);
    
    tensor2d w1_result(input.size(),tensor1d(11008,0 ));
    tensor2d w3_result(input.size(),tensor1d(11008,0));
    tensor2d silu_result(input.size(),tensor1d(11008,0));
    tensor2d out(input.size(),tensor1d(input[0].size(),0));
    auto start2 = std::chrono::high_resolution_clock::now();
    MatMul(w3_result, input_rms, weights.w3[layer_id]);
    MatMul(w1_result, input_rms, weights.w1[layer_id]);
    //ShowTensor2D("w3_result",w3_result);
    //ShowTensor2D("w1_result",w1_result);
    auto start3 = std::chrono::high_resolution_clock::now();
    
    for(int i=0;i<w1_result.size();i++){
        for(int j=0;j<w1_result[0].size();j++)
            w1_result[i][j] = w1_result[i][j] * (1.0 / (1.0 + std::exp(-w1_result[i][j])));
    }
    
    auto result0 = w1_result * w3_result;
    auto start4 = std::chrono::high_resolution_clock::now();
    //ShowTensor2D("result0",result0);
    MatMul(out, result0, weights.w2[layer_id]);
    //ShowTensor2D("w2",out);
    auto start5 = std::chrono::high_resolution_clock::now();
    
    //if(layer_id == 7 && input.size() == 1){
        // PrintTime(start1,start2,"init");
        // PrintTime(start2,start3,"w1w3");
        // PrintTime(start3,start4,"silu");
        // PrintTime(start4,start5,"wo");
        // PrintTime(start1,start5,"ffn");
    //}
    
    
    return out;
}

tensor2d TransformerBlock(int layer_id, tensor2d & input_token, const TransformerWeights &weights, int start, int end){
    tensor2d output_token = input_token + Attention(layer_id,input_token,weights,start,end);
    //FeedForward和start_end无关
    output_token = output_token + Feedforward(layer_id,output_token,weights);
    return output_token;
}




int sample_top_p(tensor1d &probs, float p){
    std::vector<int> indices(probs.size());
    for (int i = 0; i < probs.size(); ++i) {
        indices[i] = i;
    }
    // 对下标数组进行排序，比较函数依据原始向量中的值排序
    std::sort(indices.begin(), indices.end(), [&probs](int i1, int i2) {
        return probs[i1] > probs[i2];
    });

    tensor1d cumsum(probs.size());
    std::vector<int> mask(indices.size());
    float sum = 0;
    float clean_sum = 0;
    for(int i=0;i<probs.size();i++){
        sum = sum + probs[indices[i]];
        //下面几行有逻辑优化空间，注意优化，优化后删除此注释
        cumsum[indices[i]] = sum;
        mask[indices[i]] = (cumsum[indices[i]] - probs[indices[i]] > p)?1:0;
        if(mask[indices[i]] > 0) probs[indices[i]] = 0;
        clean_sum += probs[indices[i]];
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // 生成一个随机浮点数
    float random_value = (float)dis(gen);
    
    sum = 0;
    for(int i=0;i<probs.size();i++){
        probs[i] = probs[i]/clean_sum;
        sum += probs[i];
        if(sum > random_value) return i;
    }
    return 0;

}

// input : temp
// input : vec2d n*4096
int get_next_token(tensor2d &vec, const Config &config){
    auto vec2 = vec[vec.size() - 1];
    tensor1d vec_softmax(vec2);
    for(int i=0;i<vec2.size();i++){
        vec2[i] = vec2[i]/config.temprature;
    }
    softmax(vec_softmax,vec2);

    return sample_top_p(vec_softmax,config.top_p);
}


int Transformer(const tensor2d &input_token, const TransformerWeights &weights, const Config &configs, int start, int end){
    tensor2d h = input_token;
    tensor2d output_rms(h);
    tensor2d output_transformer(h.size(),tensor1d(weights.wcls.size(),0));
    for(int i=0;i<LAYER_NUM;i++){
        h = TransformerBlock(i,h,weights,start,end);
    }
    
    for(int i=0;i<h.size();i++){
        rmsnorm(output_rms[i],h[i],weights.rms_final_weight);
    }
    
    MatMul(output_transformer,output_rms,weights.wcls);


    int next_token = get_next_token(output_transformer, configs);

    return next_token;
}