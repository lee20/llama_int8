#include <string>
#include <fstream>
#include <iostream>
#include "transformer.h"
#include "load8bit.h"



void ReadWeight2Vector2D(std::string filename, tensor2d &vec){
    std::string path1 = "/home/liyanjun/llama/llama2/llama2chatweightfp32/" + filename;
    std::ifstream file(path1, std::ios::binary);  // 以二进制模式打开文件
    if (!file) {
        throw std::runtime_error("Failed to open the file. filename: "+path1);
    }
    int rows = vec.size();
    int cols = vec[0].size();
    size_t numElements = rows * cols;
    
    // 创建一个临时 vector 用于存储从文件中读取的一维数据
    std::vector<float> data(numElements);

    // 读取文件中的数据到一维 vector 中
    if (!file.read(reinterpret_cast<char*>(data.data()), numElements * sizeof(float))) {
        throw std::runtime_error("Failed to read the data from the file.");
    }

    file.close();  // 关闭文件

    // 将一维 vector 数据转成二维 vector
    vec.resize(rows, std::vector<float>(cols));  // 调整 vec 的大小
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            vec[i][j] = data[i * cols + j];  // 将数据放入二维 vector 中
        }
    }
}



void ReadActInfo(int index, tensor1d &vec, std::string name){
    std::ifstream file("model_weight/bin/" + name + std::to_string(index) + ".bin", std::ios::binary); // 以二进制模式打开文件
    size_t numElements = vec.size();

    // 读取文件中的数据到一维 vector 中
    if (!file.read(reinterpret_cast<char *>(vec.data()), numElements * sizeof(float)))
    {
        throw std::runtime_error("Failed to read the data from the file.");
    }

    file.close(); // 关闭文件

}

float ChangeWeight(const tensor2d &weight, tensor8b2d &weight8, tensor1d &act, tensor1d &s_coef)
{
    float alpha = 0.5;
    int column_len = act.size();
    std::vector<float> column_max(column_len,0);
    float quant_max = 0;
    float delta = 0;

    // 求列最大
    for(int i=0;i<weight.size();i++){
        for(int j=0;j<weight[0].size();j++){
            column_max[j] = (abs(weight[i][j]) > column_max[j]) ? abs(weight[i][j]) : column_max[j];
        }
    }
    // 除出系数
    for(int i=0;i<column_len;i++){
        s_coef[i] = pow(act[i] , alpha) / pow(column_max[i] , 1-alpha);
    }
    // 量化权重
    for(int i=0;i<weight.size();i++){
        for(int j=0;j<weight[0].size();j++){
            quant_max = (quant_max > abs(weight[i][j]* s_coef[j]))? quant_max : weight[i][j]* s_coef[j];
        }
    }
    // 保存量化的权重和scale
    delta = quant_max / 127.0;
    std::vector<int8_t> temp_weight(column_len,0);
    for(int i=0;i<weight.size();i++){
        for(int j=0;j<weight[0].size();j++){
            temp_weight[j] = static_cast<int8_t>(weight[i][j]* s_coef[j]/delta);
        }
        for(int j=0;j<weight[0].size();j+=32){
            weight8[i][j/32] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&temp_weight[j]));
        }
    }
    return delta;
}

void change8bit(TransformerWeights &weights)
{
    tensor3d wqtemp;
    tensor3d(32, tensor2d(4096, tensor1d(4096))).swap(wqtemp);
    
    tensor8b3d(32,tensor8b2d(4096,tensor8b1d(128,_mm256_setzero_si256()))).swap(weights.wq8.weight8);
    tensor8b3d(32,tensor8b2d(4096,tensor8b1d(128,_mm256_setzero_si256()))).swap(weights.wk8.weight8);
    tensor8b3d(32,tensor8b2d(4096,tensor8b1d(128,_mm256_setzero_si256()))).swap(weights.wv8.weight8);
    tensor8b3d(32,tensor8b2d(4096,tensor8b1d(128,_mm256_setzero_si256()))).swap(weights.wo8.weight8);
    tensor8b3d(32,tensor8b2d(11008,tensor8b1d(128,_mm256_setzero_si256()))).swap(weights.w18.weight8);
    tensor8b3d(32,tensor8b2d(11008,tensor8b1d(128,_mm256_setzero_si256()))).swap(weights.w38.weight8);
    tensor8b3d(32,tensor8b2d(4096,tensor8b1d(344,_mm256_setzero_si256()))).swap(weights.w28.weight8);
    tensor2d(32,tensor1d(4096,0)).swap(weights.wq8.scale);
    tensor2d(32,tensor1d(4096,0)).swap(weights.wk8.scale);
    tensor2d(32,tensor1d(4096,0)).swap(weights.wv8.scale);
    tensor2d(32,tensor1d(4096,0)).swap(weights.wo8.scale);
    tensor2d(32,tensor1d(4096,0)).swap(weights.w18.scale);
    tensor2d(32,tensor1d(4096,0)).swap(weights.w38.scale);
    tensor2d(32,tensor1d(11008,0)).swap(weights.w28.scale);
    tensor1d(32,0).swap(weights.wq8.delta);
    tensor1d(32,0).swap(weights.wk8.delta);
    tensor1d(32,0).swap(weights.wv8.delta);
    tensor1d(32,0).swap(weights.wo8.delta);
    tensor1d(32,0).swap(weights.w18.delta);
    tensor1d(32,0).swap(weights.w38.delta);
    tensor1d(32,0).swap(weights.w28.delta);


    std::vector<float> actinfo(4096,0);
    std::vector<float> actinfo2(11008,0);

    
    for (int i = 0; i < 32; i++)
    {
        ReadActInfo(i,actinfo,"statx");  
        weights.wq8.delta[i] = ChangeWeight(weights.wq[i],weights.wq8.weight8[i],actinfo,weights.wq8.scale[i]);
        weights.wk8.delta[i] = ChangeWeight(weights.wk[i],weights.wk8.weight8[i],actinfo,weights.wk8.scale[i]);
        weights.wv8.delta[i] = ChangeWeight(weights.wv[i],weights.wv8.weight8[i],actinfo,weights.wv8.scale[i]);

        ReadActInfo(i,actinfo,"stato");
        weights.wo8.delta[i] = ChangeWeight(weights.wo[i],weights.wo8.weight8[i],actinfo,weights.wo8.scale[i]);

        ReadActInfo(i,actinfo,"stat1");
        weights.w18.delta[i] = ChangeWeight(weights.w1[i],weights.w18.weight8[i],actinfo,weights.w18.scale[i]);
        weights.w38.delta[i] = ChangeWeight(weights.w3[i],weights.w38.weight8[i],actinfo,weights.w38.scale[i]);

        ReadActInfo(i,actinfo2,"stat2");
        weights.w28.delta[i] = ChangeWeight(weights.w2[i],weights.w28.weight8[i],actinfo2,weights.w28.scale[i]);
    }
}