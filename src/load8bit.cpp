#include <string>
#include <fstream>
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



void ReadActInfo(int index, tensor1d &vec){
    std::ifstream file("model_weight/bin/statx" + std::to_string(index) + ".bin", std::ios::binary); // 以二进制模式打开文件
    size_t numElements = vec.size();

    // 读取文件中的数据到一维 vector 中
    if (!file.read(reinterpret_cast<char *>(vec.data()), numElements * sizeof(float)))
    {
        throw std::runtime_error("Failed to read the data from the file.");
    }

    file.close(); // 关闭文件

}

float ChangeWeight(tensor2d &src, tensor8b2d &dst, tensor1d &act, tensor1d &s_coef)
{
    float alpha = 0.5;
    std::vector<float> column_max(4096,0);
    float quant_max = 0;
    float delta = 0;

    // 求列最大
    for(int i=0;i<src.size();i++){
        for(int j=0;j<src[0].size();j++){
            column_max[j] = (abs(src[i][j]) > column_max[j]) ? abs(src[i][j]) : column_max[j];
        }
    }
    // 除出系数
    for(int i=0;i<4096;i++){
        s_coef[i] = pow(act[i] , alpha) / pow(column_max[i] , 1-alpha);
    }
    // 量化权重
    
    for(int i=0;i<src.size();i++){
        for(int j=0;j<src[0].size();j++){
            src[i][j] = src[i][j] * s_coef[j];
            
            quant_max = (quant_max > abs(src[i][j]))? quant_max : src[i][j];
        }
    }
    // 保存量化的权重和scale
    delta = quant_max / 127.0;
    std::vector<int8_t> temp_weight(4096,0);
    for(int i=0;i<src.size();i++){
        for(int j=0;j<src[0].size();j++){
            temp_weight[j] = static_cast<int8_t>(src[i][j]/delta);
        }
        for(int j=0;j<src[0].size();j+=32){
            dst[i][j/32] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[i][j]));
        }
    }


    return delta;

    //t1 = torch.load("statx.pt").detach().cpu().type(torch.float)
    // weight = torch.load("weight.pt").detach().cpu().type(torch.float)
    // random_tensor = torch.rand(4096)
    // x = random_tensor * 2 - 1
    // max_values = torch.max(torch.abs(weight), dim=0)[0]
    // s = torch.pow(t1,alpha) / torch.pow(max_values,1-alpha)
    // input_equ = x * s
    // weight_equ = weight/s
    // deltaq, weightq = quantint8(weight_equ)
    // deltax, xqq = quantint8(input_equ)
}

void load8bit(TransformerWeights &weights)
{
    tensor3d wqtemp;
    tensor3d(32, tensor2d(4096, tensor1d(4096))).swap(wqtemp);
    tensor8b3d(32,tensor8b2d(4096,tensor8b1d(128,_mm256_setzero_si256()))).swap(weights.wq8);
    tensor2d(32,tensor1d(4096,0)).swap(weights.scale_q);
    tensor1d(32,0).swap(weights.delta_q);


    std::vector<float> actinfo(4096,0);
    
    
    for (int i = 0; i < 32; i++)
    {
        ReadWeight2Vector2D("layers." + std::to_string(i) + ".attention.wq.weight", wqtemp[i]);
        ReadActInfo(i,actinfo);

        float delta_weight = ChangeWeight(weights.wq[i],weights.wq8[i],actinfo,weights.scale_q[i]);
        weights.delta_q[i] = delta_weight;
    }
}