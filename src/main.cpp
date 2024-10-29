#include <string>
#include <fstream>
#include <tokenizers_cpp.h>
#include <iostream>
#include <string>
#include "transformer.h"
#include "load8bit.h"
#include <chrono>
#include <random>
#include <omp.h>
#include <algorithm>

using tokenizers::Tokenizer;


// 定义
std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        std::cerr << "Cannot open " << path << std::endl;
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}



tensor2d LoadEmbeddingData(const std::string& path){
    int rows = 32000, cols = 4096;
    tensor2d result(rows, tensor1d(cols,0));
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file!" <<path <<  std::endl;
        return {};
    }
    
    float buffer;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            file.read(reinterpret_cast<char*>(&buffer), sizeof(buffer));
            result[i][j] = buffer; // Convert half to float
        }
    }

    file.close();
    return result;

}


std::vector<int> TokenizerGenerator(std::string input_str){
    // 转换为token数组
    std::string input_string = "[INST] " + input_str + " [/INST]";
    auto blob = LoadBytesFromFile("./model_weight/tokenizer.model");
    auto tok = Tokenizer::FromBlobSentencePiece(blob);
    std::vector<int> ids = tok->Encode(input_string);
    ids.insert(ids.begin(),1);

    return ids;
}




std::string Detokenizer(std::vector<int> nums){
    auto blob = LoadBytesFromFile("./model_weight/tokenizer.model");
    auto tok = Tokenizer::FromBlobSentencePiece(blob);
    std::string decoded_prompt = tok->Decode(nums);
    return decoded_prompt;
}


tensor2d Embed(const std::vector<int> & ids, const tensor2d &embedding_dict){
    tensor2d result(0, tensor1d(embedding_dict[0].size(),0));
    for(auto id : ids){
        result.push_back(embedding_dict[id]);
    }
    return result;
}



void InitConfig(Config &config){
    config.dim = 4096;
    config.hidden_dim = 11008;
    config.n_layers = 32;
    config.n_heads = 32;
    config.n_kv_heads = 32;
    config.vocab_size = 32000;
    config.seq_len = 4096;
    config.temprature = 0.5;
    config.top_p = 0.9;
    config.bit_length = 8;
}




void ReadWeight2Vector1D(std::string filename, tensor1d &vec){
    std::string path1 = "/home/liyanjun/llama/llama2/llama2chatweightfp32/" + filename;
    std::ifstream file(path1, std::ios::binary);  // 以二进制模式打开文件
    if (!file) {
        throw std::runtime_error("Failed to open the file. filename: "+path1);
    }
    int rows = vec.size();
    size_t numElements = rows;
    
    // 创建一个临时 vector 用于存储从文件中读取的一维数据
    std::vector<float> data(numElements);

    // 读取文件中的数据到一维 vector 中
    if (!file.read(reinterpret_cast<char*>(data.data()), numElements * sizeof(float))) {
        throw std::runtime_error("Failed to read the data from the file : "+filename);
    }

    file.close();  // 关闭文件
    for (size_t i = 0; i < rows; ++i) {
        vec[i] = data[i];
    }
}

void ReadWeight1D8(std::string filename, tensor1d &vec){

    std::ifstream file(filename, std::ios::binary);  // 以二进制模式打开文件
    if (!file) {
        throw std::runtime_error("Failed to open the file. filename: "+filename);
    }
    int rows = vec.size();
    size_t numElements = rows;
    
    // 创建一个临时 vector 用于存储从文件中读取的一维数据
    std::vector<float> data(numElements);

    // 读取文件中的数据到一维 vector 中
    if (!file.read(reinterpret_cast<char*>(data.data()), numElements * sizeof(float))) {
        throw std::runtime_error("Failed to read the data from the file : "+filename);
    }

    file.close();  // 关闭文件
    for (size_t i = 0; i < rows; ++i) {
        vec[i] = data[i];
    }
}

void LoadWeight32(Config &config, TransformerWeights &weights){

    int head_size = config.dim / config.n_heads;
    tensor2d(config.vocab_size, tensor1d(config.dim)).swap(weights.token_embedding_table);
    tensor2d(config.n_layers, tensor1d(config.dim)).swap(weights.rms_att_weight);
    tensor2d(config.n_layers, tensor1d(config.dim)).swap(weights.rms_ffn_weight);
    tensor1d(config.dim).swap(weights.rms_final_weight);
    tensor2d(8192,tensor1d(head_size / 2)).swap(weights.freq_cis_image);
    tensor2d(8192,tensor1d(head_size / 2)).swap(weights.freq_cis_real);
    tensor2d(config.n_layers, tensor1d(config.dim)).swap(weights.rms_ffn_weight);
    tensor2d(config.vocab_size,tensor1d(config.dim)).swap(weights.wcls);

    for(int i=0;i<config.n_layers;i++){
        ReadWeight2Vector1D("layers." + std::to_string(i)+".attention_norm.weight",weights.rms_att_weight[i]);
        ReadWeight2Vector1D("layers." + std::to_string(i)+".ffn_norm.weight",weights.rms_ffn_weight[i]);
    }
    ReadWeight2Vector2D("tok_embeddings.weight",weights.token_embedding_table);
    ReadWeight2Vector1D("norm.weight",weights.rms_final_weight);
    ReadWeight2Vector2D("output.weight",weights.wcls);
    ReadWeight2Vector2D("real.bin",weights.freq_cis_real);
    ReadWeight2Vector2D("imag.bin",weights.freq_cis_image);

    if(config.bit_length == 32){
        tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wq);
        tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wk);
        tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wv);
        tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wo);
        tensor3d(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim))).swap(weights.w1);
        tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.hidden_dim))).swap(weights.w2);
        tensor3d(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim))).swap(weights.w3);

        for(int i=0;i<config.n_layers;i++){
            ReadWeight2Vector2D("layers." + std::to_string(i)+".attention.wq.weight",weights.wq[i]);
            ReadWeight2Vector2D("layers." + std::to_string(i)+".attention.wk.weight",weights.wk[i]);
            ReadWeight2Vector2D("layers." + std::to_string(i)+".attention.wv.weight",weights.wv[i]);
            ReadWeight2Vector2D("layers." + std::to_string(i)+".attention.wo.weight",weights.wo[i]);
            ReadWeight2Vector2D("layers." + std::to_string(i)+".feed_forward.w1.weight",weights.w1[i]);
            ReadWeight2Vector2D("layers." + std::to_string(i)+".feed_forward.w2.weight",weights.w2[i]);
            ReadWeight2Vector2D("layers." + std::to_string(i)+".feed_forward.w3.weight",weights.w3[i]);
        }
    }
}

void ReadWeight2D8(std::string filename, tensor8b2d &vec){
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    int rows = vec.size();
    int cols = vec[0].size();

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // 创建一个 32 字节的缓冲区以读取数据
            alignas(32) int8_t temp[32];
            inFile.read(reinterpret_cast<char*>(temp), 32);
            // 将缓冲区数据加载到 __m256i
            vec[i][j] = _mm256_load_si256(reinterpret_cast<const __m256i*>(temp));
        }
    }

    inFile.close();
    if (!inFile.good()) {
        std::cerr << "读取文件时出错: " << filename << std::endl;
    }
}
void ReadWeight8bit(int layerid, std::string filename, Int8Weight &weight){
    ReadWeight2D8("./model_weight/weight/weight."+filename+"."+std::to_string(layerid)+".bin", weight.weight8[layerid]);
    ReadWeight1D8("./model_weight/weight/scale."+filename+"."+std::to_string(layerid)+".bin", weight.scale[layerid]);

    tensor1d delta(32,0);
    ReadWeight1D8("./model_weight/weight/delta."+filename+".bin", delta);
    weight.delta[layerid] = delta[layerid];

}


void LoadWeight8(Config &config, TransformerWeights &weights){
    if(config.bit_length == 32) return;
    
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

    for(int i=0;i<config.n_layers;i++){
        ReadWeight8bit(i, "wq",weights.wq8);
        ReadWeight8bit(i, "wk",weights.wk8);
        ReadWeight8bit(i, "wv",weights.wv8);
        ReadWeight8bit(i, "wo",weights.wo8);
        ReadWeight8bit(i, "w1",weights.w18);
        ReadWeight8bit(i, "w3",weights.w38);
        ReadWeight8bit(i, "w2",weights.w28);
    }
}




int main(){
    
    // 输入数据
    std::string input_string = "Tell a story about Greece mythology, with an element of jokes, an element of Aphrodite, the goddess of love, and take place on the island of Cyprus.";
    std::string embedding_filepath = "/home/liyanjun/llama/llama2/llama2chatweightfp32/tok_embeddings.weight.bin";

    // 加载数据和配置
    auto start = std::chrono::high_resolution_clock::now();
    TransformerWeights transformer_weights;
    Config llama2_config;
    InitConfig(llama2_config);

    LoadWeight32(llama2_config, transformer_weights);
    
    LoadWeight8(llama2_config, transformer_weights);
    auto ids = TokenizerGenerator(input_string);
    auto embedded = Embed(ids, transformer_weights.token_embedding_table);
    // if(llama2_config.bit_length == 32) //这个表示初始量化
    //     change8bit(transformer_weights);
    auto end = std::chrono::high_resolution_clock::now();
    PrintTime(start,end,"Load 8bit weight");


    
    // prefill阶段
    start = std::chrono::high_resolution_clock::now();
    int next_token = Transformer(embedded, transformer_weights, llama2_config, 0, embedded.size());
    end = std::chrono::high_resolution_clock::now();
    PrintTime(start, end, "Prefill");
    
    // decode
    int start_pos = embedded.size();
    std::vector<int> result;
    while(1){
        std::string result_str = Detokenizer(std::vector<int>(1,next_token));        
        result.push_back(next_token);
        
        tensor2d input_vec = Embed(std::vector<int>(1,next_token), transformer_weights.token_embedding_table);
        start = std::chrono::high_resolution_clock::now();
        next_token = Transformer(input_vec,transformer_weights,llama2_config,start_pos,start_pos+1);
        end = std::chrono::high_resolution_clock::now();
        start_pos += 1;
        
        PrintTime(start,end,"Decode"+std::to_string(next_token));
    }




    return 0;
}
