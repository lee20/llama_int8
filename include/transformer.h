#ifndef TRANSFORMER_H

#define TRANSFORMER_H

#include<vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <immintrin.h>

typedef std::vector<float> tensor1d;
typedef std::vector<tensor1d> tensor2d;
typedef std::vector<tensor2d> tensor3d;

typedef std::vector<__m256i> tensor8b1d;
typedef std::vector<tensor8b1d> tensor8b2d;
typedef std::vector<tensor8b2d> tensor8b3d;

#define LAYER_NUM 32

struct Int8Weight {
    tensor8b3d weight8;
    tensor2d scale;
    tensor1d delta; 
};

struct TransformerWeights {
    tensor2d token_embedding_table;  // [vocab_size, dim]
    // weights for rmsnorms
    tensor2d rms_att_weight;  // [layer, dim]
    tensor2d rms_ffn_weight;  // [layer, dim]
    // weights for attention matmuls
    tensor3d wq;  // [layer, dim, dim]
    tensor3d wk;  // [layer, dim, dim]
    tensor3d wv;  // [layer, dim, dim]

    
    tensor3d wo;  // [layer, dim, dim]
    // weights for ffn
    tensor3d w1;  // [layer, hidden_dim, dim]
    tensor3d w2;  // [layer, dim, hidden_dim]
    tensor3d w3;  // [layer, hidden_dim, dim]
    // final rmsnorm
    tensor1d rms_final_weight;  // [dim]
    tensor2d wcls; //
    // freq_cis for RoPE relatively positional embeddings
    tensor2d freq_cis_image;  // [seq_len, (dim/n_heads)/2]
    tensor2d freq_cis_real;

    Int8Weight wq8;
    Int8Weight wk8;
    Int8Weight wv8;
    Int8Weight wo8;
    Int8Weight w18;
    Int8Weight w38;
    Int8Weight w28;
};

struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    float temprature;
    float top_p;
    int bit_length;
};

//tensor2d Attention(int layer_id, tensor2d input);

//tensor2d Feedforward(int layer_id, tensor2d input);

//tensor2d TransformerBlock(int layer_id, tensor2d input_token);

//tensor2d Transformer(tensor2d input_token);

tensor2d operator+(const tensor2d& mat1, const tensor2d& mat2);

//tensor2d Transformer(tensor2d input_token, Config configs, TransformerWeights weights);
int Transformer(const tensor2d &input_token, const TransformerWeights &weights, const Config &configs, int start=0, int end=0);

void ShowTensor2D(tensor2d mat);

void softmax(tensor1d &output, tensor1d &input, int max_pos = -1);

void PrintTime(std::chrono::time_point<std::chrono::high_resolution_clock> start,std::chrono::time_point<std::chrono::high_resolution_clock> end, std::string name);

#endif