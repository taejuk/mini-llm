#pragma once
#include <cstdio>

namespace mini_llm::runtime {

class PhysicalBlock {
private:
    inline static int next_id = 0;  
    int id;
    int block_size;
    int hidden_dim;
    float* key_data;    
    float* value_data;  
    int refcount;
    int filled;

public:
    PhysicalBlock(int bsz, int hdim, float* k_data, float* v_data)
        : id(next_id++), block_size(bsz), hidden_dim(hdim),
          key_data(k_data), value_data(v_data), refcount(0), filled(0) {}

    void inc_ref() { refcount++; }
    void dec_ref() { if (refcount > 0) refcount--; }
    int  get_refcount() const { return refcount; }

    void reset() { refcount = 0; }

    float* get_key_at(int n)   const { return key_data   + n * hidden_dim; }
    float* get_value_at(int n) const { return value_data + n * hidden_dim; }

    float* get_key()   const { return key_data; }
    float* get_value() const { return value_data; }

    int get_block_size() const { return block_size; }
    int get_hidden_dim() const { return hidden_dim; }
    int get_id()         const { return id; }
    void add_kv(float* new_k, float* new_v) {}
    void add_kv_batch(float* new_k, float* new_v,int nums) {}
};

}