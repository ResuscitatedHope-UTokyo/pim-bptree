#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <random>
#include <unordered_set>

extern "C" {
#include <dpu.h>
#include <dpu_log.h>
#include "common.h"
}

#define N (500000 + 2500)

uint64_t n = N;
kvpair_t query_buffer[N];

bool compareKvPairs(const kvpair_t &a, const kvpair_t &b) {
    return a.key < b.key;
}

void make_query()
{
  std::mt19937 gen(74755); 
  std::uniform_int_distribution<int> dist(0, 2147483647); 
  
  std::unordered_set<int> unique_keys;
  unique_keys.reserve(N);

  for (int i = 0; i < N; i++) {
    int key;
    do {
        key = dist(gen);
    } while (unique_keys.count(key));
    unique_keys.insert(key);

    query_buffer[i].key = key;
    query_buffer[i].value = i;
  }

  std::shuffle(query_buffer, query_buffer + N, gen);
  std::sort(query_buffer, query_buffer + 500000, compareKvPairs);
  std::sort(query_buffer + 500000, query_buffer + N, compareKvPairs);
}

int main(int argc, char* argv[])
{
    struct dpu_set_t set;
    struct dpu_set_t dpu;

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <binary path>\n", argv[0]);
        exit(1);
    }

    make_query();
    
    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    DPU_ASSERT(dpu_load(set, argv[1], NULL));

    
    DPU_ASSERT(dpu_broadcast_to(set, "nr_queries", 0,
&n, sizeof(uint64_t),
DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, "query_buffer", 0,
query_buffer, sizeof(kvpair_t) * n,
DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    DPU_FOREACH(set, dpu)
    {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }
    DPU_ASSERT(dpu_free(set));

    return 0;
}
