#include <cstdio>
#include <cstdlib>
extern "C" {
#include <dpu.h>
#include <dpu_log.h>
#include "common.h"
}

#define N 1000

uint64_t n = N;
kvpair_t query_buffer[N];

void make_query()
{
  int seed = 74755;
  for (int i = 0; i < N; i++) {
    query_buffer[i].key = seed;
    query_buffer[i].value = i;
    seed = (seed * 1309 + 13849) & 65535;
  }
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
