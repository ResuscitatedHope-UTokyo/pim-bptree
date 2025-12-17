#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include "../common.h"

__mram_noinit_keep uint64_t nr_queries;
__mram_noinit_keep kvpair_t query_buffer[MAX_QUERIES];

__mram_ptr uint8_t *mram_free = __sys_used_mram_end;

__mram_ptr void *mram_alloc(const size_t n)
{
  __mram_ptr void* p = mram_free;
  mram_free += (n + 7) & ~7;
  return p;
}

typedef struct _list_element {
  int key;
  int value;
  __mram_ptr struct _list_element* next;
} list_element_t;

__mram_ptr list_element_t* head;

size_t round_up_dma(size_t n)
{
  return (n + 7) & ~7;
}

__mram_ptr list_element_t* make_element(kvpair_t kv,
				       __mram_ptr list_element_t* next)
{
  list_element_t tmp;
  __mram_ptr list_element_t* e =
    (__mram_ptr list_element_t*) mram_alloc(sizeof(list_element_t));
  tmp.key = kv.key;
  tmp.value = kv.value;
  tmp.next = next;
  mram_write(&tmp, e, round_up_dma(sizeof(tmp)));
  return e;
}  
  
void insert(kvpair_t kv)
{
  __mram_ptr list_element_t* prev = NULL;
  __mram_ptr list_element_t* p = head;

  while (p != NULL && kv.key > p->key) {
    prev = p;
    p = p->next;
  }
  if (p != NULL && kv.key == p->key)
    p->value = kv.value;
  else {
    p = make_element(kv, p);
    if (prev == NULL)
      head = p;
    else
      prev->next = p;
  }
}

void print_list()
{
  __mram_ptr list_element_t* p;
  int i;
  for (p = head, i = 0; p != NULL && i < 10; p = p->next, i++)
    printf("%d: %d -> %d\n", i, p->key, p->value);
}

int main()
{
  if (me() == 0) {
    int i;
    for (i = 0; i < nr_queries; i++)
      insert(query_buffer[i]);

    print_list();
  }
  return 0;
}
