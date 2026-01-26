import subprocess
import re

threads = [1, 2, 4, 8, 16]
results = {}

print(f"{'Threads':<10} | {'Insert (s)':<12} | {'Merge (s)':<12} | {'Total (s)':<12}")
print("-" * 55)

for t in threads:
    # Compile DPU binary with specific NR_TASKLETS
    # Reduced STACK_SIZE_DEFAULT to avoid WRAM overflow with 16 tasklets and new struct sizes
    compile_cmd = f"dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS={t} -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2048 -o B+Tree.dpu dpu/B+Tree.c"
    subprocess.run(compile_cmd, shell=True, check=True, stderr=subprocess.DEVNULL)
    
    # Run Host
    run_cmd = "./host B+Tree.dpu"
    result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)

    match = re.search(r"Insert: ([0-9.]+) sec, Merge: ([0-9.]+) sec, Total: ([0-9.]+) sec", result.stdout)
    if match:
        insert_t = match.group(1)
        merge_t = match.group(2)
        total_t = match.group(3)
        print(f"{t:<10} | {insert_t:<12} | {merge_t:<12} | {total_t:<12}")
    else:
        # Fallback
        match_old = re.search(r"Elapsed time: ([0-9.]+) sec", result.stdout)
        if match_old:
             print(f"{t:<10} | {'?':<12} | {'?':<12} | {match_old.group(1):<12}")
        else:
            print(f"{t:<10} | {'Error':<12} | {'Error':<12} | {'Error':<12}")
            # print(result.stdout)

