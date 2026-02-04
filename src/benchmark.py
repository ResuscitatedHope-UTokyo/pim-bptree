import subprocess
import re

try:
    commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
    print(f"Current Git Commit: {commit_id}")
except:
    print("Could not get git commit id")
# exit(0)
threads = [1, 2, 4, 8, 16]
results = {}

print(f"{'Threads':<10} | {'Insert (s)':<12} | {'ParaMerge (s)':<14} | {'SerialMerge (s)':<16} | {'Merge (s)':<12} | {'Total (s)':<12}")
print("-" * 95)

for t in threads:
    # Compile DPU binary with specific NR_TASKLETS
    # Reduced STACK_SIZE_DEFAULT to avoid WRAM overflow with 16 tasklets and new struct sizes
    compile_cmd = f"dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS={t} -Wl,--no-gc-sections -DSTACK_SIZE_DEFAULT=2048 -o B+Tree.dpu dpu/B+Tree.c"
    subprocess.run(compile_cmd, shell=True, check=True, stderr=subprocess.DEVNULL)
    
    # Run Host
    run_cmd = "./host B+Tree.dpu"
    result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)

    # New format: Insert: X sec, ParallelMerge: X sec, SerialMerge: X sec, Merge: X sec, Total: X sec
    match = re.search(r"Insert: ([0-9.]+) sec, ParallelMerge: ([0-9.]+) sec, SerialMerge: ([0-9.]+) sec, Merge: ([0-9.]+) sec, Total: ([0-9.]+) sec", result.stdout)
    if match:
        insert_t = match.group(1)
        para_merge_t = match.group(2)
        serial_merge_t = match.group(3)
        merge_t = match.group(4)
        total_t = match.group(5)
        print(f"{t:<10} | {insert_t:<12} | {para_merge_t:<14} | {serial_merge_t:<16} | {merge_t:<12} | {total_t:<12}")
    else:
        # Fallback to old format
        match_old = re.search(r"Insert: ([0-9.]+) sec, Merge: ([0-9.]+) sec, Total: ([0-9.]+) sec", result.stdout)
        if match_old:
            print(f"{t:<10} | {match_old.group(1):<12} | {'?':<14} | {'?':<16} | {match_old.group(2):<12} | {match_old.group(3):<12}")
        else:
            print(f"{t:<10} | {'Error':<12} | {'Error':<14} | {'Error':<16} | {'Error':<12} | {'Error':<12}")