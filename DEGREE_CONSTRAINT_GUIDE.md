# B+树度约束检查指南

## 概述

度约束（Degree Constraint）是B+树的关键特性，用于保证树的平衡性。本指南说明如何验证所有节点是否满足度限制。

## B+树度参数

在本项目中，B+树的阶数 **m = 31**，因此：

| 参数 | 值 | 说明 |
|------|-----|------|
| **MAX_KEYS** | 30 | 每个节点最多容纳的键数 (m - 1) |
| **MIN_KEYS** | 15 | 非根内部节点最少容纳的键数 (⌈m/2⌉ - 1) |
| **根节点** | 1-30 | 根可以有1到MAX_KEYS个键 |
| **内部节点** | 15-30 | 非根内部节点必须有MIN_KEYS到MAX_KEYS个键 |
| **叶节点** | 0-30 | 叶节点可以有0到MAX_KEYS个键 |

## 验证方法

### 方法1：快速检查（已实现）

**位置**：`verify.cpp` 中的 `VERIFY-3` 检查

**功能**：
- 检查根节点大小是否在 [1, MAX_KEYS] 范围内
- DPU端执行快速root检查，避免WRAM压力

**运行**：
```bash
./verify B+Tree.dpu
```

**输出示例**：
```
[VERIFY-3] Degree constraint check:
[VERIFY-3]   MIN_KEYS=15, MAX_KEYS=30 (for B-tree of order 31)
[VERIFY-3]   Root node size=12 (valid range: 1-30)
[VERIFY-3] ✓ Root node satisfies constraints
```

### 方法2：完整递归检查（需DPU端实现）

**原理**：
递归遍历整棵树，检查每个节点的大小

**伪代码**：
```c
void check_all_nodes_degress(node, height)
  if height == 0:
    return  // 空树
  
  // 检查当前节点
  if node == root:
    assert 1 <= node.size <= MAX_KEYS
  else:
    assert MIN_KEYS <= node.size <= MAX_KEYS
  
  // 递归检查子节点
  if height > 1:
    for each child in node.children:
      check_all_nodes_degrees(child, height - 1)
```

**限制**：
- 递归深度过大会导致WRAM栈溢出（已在项目中观察到）
- 树高度为5，最多递归5层，存储局部变量消耗的栈空间

### 方法3：迭代堆栈方法（推荐用于完整检查）

**原理**：
使用MRAM中的栈代替递归，避免WRAM压力

**实现要点**：
```c
// 在MRAM中分配栈空间
typedef struct {
  void* node_ptr;
  int height;
  int parent_height;
} StackFrame;

StackFrame mram_stack[MAX_TREE_HEIGHT * MAX_NODES_PER_LEVEL];
int stack_ptr = 0;

// 迭代遍历而不是递归
while (stack_ptr > 0) {
  frame = mram_stack[--stack_ptr];
  // 检查节点...
  // 如果是内部节点，将子节点push到栈
}
```

## 当前实现状态

### ✓ 已实现
- [x] 快速根节点检查（DPU端）
- [x] 根节点大小验证（Host端）
- [x] 树结构连通性检查
- [x] 叶链排序检查

### ⚠️ 部分实现
- [x] 理论框架（见本文档）
- [ ] 完整递归遍历（因WRAM压力而暂时搁置）

### 📋 待实现
- [ ] 基于MRAM栈的迭代遍历
- [ ] 详细度违反报告
- [ ] 自动修复（如果发现违反）

## 问题排查

### 问题1：根节点大小无效

**症状**：
```
[VERIFY-3] ✗ Root node violates constraints
```

**可能原因**：
1. 树初始化失败
2. 根节点指针损坏
3. 树高度为0

**解决方案**：
1. 检查初始化阶段输出
2. 验证MRAM指针有效性
3. 查看树高度值

### 问题2：完整检查导致DPU崩溃

**症状**：
```
illegal WRAM write - address=0x7fff... length=... bytes
```

**原因**：
递归遍历导致WRAM栈溢出

**解决方案**：
- 使用迭代方法替代递归
- 将栈存储在MRAM而不是WRAM
- 限制每次检查的子树大小

## 性能指标

在模拟器上运行（500K + 2.5K键）：

| 检查项 | 时间 | 备注 |
|--------|------|------|
| 快速根检查 | <1ms | DPU端执行 |
| 叶链遍历 | ~10ms | 33K+叶节点 |
| 排序验证 | ~10ms | 检查升序 |
| 结构检查 | <1ms | 快速拓扑检查 |

## 参考代码

### DPU端（B+Tree.c）
```c
// 快速检查函数
static int check_degree_constraints_quick(
  __mram_ptr void* node_ptr, 
  int node_size, 
  int height) {
  if (node_ptr == NULL || height == 0) return 0;
  
  // 根可以有 1 到 MAX_KEYS 个键
  if (node_size < 1 || node_size > MAX_KEYS) {
    printf("[WARN] Root node invalid size\n");
    return 1;
  }
  return 0;
}
```

### Host端（verify.cpp）
```cpp
bool verify_degree_constraint(struct dpu_set_t dpu_set, 
                             TreeExportInfo &export_info) {
  printf("[VERIFY-3] Degree constraint check:\n");
  printf("[VERIFY-3]   MIN_KEYS=15, MAX_KEYS=30\n");
  printf("[VERIFY-3]   Root node size=%d\n", 
         export_info.global_root_size);
  
  if (export_info.global_root_size >= 1 && 
      export_info.global_root_size <= MAX_KEYS) {
    printf("[VERIFY-3] ✓ Root node satisfies constraints\n");
    return true;
  }
  return false;
}
```

## 建议下一步

1. **完整验证实现**：
   - 实现基于MRAM栈的迭代遍历
   - 添加详细的违反报告

2. **性能优化**：
   - 批量检查多个子树（并行化）
   - 缓存频繁访问的节点

3. **自动修复**：
   - 检测根节点大小问题后自动修复
   - 内部节点重平衡

4. **测试扩展**：
   - 在UPMEM硬件上验证
   - 测试不同大小的树（不同键数量）
   - 压力测试（度接近限制的情况）

## 相关文件

- 实现：`src/dpu/B+Tree.c`（DPU端）、`src/verify.cpp`（Host端）
- 报告：`VERIFICATION_REPORT.md`
- 测试脚本：`src/test_load_balance.sh`

---

**最后更新**：2026-03-29
**维护者**：B+树验证团队
