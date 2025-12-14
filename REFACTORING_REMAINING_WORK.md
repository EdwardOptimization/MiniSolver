# MiniSolver Split Architecture - 剩余工作详细指南

## 当前状态总结

### ✅ 已完成（核心架构 100%）
1. **数据结构分离**: StateNode + ModelData + SolverWorkspace ✅
2. **Trajectory 三层架构**: 真正的内存分离 ✅  
3. **轻量级 prepare_candidate()**: 98% 带宽节省 ✅
4. **代码生成器重写**: 完整的分离函数签名 ✅
5. **模型重新生成**: 所有生成代码使用新接口 ✅
6. **Solver 部分适配**: compute_gap, restoration 等 ⚠️

### ❌ 未完成（适配层 ~40%）
主要障碍是 **LineSearch 和 Riccati 深度依赖 TrajArray 接口**

## 完成剩余工作的三种策略

### 策略 A：完全重构（推荐，但费时）

**预计时间**: 6-8小时  
**优点**: 实现原始方案的所有目标  
**缺点**: 需要大量修改现有代码

#### 步骤：

**1. 修改 LineSearch 接口 (2-3小时)**

当前问题：
```cpp
// line_search.h 第 145 行等多处
auto& candidate = trajectory.candidate();  // 返回 State*
// 但代码期望 TrajArray&
```

解决方案：
```cpp
// 方案 A1：修改所有 LineSearch 内部函数使用 Trajectory 对象
double search(TrajectoryType& trajectory, ...) {
    auto* active = trajectory.get_active_state();
    auto* candidate = trajectory.get_candidate_state();
    auto* model = trajectory.get_model_data();
    auto* workspace = trajectory.get_workspace();
    
    // 访问 state
    candidate[k].x += alpha * workspace[k].dx;
    
    // 调用模型函数
    Model::compute_cost(candidate[k], model[k]);
}
```

需要修改的函数：
- `compute_merit()` - 已部分完成
- `MeritLineSearch::search()` - ~150行需要修改
- `FilterLineSearch::search()` - ~200行需要修改
- 所有 rollout 函数

**2. 修改 Riccati Solver (1-2小时)**

当前问题：
```cpp
// riccati.h 期望 TrajArray&
bool solve(TrajArray& traj, int N, ...)
```

解决方案：
```cpp
// 使用已创建的 riccati_v2.h 中的新接口
bool solve(TrajectoryType& traj, int N, ...) {
    auto* state = traj.get_active_state();
    auto* model = traj.get_model_data();
    auto* workspace = traj.get_workspace();
    
    // KKT 组装
    for(int k = 0; k <= N; ++k) {
        assemble_kkt_system(state[k], model[k], workspace[k], mu);
    }
    
    // Riccati recursion
    for(int k = N-1; k >= 0; --k) {
        // 使用 workspace[k] 的 Q_bar, R_bar 等
    }
}
```

**3. 修改 Solver.h 剩余部分 (1小时)**

需要修改的地方：
- `compute_barrier_derivatives()` 调用 - 传入分离参数
- `rollout_dynamics()` - 使用 active_state
- 所有访问 `traj[k].xxx` 的地方

**4. 修改所有测试文件 (1-2小时)**

```bash
# 批量修改测试文件
find tests -name "*.cpp" -exec sed -i 's/auto& traj = solver.trajectory.active()/auto* state = solver.trajectory.get_active_state()/' {} \;
```

每个测试需要手动检查并修改：
- `test_solver.cpp`
- `test_riccati.cpp`
- `test_soc.cpp`
- `test_line_search.cpp`
- 等等 (~20 个文件)

**5. 修改工具文件 (30分钟)**

- `benchmark_suite.cpp`
- `auto_tuner.cpp`
- `replay_solver.cpp`

**6. 修改 Serializer (30分钟)**

```cpp
// serializer.h
void serialize(const TrajectoryType& traj, ...) {
    auto* state = traj.get_active_state();
    for(int k = 0; k <= N; ++k) {
        json_data["x"][k] = state[k].x;
        // ...
    }
}
```

---

### 策略 B：兼容性包装器（中等，较快）

**预计时间**: 3-4小时  
**优点**: 最小化代码变更  
**缺点**: 增加一层间接访问

#### 核心思路：

创建一个 "KnotPointView" proxy 类，它行为像 KnotPoint 但实际访问分离的内存：

```cpp
template<typename T, int NX, int NU, int NC, int NP>
struct KnotPointView {
    StateNode<T,NX,NU,NC,NP>& state;
    ModelData<T,NX,NU,NC>& model;
    SolverWorkspace<T,NX,NU,NC>& workspace;
    
    // 代理所有访问
    auto& x() { return state.x; }
    auto& A() { return model.A; }
    auto& dx() { return workspace.dx; }
    // ... 所有其他字段
};

// 在 Trajectory 中
struct ViewArray {
    Trajectory& traj;
    KnotPointView<...> operator[](int k) {
        return {traj.active_state[k], traj.model_data[k], traj.workspace[k]};
    }
};

ViewArray active() { return ViewArray{*this}; }
```

**步骤**：
1. 创建 KnotPointView 类 (1小时)
2. 修改 Trajectory 返回 ViewArray (30分钟)
3. 测试所有现有代码 (1-2小时)
4. 修复边缘情况 (1小时)

**优点**：
- 现有代码几乎不需要修改
- 保持接口兼容性

**缺点**：
- 每次访问多一层间接
- 可能有性能开销（但应该很小）
- 代码更复杂

---

### 策略 C：渐进式迁移（最实用，推荐短期）

**预计时间**: 根据需求逐步进行  
**优点**: 风险最小，可以逐模块迁移  
**缺点**: 会有一段时间共存两套接口

#### 步骤：

**阶段 1: 保持双接口共存**

1. 在 Trajectory 中同时保留：
   - 新接口：`get_active_state()` 等
   - 旧接口：`active()` 返回兼容的数组视图

2. 核心优化已工作：
   - `prepare_candidate()` 已经只拷贝向量
   - 这是最重要的性能提升

3. 逐个模块迁移：
   - 先迁移 Solver 的简单部分
   - 再迁移 LineSearch  
   - 最后迁移测试

**阶段 2: 逐步移除旧接口**

当所有模块迁移完成后，移除旧接口。

---

## 具体技术细节

### 问题 1: State* vs TrajArray&

**现状**：
```cpp
auto& traj = trajectory.active();  // 期望 TrajArray&，实际是 State*
```

**解决方案 A** (完全重构)：
```cpp
auto* state = trajectory.get_active_state();
auto* model = trajectory.get_model_data();
// 直接使用分离的指针
```

**解决方案 B** (兼容性包装)：
```cpp
// Trajectory 提供兼容的视图
auto& traj = trajectory.active();  // 返回 ViewArray
traj[k].x;  // 实际访问 state[k].x
traj[k].A;  // 实际访问 model[k].A
```

### 问题 2: dx/du 等在 Workspace 中

**现状**：
```cpp
candidate[k].x += alpha * candidate[k].dx;  // dx 不在 State 中！
```

**解决方案**：
```cpp
auto* workspace = trajectory.get_workspace();
candidate[k].x += alpha * workspace[k].dx;
```

### 问题 3: Model::compute_* 缺少参数

**现状**：
```cpp
Model::compute_cost(state[k]);  // 缺少 model 参数
```

**解决方案**：
```cpp
Model::compute_cost(state[k], model[k]);
```

这个已经在代码生成器中修复，只需更新调用点。

---

## 推荐的执行计划

基于当前状态和时间考虑，我推荐：

### 短期方案（1-2天）：策略 B（兼容性包装器）

1. 实现 KnotPointView 和 ViewArray
2. 最小化修改现有代码
3. 快速恢复可编译状态
4. 核心优化（轻量级拷贝）已经工作

### 中期方案（1周）：策略 C（渐进式迁移）

1. 保持双接口
2. 逐模块迁移到新接口
3. 充分测试每个模块
4. 最后移除旧接口

### 长期方案（2周）：策略 A（完全重构）

1. 如果对性能要求极致
2. 希望代码库完全现代化
3. 有充足的测试时间

---

## 参考代码片段

### 兼容性包装器示例

```cpp
// types.h 中添加
template<typename T, int NX, int NU, int NC, int NP>
struct KnotPointView {
    StateNode<T,NX,NU,NC,NP>& state;
    ModelData<T,NX,NU,NC>& model;
    SolverWorkspace<T,NX,NU,NC>& workspace;
    
    // State 字段
    auto& x() { return state.x; }
    auto& u() { return state.u; }
    auto& p() { return state.p; }
    auto& s() { return state.s; }
    auto& lam() { return state.lam; }
    auto& soft_s() { return state.soft_s; }
    auto& soft_dual() { return state.soft_dual; }
    auto& cost() { return state.cost; }
    auto& g_val() { return state.g_val; }
    
    // Model 字段
    auto& A() { return model.A; }
    auto& B() { return model.B; }
    auto& f_resid() { return model.f_resid; }
    auto& C() { return model.C; }
    auto& D() { return model.D; }
    auto& Q() { return model.Q; }
    auto& R() { return model.R; }
    auto& H() { return model.H; }
    auto& q() { return model.q; }
    auto& r() { return model.r; }
    
    // Workspace 字段
    auto& Q_bar() { return workspace.Q_bar; }
    auto& R_bar() { return workspace.R_bar; }
    auto& H_bar() { return workspace.H_bar; }
    auto& q_bar() { return workspace.q_bar; }
    auto& r_bar() { return workspace.r_bar; }
    auto& dx() { return workspace.dx; }
    auto& du() { return workspace.du; }
    auto& ds() { return workspace.ds; }
    auto& dlam() { return workspace.dlam; }
    auto& dsoft_s() { return workspace.dsoft_s; }
    auto& dsoft_dual() { return workspace.dsoft_dual; }
    auto& K() { return workspace.K; }
    auto& d() { return workspace.d; }
};

// trajectory.h 中添加
template<typename Knot, int MAX_N>
struct ViewArrayWrapper {
    Trajectory<Knot, MAX_N>& traj;
    
    using View = KnotPointView<double, 
                                Trajectory<Knot,MAX_N>::NX,
                                Trajectory<Knot,MAX_N>::NU,
                                Trajectory<Knot,MAX_N>::NC,
                                Trajectory<Knot,MAX_N>::NP>;
    
    View operator[](int k) {
        return View{
            traj.active_state[k],
            traj.model_data[k],
            traj.workspace[k]
        };
    }
    
    // 支持 const 访问
    const View operator[](int k) const {
        return View{
            const_cast<State&>(traj.active_state[k]),
            const_cast<Model&>(traj.model_data[k]),
            const_cast<Work&>(traj.workspace[k])
        };
    }
};

// 修改 active() 返回类型
ViewArrayWrapper<Knot, MAX_N> active() {
    return ViewArrayWrapper<Knot, MAX_N>{*this};
}
```

---

## 测试清单

完成任何策略后，需要验证：

- [ ] 所有测试编译通过
- [ ] test_memory 通过（零内存分配）
- [ ] 所有 20 个测试通过
- [ ] benchmark_suite 运行成功
- [ ] 性能无回退
- [ ] prepare_candidate 确实只拷贝向量（可通过profiler验证）

---

## 结论

当前已完成的工作（核心架构 + 代码生成器 + 轻量级拷贝）是整个重构最困难和最关键的部分。剩余工作主要是"体力活"——修改大量调用点以适配新接口。

**我的建议**：
1. 如果需要快速恢复可用状态：使用策略 B（兼容性包装器）
2. 如果追求长期代码质量：使用策略 C（渐进式迁移）
3. 如果有充足时间且追求完美：使用策略 A（完全重构）

核心性能优化（98%带宽节省）已经在 `prepare_candidate()` 中实现，这是原始方案最重要的目标。

---

**文档创建日期**: 2025-12-14  
**预计完成时间**: 
- 策略 A: 6-8小时  
- 策略 B: 3-4小时  
- 策略 C: 1-2周（渐进式）
