const GITHUB_REPOSITORY = "https://github.com/liulang5945-netizen/Seed";

const kernelSteps = [
  {
    method: "BytePatchEncoder.forward(byte_ids)",
    stage: "STEP 01 / PURE UTF-8 BYTE & MACRO VOCAB",
    title: "纯 UTF-8 字节基元，以及可中性生长的宏词表",
    description: "全面移除 BPE 与固定词表限制：0–255 直入原始 UTF-8 字节，256–258 为流边界与填充符；3 层严格因果卷积（dilation=1,2,4）构成本地字节特征，并支持在学习需要更粗粒度组合时通过 DynamicByteVocab 无损中性挂接新宏词元。",
    code: "# BYTE_VOCAB = 259 (0..255 raw bytes, EOS/BOS/PAD)\n# 3-layer causal Conv1d (k=3, d=1/2/4) -> 8-byte patch\na_byte, e_patch = self.encoder(byte_ids)\n# Lossless dynamic macro birth (0.0 perturbation):\nmodel.grow_macro_symbol_neutral_(b'\\xe5\\x8e\\x9f\\xe7\\x94\\x9f')",
    facts: ["259 纯字节基元", "0% OOV", "零扰动宏词元扩展"],
    source: `${GITHUB_REPOSITORY}/blob/main/taiji/native_cognitive_backbone.py`,
  },
  {
    method: "StatefulNeuralGroup.step_tick_cached(...)",
    stage: "STEP 02 / HETEROGENEOUS PERSISTENT GROUPS",
    title: "四组异质时间尺度神经群与跨窗口持久内态",
    description: "放弃 Transformer 堆叠注意力：4 个不同时间常数（τ_init ∈ {1, 4, 16, 64}）的有状态神经群同步通过 Q/K/V 消息传递，推动个体单元电位 z、自适应阈值 a、循环连接 W_rec 与长程慢隐态 H 跨窗口连续演化。",
    code: "v = F.linear(x_dyn, r_dyn) + e_r + F.linear(z_alive, w_eff) - b * a_prev\nz_next = (c * z_prev + (1.0 - c) * torch.tanh(v)) * alive\na_next = ((1.0 - eta) * a_prev + eta * z_next) * alive\nh_next = (1.0 - ell) * h_prev + ell * torch.tanh(pre_act)",
    facts: ["零 Transformer 架构", "τ ∈ {1, 4, 16, 64}", "H / z / a / F 跨窗持久化"],
    source: `${GITHUB_REPOSITORY}/blob/main/taiji/native_cognitive_backbone.py`,
  },
  {
    method: "u2_operators.fp1(f_prev, k, v, alpha, gains, chi, radius)",
    stage: "STEP 03 / FP1 KKT FAST MEMORY",
    title: "带 Frobenius 范数约束的精确 KKT 在线快记忆",
    description: "每个神经群持有在线可写可冻结的快记忆矩阵 F，通过精确 KKT 球约束二次求解器与 VJP 端到端求导写入事实记忆；允许权限置零（allowed=False）时保持逐位冻结，并显式支持 no-F 强基线消融开关。",
    code: "# min_F 0.5*||F - F_prev||^2 + 0.5*sum_i g_i ||F^T k_i - v_i||^2\n# s.t. ||F||_F <= radius  (exact KKT + active VJP)\nf_next = fp1(f_b, k_seq, v_seq, alpha, gains, chi, radius, allowed=True)",
    facts: ["精确 KKT 约束求解", "allowed=False 逐位冻结", "no-F 基线可对照"],
    source: `${GITHUB_REPOSITORY}/blob/main/taiji/u2_operators.py`,
  },
  {
    method: "grow_stateful_units_svd_neutral_ / recycle_retired_",
    stage: "STEP 04 / NEUTRAL BIRTH & 5-STAGE LIFECYCLE",
    title: "零扰动 SVD 中性出生与五态单元生命周期",
    description: "当能力遇到瓶颈压力时，仅在影子沙盒中沿误差-输入信用矩阵左奇异方向出生新单元（r_new=0 保证出生瞬时输出零扰动，p_new=c·U_k 确保首步即有启动梯度），并配合五态生命周期与 AdamW 动量迁移保护老知识。",
    code: "shadow.mark_all_mature_()            # 先冻结老单元\nshadow.grow_stateful_units_svd_neutral_(budget=32)\n# 适应期仅新生 immature 单元可塑：\nshadow.apply_selective_plasticity_grad_mask(freeze_mature_slow=True)",
    facts: ["出生微扰 Δy = 0", "5 态单元生命周期", "retired ID 永久不复用"],
    source: `${GITHUB_REPOSITORY}/blob/main/taiji/native_growth_controller.py`,
  },
  {
    method: "NativeExperienceCopyHead.step_emission(...)",
    stage: "STEP 05 / U2 §7 COPY HEAD & LEGAL UTF-8",
    title: "位置敏感原始经验复制头与合法 UTF-8 解码",
    description: "结合对数域位置游标（u2_operators.emission / accept / revoke_records）与逐字节 UTF-8 状态机掩码（utf8_mask），既能在有来源记录时实现精确可撤销拷贝，也保证任何自由采样永不截断非法 UTF-8 多字节序列。",
    code: "em = self.copy_head.step_emission(ctx, gen_logits, cursor, rec_bytes, valid, mask)\nsym = int(torch.multinomial(em['log_output'].exp(), 1).item())\ncursor = self.copy_head.step_accept(cursor, em, sym, rec_bytes, valid)",
    facts: ["U2 对数域游标复制", "支持单条经历撤销", "100% 合法 UTF-8 输出"],
    source: `${GITHUB_REPOSITORY}/blob/main/taiji/native_cognitive_backbone.py`,
  },
  {
    method: "TaijiNativeRuntime + OperationalStateMachine",
    stage: "STEP 06 / GOVERNED RUNTIME & 8-STATE OPS FSM",
    title: "九态外部动作台账、五平面学习分流与八态运营状态机",
    description: "统一运行时 TaijiNativeRuntime 将原生底座、条件价值头（ConditionalValueHead）、9 态防重入外部动作台账（ActionLifecycleLedger）与 5 平面结果分流（OutcomeRouter）合为一体，屏蔽自生成/虚构反馈，并支持 §15 八态日常影子升级。",
    code: "res = runtime.propose_and_execute_action(aid, idempotency_key=k, executor_fn=fn)\n# 5 平面隔离：state / episodic / fast_f / medium_psi / slow_theta\n# 仅真授权 real 结果且满足 retention_drop <= 0.02 方允许入账信用",
    facts: ["9 态防假回滚台账", "5 平面可信度门控", "8 态运营 FSM（N37）"],
    source: `${GITHUB_REPOSITORY}/blob/main/taiji/native_runtime.py`,
  },
];

const kernelButtons = Array.from(document.querySelectorAll(".kernel-step"));
const kernelMethod = document.getElementById("kernel-method");
const kernelStage = document.getElementById("kernel-stage");
const kernelTitle = document.getElementById("kernel-title");
const kernelDescription = document.getElementById("kernel-description");
const kernelCode = document.getElementById("kernel-code");
const kernelFacts = document.getElementById("kernel-facts");
const kernelSource = document.getElementById("kernel-source");

function activateKernelStep(index) {
  const step = kernelSteps[index];
  kernelButtons.forEach((button, buttonIndex) => {
    const selected = buttonIndex === index;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  kernelMethod.textContent = step.method;
  kernelStage.textContent = step.stage;
  kernelTitle.textContent = step.title;
  kernelDescription.textContent = step.description;
  kernelCode.textContent = step.code;
  kernelFacts.replaceChildren(...step.facts.map((fact) => {
    const item = document.createElement("li");
    item.textContent = fact;
    return item;
  }));
  kernelSource.href = step.source;
  document.getElementById("kernel-detail").setAttribute("aria-labelledby", `kernel-step-${index}`);
}

kernelButtons.forEach((button, index) => {
  button.addEventListener("click", () => activateKernelStep(index));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp", "ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    let nextIndex = index;
    if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = kernelButtons.length - 1;
    else if (["ArrowDown", "ArrowRight"].includes(event.key)) nextIndex = (index + 1) % kernelButtons.length;
    else nextIndex = (index - 1 + kernelButtons.length) % kernelButtons.length;
    activateKernelStep(nextIndex);
    kernelButtons[nextIndex].focus();
  });
});

if (kernelButtons.length === kernelSteps.length && kernelButtons.length > 0) {
  activateKernelStep(0);
}
