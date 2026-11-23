const GITHUB_REPOSITORY = "https://github.com/liulang5945-netizen/Seed";

const kernelSteps = [
  {
    method: "ByteSensor.encode(symbol)",
    stage: "STEP 01 / INPUT ENCODING",
    title: "先把字节变成明确的输入符号",
    description: "文本示例不经过 tokenizer。0–255 直接表示原始字节，256 表示数据流边界；编码器把当前符号写成 one-hot 感受器活动，起止边界可以分别控制。",
    code: "# 0..255 = raw bytes; 256 = stream boundary\nvalue = torch.zeros(alphabet_size)\nvalue[int(symbol)] = 1.0",
    facts: ["257 个输入符号", "无 tokenizer", "边界符可独立控制"],
    source: `${GITHUB_REPOSITORY}/blob/main/taiji/organs.py#L17-L64`,
  },
  {
    method: "Taiji.observe(symbol, learn=True)",
    stage: "STEP 02 / SETTLE THE PAST",
    title: "新观察先给上一次预测打分",
    description: "状态里保存着上一步的概率分布。真实符号到达后，系统读取它此前的概率并计算 surprise；如果还有待结算的行动经历，也会把行动、奖励和真实结果写入情景场。",
    code: "prior_probability = previous.motor_probabilities[symbol]\nsurprise = -log(max(prior_probability, 1e-12))\nprior_prediction = previous.motor_probabilities.argmax()",
    facts: ["先结算，再更新", "surprise = −log p", "行动结果进入经历"],
    source: `${GITHUB_REPOSITORY}/blob/main/taiji/model.py#L2261-L2267`,
  },
  {
    method: "TaijiFabric.step(sensor_activity, previous_regions)",
    stage: "STEP 03 / CAUSAL STATE UPDATE",
    title: "稀疏神经图推进内部状态",
    description: "皮层不是一张稠密矩阵。固定扇入连接做快速预测、慢速巩固和递归下一状态预测，再把误差沿连接反向投递；膜电位、抑制、阈值和活动痕迹一起更新。",
    code: "fast + slow sparse prediction\n→ reciprocal error backprojection\n→ recurrent next-state prediction\n→ membrane / inhibition / threshold / trace update",
    facts: ["固定扇入稀疏连接", "快状态与慢巩固", "误差沿原连接回投"],
    source: `${GITHUB_REPOSITORY}/blob/main/TAIJI_CODE_WIKI.md#L12-L18`,
  },
  {
    method: "EpisodicField.recall(cortical_state)",
    stage: "STEP 04 / OPTIONAL MEMORY",
    title: "需要时，取回与当前状态相似的经历",
    description: "情景场用分布式线索和递归模式补全取回经历，再把反馈延迟一个因果时步送回皮层。注意：learn_bytes 默认 use_memory=False，避免记忆任务污染基础字节预测信号。",
    code: "recall = EpisodicField.recall(cortical_state)\n# action / outcome / value / time / source readouts\nnext_tick_feedback = recall.cortical_projection",
    facts: ["按线索模式补全", "反馈延迟一个时步", "字节学习默认关闭"],
    source: `${GITHUB_REPOSITORY}/blob/main/TAIJI_CODE_WIKI.md#L20-L26`,
  },
  {
    method: "ByteMotor.probabilities(context + memory_evidence)",
    stage: "STEP 05 / READOUT",
    title: "把内部状态读成下一步概率",
    description: "行动路径由 ByteMotor 合成概率；字节学习和生成则使用独立的 predictive readout。两条读出路径分开，下一字节误差不会被误当成行动奖励。",
    code: "action path:     ByteMotor.probabilities(context)\npredictive path: predictive_readout(context)\nchoice:          argmax 或 multinomial sample",
    facts: ["行动与预测读出分离", "贪心或采样", "生成结果会反馈回状态"],
    source: `${GITHUB_REPOSITORY}/blob/main/taiji/model.py#L3594-L3657`,
  },
  {
    method: "TaijiState(...)",
    stage: "STEP 06 / ATOMIC STATE",
    title: "把这一步完整写回持续状态",
    description: "区域活动、记忆状态、运动上下文、概率、逻辑时间以及待处理行动或经历共同进入新状态。checkpoint 不只保存权重，还保存拓扑、快速状态、事务状态和随机数状态。",
    code: "TaijiState(\n  regions, memory, motor_context,\n  motor_probabilities, tick,\n  pending_action, pending_experience\n)",
    facts: ["状态原子替换", "保存拓扑与参数", "恢复随机数与事务状态"],
    source: `${GITHUB_REPOSITORY}/blob/main/TAIJI_CODE_WIKI.md#L60-L62`,
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
