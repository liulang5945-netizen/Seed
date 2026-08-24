import { useState, useMemo } from "react";

// ── Data ──────────────────────────────────────────────────────────────────────

type Finding = {
  id: string; sev: "critical"|"high"|"medium"|"low"; cat: string;
  title: string; desc: string; file: string; fix: string;
  effort: "low"|"medium"|"high"; hours: number; impact: "low"|"medium"|"high";
  quickWin: boolean;
};

const FINDINGS: Finding[] = [
  { id:"F01", sev:"high", cat:"CI/CD", title:"前端 ESLint 为 advisory 模式", desc:"ci.yml 中 eslint 设置了 continue-on-error: true，即使发现严重问题 CI 仍通过。前端 lint 不阻断 PR 合并。", file:".github/workflows/ci.yml", fix:"修复现有 ESLint 错误后移除 continue-on-error: true。", effort:"low", hours:3, impact:"high", quickWin:true },
  { id:"F02", sev:"high", cat:"Testing", title:"覆盖率阈值 fail_under=0", desc:"配置了 coverage 但 fail_under=0，覆盖率追踪形同虚设。任何覆盖率退化都不会导致 CI 失败。", file:"pyproject.toml", fix:"设置合理 fail_under（如当前覆盖率的 90%），或使用 codecov PR 评论。", effort:"low", hours:2, impact:"high", quickWin:true },
  { id:"F03", sev:"high", cat:"CI/CD", title:"mypy 已配置但未在 CI 运行", desc:"pyproject.toml 有完整 [tool.mypy] 配置，但 CI 中没有 mypy 步骤。类型错误只能运行时发现。", file:".github/workflows/ci.yml", fix:"添加 mypy step，先用 advisory 模式评估错误数量，修复后改为 blocking。", effort:"medium", hours:8, impact:"medium", quickWin:false },
  { id:"F04", sev:"medium", cat:"Testing", title:"15 个 Vue 组件 + 6 个 View 无组件测试", desc:"vitest 仅覆盖 stores/ 和 composables/，15 个 .vue 组件和 6 个 View 页面完全无渲染/交互测试。@vue/test-utils 已安装未使用。", file:"frontend/src/__tests__/", fix:"为核心组件添加 @vue/test-utils 挂载测试：ChatView、AppSidebar、RuntimeExceptionCenter。", effort:"high", hours:20, impact:"medium", quickWin:false },
  { id:"F05", sev:"medium", cat:"Build", title:"CI 无桌面构建验证", desc:"CI 有 build-frontend job 但不验证 PyInstaller 打包或 NSIS。scripts/release.py 已自动化但 CI 不运行。桌面端代码破坏只能靠本地发现。", file:".github/workflows/ci.yml", fix:"在 test-windows 中添加 PyInstaller --verify 或添加 build-desktop job。", effort:"medium", hours:6, impact:"medium", quickWin:false },
  { id:"F06", sev:"medium", cat:"Testing", title:"Playwright 仍仅截图，无 E2E 断言", desc:"shoot-fe.cjs 使用 Playwright 截图但无 expect() 断言。端到端流程（启动→登录→发消息→验证回复）完全无自动化验证。", file:"frontend/shoot-fe.cjs", fix:"添加 expect(page.locator(...)).toBeVisible() 等断言，验证关键 UI 元素。", effort:"medium", hours:8, impact:"medium", quickWin:false },
  { id:"F07", sev:"medium", cat:"CI/CD", title:"Windows CI 无 pip 缓存且无覆盖率", desc:"test-windows 没有 cache: pip 配置，每次重新安装依赖。也不生成覆盖率报告或 JUnit XML，无法对比跨平台差异。", file:".github/workflows/ci.yml", fix:"添加 cache: pip + --cov --junitxml 参数 + upload-artifact。", effort:"low", hours:1, impact:"low", quickWin:true },
  { id:"F08", sev:"medium", cat:"Quality", title:"ruff 规则仅 E/F/W，缺少 import 排序等", desc:"select 仅 [\"E\",\"F\",\"W\"]，缺少 I（isort）、UP（pyupgrade）、ANN（类型注解）、B（bugbear）等有价值规则。", file:"pyproject.toml", fix:"逐步添加 I + UP 规则，评估后选择性启用 ANN 和 B。", effort:"medium", hours:4, impact:"low", quickWin:false },
  { id:"F09", sev:"medium", cat:"Testing", title:"OpenAPI 快照自动更新，可能掩盖破坏性变更", desc:"检测到 schema 变化时自动更新快照并 fail。但开发者直接 commit 更新后的快照，破坏性变更会被静默合并。", file:"tests/test_openapi_snapshot.py", fix:"添加 --snapshot-update 参数控制更新行为，默认严格对比。", effort:"low", hours:2, impact:"medium", quickWin:true },
  { id:"F10", sev:"medium", cat:"Security", title:"Dependabot 仅跟踪更新，无 CVE 扫描", desc:"dependabot.yml 配置了周更但 CI 无 pip-audit 步骤。Dependabot 不主动报告当前版本的已知 CVE。", file:".github/dependabot.yml", fix:"CI 添加 pip-audit 步骤扫描已知 CVE，高危时阻断 CI。", effort:"low", hours:2, impact:"medium", quickWin:true },
  { id:"F11", sev:"medium", cat:"Testing", title:"verify 脚本输出格式不统一", desc:"8+ 个 verify 脚本输出格式各异，CI 仅通过退出码判断。无法统一收集并绘制指标趋势图。", file:"scripts/training/", fix:"定义统一 JSON 输出 schema（metrics/threshold/status），CI 收集归档。", effort:"medium", hours:8, impact:"low", quickWin:false },
  { id:"F12", sev:"low", cat:"Workflow", title:"无 Docker/容器化支持", desc:"有 devcontainer.json 但无 Dockerfile/docker-compose.yml。生产部署依赖目标机器环境，无法保证环境一致性。", file:".devcontainer/", fix:"添加 Dockerfile（多阶段构建）和 docker-compose.yml，支持容器化部署。", effort:"medium", hours:8, impact:"low", quickWin:false },
  { id:"F13", sev:"low", cat:"Quality", title:"无性能基准测试", desc:"无自动化性能测试（推理延迟、训练吞吐量、内存占用）。性能退化只能在用户反馈中发现。", file:"tests/", fix:"添加 pytest-benchmark 覆盖关键路径，CI 对比基准超阈值警告。", effort:"medium", hours:12, impact:"low", quickWin:false },
  { id:"F14", sev:"low", cat:"Quality", title:"ESLint import/no-unresolved 被禁用", desc:".eslintrc.cjs 禁用了 import 路径解析检查，错误的 import 路径不会被 ESLint 捕获。", file:"frontend/.eslintrc.cjs", fix:"配置 eslint-import-resolver-alias 映射 @ alias，启用 import/no-unresolved。", effort:"low", hours:2, impact:"low", quickWin:false },
];

const SEV: Record<string, { bg:string; ring:string; text:string; label:string }> = {
  critical: { bg:"#fef2f2", ring:"#fca5a5", text:"#dc2626", label:"Critical" },
  high:     { bg:"#fff7ed", ring:"#fdba74", text:"#ea580c", label:"High" },
  medium:   { bg:"#fefce8", ring:"#fde047", text:"#ca8a04", label:"Medium" },
  low:      { bg:"#f0fdf4", ring:"#86efac", text:"#16a34a", label:"Low" },
};

const IMPACT_MAP: Record<string,number> = { low:1, medium:2, high:3 };

const CAT_SCORES: { cat:string; score:string; color:string }[] = [
  { cat:"Dev Workflow", score:"B", color:"#22c55e" },
  { cat:"Build", score:"B", color:"#22c55e" },
  { cat:"Security", score:"B-", color:"#eab308" },
  { cat:"Quality", score:"B-", color:"#eab308" },
  { cat:"CI/CD", score:"B-", color:"#f97316" },
  { cat:"Testing", score:"C-", color:"#ef4444" },
];

const STRENGTHS = [
  "ruff + black 双重 blocking 门禁 + pre-commit hooks 本地校验，CI 双重守护",
  "vitest 覆盖 3 stores + 4 composables 共 51 个测试用例，前端逻辑测试从 0 到 1",
  "版本号以 pyproject.toml 为单一事实来源，sync_version.py 自动同步 5 文件 + CI --check",
  "scripts/release.py 一键编排前端→PyInstaller→NSIS→产物校验，构建流程标准化",
  "OpenAPI 快照测试 + 安全中间件状态暴露 + Dependabot 三生态周更，工程化基线扎实",
];

const PHASES = [
  { name:"快速收益", time:"1-2 天", color:"#22c55e", items:[
    { text:"ESLint 改为 blocking", ref:"F01" },
    { text:"设置覆盖率阈值", ref:"F02" },
    { text:"OpenAPI 快照严格模式", ref:"F09" },
    { text:"Windows CI 缓存+覆盖率", ref:"F07" },
    { text:"添加 pip-audit", ref:"F10" },
  ]},
  { name:"测试深度", time:"1 周", color:"#3b82f6", items:[
    { text:"Vue 组件测试", ref:"F04" },
    { text:"Playwright E2E 断言", ref:"F06" },
    { text:"verify 输出统一", ref:"F11" },
  ]},
  { name:"CI 完整性", time:"1 周", color:"#a855f7", items:[
    { text:"mypy 入 CI", ref:"F03" },
    { text:"桌面构建验证", ref:"F05" },
    { text:"ruff 规则扩展", ref:"F08" },
  ]},
  { name:"工程化", time:"持续", color:"#f97316", items:[
    { text:"Docker 容器化", ref:"F12" },
    { text:"性能基准测试", ref:"F13" },
    { text:"ESLint import 解析", ref:"F14" },
  ]},
];

// R1 → R2 improvement delta
const DELTA = {
  before: { score:"C+", findings:18, critical:2, high:5, gaps:["无 lint 门禁","前端零测试","无版本管理","无 pre-commit","无 devcontainer"] },
  after:  { score:"B-", findings:14, critical:0, high:3, gaps:["ESLint advisory","覆盖率阈值=0","mypy 未入 CI","无组件测试","无桌面构建验证"] },
};

// ── Helpers ───────────────────────────────────────────────────────────────────

const s = (v: React.CSSProperties) => v;

function ScoreGauge({ score }: { score: string }) {
  const pctMap: Record<string,number> = { "A+":95, A:90, "B+":85, B:80, "B-":77, "C+":75, C:70, "D+":65, D:60, F:40 };
  const pct = pctMap[score] ?? 50;
  const r = 42, cx = 54, cy = 54;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ * 0.75;
  const gap = circ - dash;
  const hue = Math.max(0, Math.min(120, (pct - 50) * 2.4));
  return (
    <svg width={108} height={108} viewBox="0 0 108 108">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth={8}
        strokeDasharray={`${circ * 0.75} ${circ * 0.25}`} strokeLinecap="round"
        transform={`rotate(135 ${cx} ${cy})`} />
      <circle cx={cx} cy={cy} r={r} fill="none" stroke={`hsl(${hue},80%,60%)`} strokeWidth={8}
        strokeDasharray={`${dash} ${gap}`} strokeLinecap="round"
        transform={`rotate(135 ${cx} ${cy})`}
        style={{ transition: "stroke-dasharray 0.8s ease, stroke 0.4s" }} />
      <text x={cx} y={cy - 2} textAnchor="middle" fill="#fff" fontSize={26} fontWeight={900}
        fontFamily="-apple-system,sans-serif">{score}</text>
      <text x={cx} y={cy + 16} textAnchor="middle" fill="#94a3b8" fontSize={9} fontWeight={600}
        letterSpacing={1.2} fontFamily="-apple-system,sans-serif">OVERALL</text>
    </svg>
  );
}

function DonutChart() {
  const data = [
    { label:"Critical", value:0, color:"#ef4444" },
    { label:"High", value:3, color:"#f97316" },
    { label:"Medium", value:8, color:"#eab308" },
    { label:"Low", value:3, color:"#22c55e" },
  ];
  const total = data.reduce((a, d) => a + d.value, 0);
  const r = 40, cx = 56, cy = 56, sw = 14;
  const circ = 2 * Math.PI * r;
  const GAP = 3;
  let offset = 0;
  return (
    <div style={{ display:"flex", alignItems:"center", gap:16 }}>
      <svg width={112} height={112} viewBox="0 0 112 112">
        {total === 0 ? (
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="#e5e7eb" strokeWidth={sw} />
        ) : data.filter(d => d.value > 0).map(d => {
          const segLen = (d.value / total) * circ;
          const dash = Math.max(0, segLen - GAP);
          const el = (
            <circle key={d.label} cx={cx} cy={cy} r={r} fill="none" stroke={d.color}
              strokeWidth={sw} strokeDasharray={`${dash} ${circ - dash}`}
              strokeLinecap="round" transform={`rotate(${-90 + (offset / total) * 360} ${cx} ${cy})`}
              style={{ transition:"stroke-dasharray 0.5s" }} />
          );
          offset += d.value;
          return el;
        })}
        <text x={cx} y={cy - 4} textAnchor="middle" fill="#111827" fontSize={22} fontWeight={800}
          fontFamily="-apple-system,sans-serif">{total}</text>
        <text x={cx} y={cy + 12} textAnchor="middle" fill="#9ca3af" fontSize={9}
          fontFamily="-apple-system,sans-serif">findings</text>
      </svg>
      <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
        {data.map(d => (
          <div key={d.label} style={{ display:"flex", alignItems:"center", gap:8 }}>
            <div style={{ width:10, height:10, borderRadius:3, background:d.color, flexShrink:0 }} />
            <span style={{ fontSize:12, color:"#6b7280" }}>{d.label}</span>
            <span style={{ fontSize:12, fontWeight:700, color:"#111827", marginLeft:"auto" }}>{d.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function QuadrantChart({ onDotClick }: { onDotClick?: (id: string) => void }) {
  const W = 240, H = 190, PAD = 34;
  const xMax = 24;
  const toX = (h: number) => PAD + (h / xMax) * (W - PAD * 2);
  const toY = (imp: number) => H - PAD - ((imp - 0.5) / 3) * (H - PAD * 2);
  const posMap = new Map<string, number>();
  const dots = FINDINGS.map(f => {
    const key = `${f.hours}-${f.impact}`;
    const idx = posMap.get(key) ?? 0;
    posMap.set(key, idx + 1);
    const jx = idx === 0 ? 0 : (idx % 2 === 1 ? 7 : -7);
    const jy = idx < 2 ? 0 : (idx < 4 ? -6 : 6);
    return { f, x: toX(f.hours) + jx, y: toY(IMPACT_MAP[f.impact]) + jy };
  });
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ overflow:"visible" }}>
      <rect x={PAD} y={PAD} width={(W - PAD * 2) * 0.35} height={(H - PAD * 2) * 0.5}
        rx={4} fill="#22c55e" opacity={0.04} />
      <line x1={PAD} y1={H / 2} x2={W - PAD} y2={H / 2} stroke="#e5e7eb" strokeWidth={1} strokeDasharray="4 3" />
      <line x1={W * 0.38} y1={PAD} x2={W * 0.38} y2={H - PAD} stroke="#e5e7eb" strokeWidth={1} strokeDasharray="4 3" />
      <text x={PAD + 4} y={PAD + 10} fontSize={8} fill="#86efac" fontFamily="-apple-system,sans-serif" fontWeight={600}>Quick Wins</text>
      <text x={W - PAD - 4} y={H - PAD - 4} fontSize={7} fill="#e5e7eb" textAnchor="end" fontFamily="-apple-system,sans-serif">慎重考虑</text>
      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={9} fill="#9ca3af" fontFamily="-apple-system,sans-serif">投入 (hours) →</text>
      <text x={10} y={H / 2} textAnchor="middle" fontSize={9} fill="#9ca3af" fontFamily="-apple-system,sans-serif"
        transform={`rotate(-90 10 ${H/2})`}>收益 →</text>
      {dots.map(({ f, x, y }) => {
        const c = SEV[f.sev];
        return (
          <g key={f.id} style={{ cursor: onDotClick ? "pointer" : "default" }}
            onClick={() => onDotClick?.(f.id)}>
            {f.quickWin && <circle cx={x} cy={y} r={12} fill={c.text} opacity={0.08} />}
            <circle cx={x} cy={y} r={5.5} fill={c.text} opacity={0.82}
              stroke="#fff" strokeWidth={1.5} />
            <text x={x} y={y - 9} textAnchor="middle" fontSize={7} fontWeight={700} fill={c.text}
              fontFamily="-apple-system,sans-serif">{f.id}</text>
          </g>
        );
      })}
    </svg>
  );
}

function DeltaPanel() {
  return (
    <div style={s({ display:"grid", gridTemplateColumns:"1fr auto 1fr", gap:12, alignItems:"center" })}>
      {/* Before */}
      <div style={s({ background:"#fef2f2", borderRadius:10, padding:"12px 14px", border:"1px solid #fecaca" })}>
        <div style={s({ display:"flex", alignItems:"center", gap:8, marginBottom:8 })}>
          <span style={s({ fontSize:20, fontWeight:800, color:"#dc2626" })}>{DELTA.before.score}</span>
          <span style={s({ fontSize:10, color:"#9ca3af" }}>R1 · {DELTA.before.findings} findings</span>
        </div>
        <div style={s({ display:"flex", gap:6, marginBottom:6 })}>
          <span style={s({ fontSize:10, fontWeight:600, color:"#dc2626", background:"#fee2e2",
            borderRadius:4, padding:"1px 6px" })}>{DELTA.before.critical} critical</span>
          <span style={s({ fontSize:10, fontWeight:600, color:"#ea580c", background:"#ffedd5",
            borderRadius:4, padding:"1px 6px" })}>{DELTA.before.high} high</span>
        </div>
        <div style={s({ display:"flex", flexWrap:"wrap", gap:3 })}>
          {DELTA.before.gaps.map((g, i) => (
            <span key={i} style={s({ fontSize:9, color:"#991b1b", background:"#fff1f2",
              borderRadius:3, padding:"1px 5px", border:"1px solid #fecaca" })}>{g}</span>
          ))}
        </div>
      </div>
      {/* Arrow */}
      <div style={s({ display:"flex", flexDirection:"column", alignItems:"center", gap:2 })}>
        <span style={s({ fontSize:18, color:"#22c55e", fontWeight:700 })}>→</span>
        <span style={s({ fontSize:9, color:"#22c55e", fontWeight:600 })}>+2 grades</span>
      </div>
      {/* After */}
      <div style={s({ background:"#f0fdf4", borderRadius:10, padding:"12px 14px", border:"1px solid #bbf7d0" })}>
        <div style={s({ display:"flex", alignItems:"center", gap:8, marginBottom:8 })}>
          <span style={s({ fontSize:20, fontWeight:800, color:"#16a34a" })}>{DELTA.after.score}</span>
          <span style={s({ fontSize:10, color:"#9ca3af" })}>R2 · {DELTA.after.findings} findings</span>
        </div>
        <div style={s({ display:"flex", gap:6, marginBottom:6 })}>
          <span style={s({ fontSize:10, fontWeight:600, color:"#16a34a", background:"#dcfce7",
            borderRadius:4, padding:"1px 6px" })}>{DELTA.after.critical} critical</span>
          <span style={s({ fontSize:10, fontWeight:600, color:"#ea580c", background:"#ffedd5",
            borderRadius:4, padding:"1px 6px" })}>{DELTA.after.high} high</span>
        </div>
        <div style={s({ display:"flex", flexWrap:"wrap", gap:3 })}>
          {DELTA.after.gaps.map((g, i) => (
            <span key={i} style={s({ fontSize:9, color:"#166534", background:"#ecfdf5",
              borderRadius:3, padding:"1px 5px", border:"1px solid #bbf7d0" })}>{g}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Components ────────────────────────────────────────────────────────────────

function SevBadge({ sev }: { sev: string }) {
  const m = SEV[sev];
  return (
    <span style={s({
      display:"inline-flex", alignItems:"center", fontSize:10, fontWeight:700, letterSpacing:0.4,
      color:m.text, background:m.bg, border:`1px solid ${m.ring}`, borderRadius:5,
      padding:"1px 7px", lineHeight:"18px", whiteSpace:"nowrap",
    })}>{m.label}</span>
  );
}

function EffortImpact({ f }: { f: Finding }) {
  const effColor = f.effort === "low" ? "#22c55e" : f.effort === "medium" ? "#eab308" : "#ef4444";
  const impColor = f.impact === "low" ? "#94a3b8" : f.impact === "medium" ? "#3b82f6" : "#22c55e";
  return (
    <div style={s({ display:"flex", gap:8, alignItems:"center" })}>
      <span style={s({
        fontSize:10, fontWeight:600, color:effColor, background:`${effColor}14`,
        borderRadius:4, padding:"1px 6px", border:`1px solid ${effColor}30`,
      })}>投入 {f.effort} · {f.hours}h</span>
      <span style={s({
        fontSize:10, fontWeight:600, color:impColor, background:`${impColor}14`,
        borderRadius:4, padding:"1px 6px", border:`1px solid ${impColor}30`,
      })}>收益 {f.impact}</span>
      {f.quickWin && (
        <span style={s({
          fontSize:10, fontWeight:700, color:"#059669", background:"#ecfdf5",
          borderRadius:4, padding:"1px 6px", border:"1px solid #a7f3d0",
        })}>Quick Win</span>
      )}
    </div>
  );
}

function FindingCard({ f }: { f: Finding }) {
  const [open, setOpen] = useState(false);
  const m = SEV[f.sev];
  return (
    <div style={s({
      background:"#fff", borderRadius:10, border:`1px solid ${open ? m.ring : "#e5e7eb"}`,
      overflow:"hidden", transition:"border-color 0.2s, box-shadow 0.2s",
      boxShadow: open ? `0 2px 8px ${m.ring}40` : "0 1px 2px rgba(0,0,0,0.04)",
    })}>
      <div onClick={() => setOpen(!open)} style={s({
        display:"flex", alignItems:"center", gap:8, padding:"10px 14px",
        cursor:"pointer", userSelect:"none",
      })}>
        <SevBadge sev={f.sev} />
        <span style={s({ fontSize:10, color:"#9ca3af", fontWeight:600, minWidth:48, letterSpacing:0.3 })}>{f.cat}</span>
        <span style={s({ flex:1, fontSize:13, fontWeight:600, color:"#1f2937", lineHeight:1.3 })}>
          {f.id} — {f.title}
        </span>
        <span style={s({
          fontSize:14, color:"#9ca3af", transition:"transform 0.2s",
          transform: open ? "rotate(180deg)" : "none", flexShrink:0,
        })}>▾</span>
      </div>
      {open && (
        <div style={s({ padding:"0 14px 12px", borderTop:"1px solid #f3f4f6" })}>
          <p style={s({ fontSize:12.5, color:"#374151", lineHeight:1.65, margin:"8px 0 6px" })}>{f.desc}</p>
          <div style={s({ display:"flex", alignItems:"center", gap:8, marginBottom:6, flexWrap:"wrap" })}>
            <EffortImpact f={f} />
          </div>
          <div style={s({ fontSize:11, color:"#6b7280", marginBottom:4 })}>
            <span style={s({ fontWeight:600 })}>文件 </span>
            <code style={s({ background:"#f3f4f6", padding:"1px 6px", borderRadius:4, fontSize:10 })}>{f.file}</code>
          </div>
          <div style={s({
            fontSize:11.5, color:"#059669", fontWeight:500, lineHeight:1.5,
            background:"#f0fdf4", borderRadius:6, padding:"6px 10px", marginTop:4,
          })}>
            → {f.fix}
          </div>
        </div>
      )}
    </div>
  );
}

function Roadmap() {
  return (
    <div style={s({ position:"relative", paddingLeft:20 })}>
      <div style={s({
        position:"absolute", left:7, top:8, bottom:8, width:2,
        background:"linear-gradient(to bottom, #22c55e, #3b82f6, #a855f7, #f97316)",
        borderRadius:2, opacity:0.3,
      })} />
      {PHASES.map((p, i) => (
        <div key={i} style={s({ position:"relative", marginBottom: i < PHASES.length - 1 ? 16 : 0 })}>
          <div style={s({
            position:"absolute", left:-17, top:4, width:12, height:12, borderRadius:"50%",
            background:p.color, border:"2px solid #fff", boxShadow:`0 0 0 2px ${p.color}40`,
          })} />
          <div style={s({
            background:"#f8fafc", borderRadius:10, padding:"10px 14px",
            border:"1px solid #e2e8f0",
          })}>
            <div style={s({ display:"flex", alignItems:"baseline", gap:8, marginBottom:6 })}>
              <span style={s({ fontSize:13, fontWeight:700, color:"#1e293b" })}>Phase {i+1} — {p.name}</span>
              <span style={s({ fontSize:10, color:p.color, fontWeight:600,
                background:`${p.color}14`, padding:"1px 6px", borderRadius:4 })}>{p.time}</span>
            </div>
            <div style={s({ display:"flex", flexWrap:"wrap", gap:4 })}>
              {p.items.map((item, j) => (
                <span key={j} style={s({
                  fontSize:11, color:"#475569", background:"#fff", borderRadius:5,
                  padding:"2px 8px", border:"1px solid #e2e8f0", lineHeight:1.6,
                })}>{item.text} <span style={s({ color:"#9ca3af", fontSize:10 })}>{item.ref}</span></span>
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function SeedHarnessReportR2() {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    let list = FINDINGS;
    if (filter !== "all") list = list.filter(f => f.sev === filter);
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(f => f.title.toLowerCase().includes(q) || f.desc.toLowerCase().includes(q) || f.id.toLowerCase().includes(q) || f.cat.toLowerCase().includes(q));
    }
    return list;
  }, [filter, search]);

  const quickWins = FINDINGS.filter(f => f.quickWin);

  return (
    <div style={s({
      fontFamily:"-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif",
      background:"#f4f6fb", minHeight:"100vh", padding:"28px 16px", color:"#111827",
    })}>
      <div style={s({ maxWidth:920, margin:"0 auto" })}>

        {/* ── Hero ── */}
        <div style={s({
          background:"linear-gradient(135deg,#0f172a 0%,#1e293b 60%,#334155 100%)",
          borderRadius:16, padding:"28px 24px 22px", marginBottom:16, color:"#fff",
          position:"relative", overflow:"hidden",
        })}>
          <div style={s({
            position:"absolute", top:-40, right:-40, width:160, height:160, borderRadius:"50%",
            background:"radial-gradient(circle,rgba(34,197,94,0.15),transparent 70%)",
          })} />
          <div style={s({ display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:16, position:"relative" })}>
            <div style={s({ flex:1, minWidth:240 })}>
              <h1 style={s({ fontSize:22, fontWeight:800, margin:0, letterSpacing:-0.4, lineHeight:1.3 })}>
                Seed — Harness 实践分析 (R2)
              </h1>
              <p style={s({ fontSize:12, color:"#94a3b8", margin:"4px 0 0", lineHeight:1.5 })}>
                上一轮 18 项修复后复评 · CI/CD · 测试 · 构建 · 安全 · 开发工作流 — 14 项新发现
              </p>
              <p style={s({ fontSize:11, color:"#64748b", margin:"8px 0 0", lineHeight:1.5, maxWidth:420 })}>
                C+ → B-：lint 门禁、前端测试、版本统一、pre-commit、devcontainer 均已就位。短板在测试深度和 CI 完整性。
              </p>
            </div>
            <ScoreGauge score="B-" />
          </div>
          {/* stat row */}
          <div style={s({ display:"flex", gap:8, marginTop:18, flexWrap:"wrap" })}>
            {[
              { label:"Critical", val:0, c:"#22c55e" },
              { label:"High", val:3, c:"#f97316" },
              { label:"Medium", val:8, c:"#eab308" },
              { label:"Low", val:3, c:"#22c55e" },
            ].map(d => (
              <div key={d.label} style={s({
                flex:"1 1 80px", background:"rgba(255,255,255,0.06)", borderRadius:10,
                padding:"10px 8px", textAlign:"center", backdropFilter:"blur(4px)",
                border:"1px solid rgba(255,255,255,0.06)",
              })}>
                <div style={s({ fontSize:22, fontWeight:800, color:d.c, lineHeight:1 })}>{d.val}</div>
                <div style={s({ fontSize:10, color:"#94a3b8", marginTop:3 })}>{d.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Delta Panel ── */}
        <div style={s({
          background:"#fff", borderRadius:12, padding:"16px 18px", marginBottom:14,
          border:"1px solid #e5e7eb",
        })}>
          <h3 style={s({ fontSize:13, fontWeight:700, margin:"0 0 12px", color:"#374151" })}>
            R1 → R2 改进对比
          </h3>
          <DeltaPanel />
        </div>

        {/* ── Charts Row ── */}
        <div style={s({ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14, marginBottom:14 })}>
          <div style={s({ background:"#fff", borderRadius:12, padding:"16px 18px", border:"1px solid #e5e7eb" })}>
            <h3 style={s({ fontSize:13, fontWeight:700, margin:"0 0 10px", color:"#374151" })}>严重度分布</h3>
            <DonutChart />
          </div>
          <div style={s({ background:"#fff", borderRadius:12, padding:"16px 18px", border:"1px solid #e5e7eb" })}>
            <h3 style={s({ fontSize:13, fontWeight:700, margin:"0 0 10px", color:"#374151" })}>投入-收益矩阵</h3>
            <div style={s({ display:"flex", justifyContent:"center" })}>
              <QuadrantChart />
            </div>
          </div>
        </div>

        {/* ── Quick Wins ── */}
        <div style={s({
          background:"linear-gradient(135deg,#ecfdf5,#f0fdf4)", borderRadius:12, padding:"14px 18px",
          marginBottom:14, border:"1px solid #bbf7d0",
        })}>
          <h3 style={s({ fontSize:13, fontWeight:700, margin:"0 0 8px", color:"#166534" })}>
            Quick Wins — 低投入高收益，立即可做
          </h3>
          <div style={s({ display:"flex", flexWrap:"wrap", gap:6 })}>
            {quickWins.map(f => (
              <span key={f.id} style={s({
                fontSize:11, fontWeight:600, color:"#059669", background:"#fff",
                borderRadius:6, padding:"3px 10px", border:"1px solid #a7f3d0",
                cursor:"default", lineHeight:1.6,
              })}>{f.id} {f.title} <span style={s({ color:"#9ca3af", fontWeight:400 })}>· {f.hours}h</span></span>
            ))}
          </div>
        </div>

        {/* ── Strengths ── */}
        <div style={s({
          background:"#fff", borderRadius:12, padding:"14px 18px", marginBottom:14,
          border:"1px solid #e5e7eb",
        })}>
          <h3 style={s({ fontSize:13, fontWeight:700, margin:"0 0 8px", color:"#374151" })}>核心优势</h3>
          {STRENGTHS.map((t, i) => (
            <div key={i} style={s({ fontSize:12, color:"#374151", lineHeight:1.75, paddingLeft:16, position:"relative" })}>
              <span style={s({ position:"absolute", left:0, color:"#22c55e", fontWeight:700 })}>✓</span>{t}
            </div>
          ))}
        </div>

        {/* ── Category Scores ── */}
        <div style={s({
          background:"#fff", borderRadius:12, padding:"14px 18px", marginBottom:14,
          border:"1px solid #e5e7eb",
        })}>
          <h3 style={s({ fontSize:13, fontWeight:700, margin:"0 0 10px", color:"#374151" })}>分类评分</h3>
          <div style={s({ display:"grid", gridTemplateColumns:"repeat(3, 1fr)", gap:8 })}>
            {CAT_SCORES.map(cs => (
              <div key={cs.cat} style={s({
                display:"flex", alignItems:"center", justifyContent:"space-between",
                background:"#f8fafc", borderRadius:8, padding:"8px 12px",
                border:"1px solid #e2e8f0",
              })}>
                <span style={s({ fontSize:11, color:"#475569", fontWeight:500 })}>{cs.cat}</span>
                <span style={s({ fontSize:14, fontWeight:800, color:cs.color })}>{cs.score}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Findings ── */}
        <div style={s({
          background:"#fff", borderRadius:12, padding:"16px 18px", marginBottom:14,
          border:"1px solid #e5e7eb",
        })}>
          <div style={s({ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:12, flexWrap:"wrap", gap:8 })}>
            <h3 style={s({ fontSize:13, fontWeight:700, margin:0, color:"#374151" })}>
              全部发现 {filter !== "all" || search ? `(${filtered.length})` : ""}
            </h3>
            <div style={s({ display:"flex", gap:6, alignItems:"center", flexWrap:"wrap" })}>
              <input
                value={search} onChange={e => setSearch(e.target.value)}
                placeholder="搜索..."
                style={s({
                  fontSize:11, padding:"4px 10px", borderRadius:6, border:"1px solid #d1d5db",
                  outline:"none", width:120, color:"#374151", background:"#fafafa",
                })}
              />
              {["all","high","medium","low"].map(k => {
                const active = filter === k;
                const c = k === "all" ? "#3b82f6" : SEV[k]?.text ?? "#6b7280";
                return (
                  <button key={k} onClick={() => setFilter(k)} style={s({
                    fontSize:10, fontWeight:600, padding:"3px 9px", borderRadius:5, cursor:"pointer",
                    border: active ? `1.5px solid ${c}` : "1px solid #e5e7eb",
                    background: active ? `${c}10` : "#fff", color: active ? c : "#9ca3af",
                    transition:"all 0.15s",
                  })}>{k === "all" ? "全部" : SEV[k]?.label}</button>
                );
              })}
            </div>
          </div>
          <div style={s({ display:"flex", flexDirection:"column", gap:6 })}>
            {filtered.length === 0
              ? <div style={s({ fontSize:12, color:"#9ca3af", textAlign:"center", padding:20 })}>无匹配结果</div>
              : filtered.map(f => <FindingCard key={f.id} f={f} />)
            }
          </div>
        </div>

        {/* ── Roadmap ── */}
        <div style={s({
          background:"#fff", borderRadius:12, padding:"16px 18px", marginBottom:14,
          border:"1px solid #e5e7eb",
        })}>
          <h3 style={s({ fontSize:13, fontWeight:700, margin:"0 0 14px", color:"#374151" })}>建议改进路线</h3>
          <Roadmap />
        </div>

        {/* ── Footer ── */}
        <div style={s({ textAlign:"center", fontSize:10, color:"#b0b8c4", padding:"6px 0 0" })}>
          Seed Harness Analysis R2 · 2026-08-24 · /better-harness
        </div>
      </div>
    </div>
  );
}
