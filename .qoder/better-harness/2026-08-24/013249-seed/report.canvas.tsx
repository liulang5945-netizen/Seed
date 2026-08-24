import { useState, useMemo } from "react";

// ── Data ──────────────────────────────────────────────────────────────────────

type Finding = {
  id: string; sev: "critical"|"high"|"medium"|"low"; cat: string;
  title: string; desc: string; file: string; fix: string;
  effort: "low"|"medium"|"high"; hours: number; impact: "low"|"medium"|"high";
  quickWin: boolean;
};

const FINDINGS: Finding[] = [
  { id:"F01", sev:"critical", cat:"CI/CD", title:"CI 无 Lint/Format 门禁", desc:"CI 管线完全没有 ruff/black 检查步骤。dev 依赖声明了 black 和 ruff 但无 pyproject.toml 配置段，代码风格无自动守护。", file:".github/workflows/ci.yml", fix:"添加 ruff check + black --check 步骤，配置 [tool.ruff] 规则（从 E/F/W 起步）。", effort:"low", hours:2, impact:"high", quickWin:true },
  { id:"F02", sev:"critical", cat:"Testing", title:"前端零测试覆盖", desc:"frontend/ 无任何单元/组件/E2E 测试。Playwright 仅用于截图脚本 shoot-fe.cjs，无断言。Vue 3 + Pinia + Vue Router 完全没有 vitest 配置。", file:"frontend/package.json", fix:"引入 vitest + @vue/test-utils；将截图脚本改造为含断言的 E2E 冒烟测试。", effort:"high", hours:40, impact:"high", quickWin:false },
  { id:"F03", sev:"high", cat:"CI/CD", title:"CI 无构建/打包验证", desc:"CI 不验证 PyInstaller 打包、前端构建或 NSIS 安装程序。打包流程破坏只能靠人工发现。", file:".github/workflows/ci.yml", fix:"添加 Windows runner 执行 desktop/build.py 并验证产物；至少验证 vite build 通过。", effort:"medium", hours:8, impact:"high", quickWin:false },
  { id:"F04", sev:"high", cat:"Build", title:"版本号散落 4+ 文件，不一致", desc:"pyproject.toml 为 0.1.0，其余文件为 1.6.0。版本号无单一事实来源。", file:"pyproject.toml + 3", fix:"以 pyproject.toml 为唯一版本源，脚本自动同步到其他文件。", effort:"low", hours:4, impact:"medium", quickWin:true },
  { id:"F05", sev:"high", cat:"Testing", title:"无测试覆盖率追踪", desc:"pytest 无 --cov 参数，CI 不上传覆盖率报告，无法检测覆盖率退化。", file:"pyproject.toml", fix:"添加 pytest-cov，CI 中运行 pytest --cov 并上传报告。", effort:"low", hours:4, impact:"high", quickWin:true },
  { id:"F06", sev:"high", cat:"Quality", title:"无静态类型检查", desc:"api/ 大量使用 typing 注解但无 mypy/pyright 配置，类型错误只能运行时发现。", file:"api/app.py", fix:"添加 [tool.mypy] 配置（permissive 起步），CI 添加 mypy 检查。", effort:"medium", hours:16, impact:"medium", quickWin:false },
  { id:"F07", sev:"high", cat:"Security", title:"安全中间件可静默降级", desc:"import 失败时速率限制完全关闭，仅 WARNING 日志，/api/health 不暴露此状态。运维人员很难注意到。", file:"api/app.py", fix:"在 /api/health 添加 security_middleware 字段；生产模式下拒绝启动。", effort:"low", hours:2, impact:"high", quickWin:true },
  { id:"F08", sev:"high", cat:"CI/CD", title:"CI 仅 Ubuntu，Windows 路径无守护", desc:"桌面端为 Windows PyQt6 应用，CI 无 Windows runner，CREATE_NO_WINDOW / ctypes WinDLL 等代码无验证。", file:".github/workflows/ci.yml", fix:"添加 windows-latest 到 CI matrix，至少覆盖 pytest + import smoke test。", effort:"medium", hours:6, impact:"high", quickWin:false },
  { id:"F09", sev:"medium", cat:"CI/CD", title:"CI 无测试产物上传", desc:"CI 失败时无法下载完整 pytest 报告或日志，诊断困难。", file:".github/workflows/ci.yml", fix:"添加 --junitxml + actions/upload-artifact 上传测试结果和日志。", effort:"low", hours:1, impact:"medium", quickWin:true },
  { id:"F10", sev:"medium", cat:"Quality", title:"前端无 ESLint / Prettier", desc:"Vue SFC 文件无代码质量工具配置，风格和 import 顺序完全无约束。", file:"frontend/", fix:"引入 eslint + eslint-plugin-vue + prettier，CI 添加 lint 步骤。", effort:"low", hours:3, impact:"medium", quickWin:true },
  { id:"F11", sev:"medium", cat:"Testing", title:"verify 脚本无结构化输出契约", desc:"18 个 verify 脚本输出格式各异，CI 无法提取指标趋势数据。", file:"scripts/training/", fix:"定义统一 JSON 输出 schema（metrics/threshold/status），CI 收集绘制趋势。", effort:"medium", hours:12, impact:"medium", quickWin:false },
  { id:"F12", sev:"medium", cat:"Build", title:"打包流程纯手动", desc:"需手动执行 npm build → python build.py → NSIS，无一键构建脚本，无产物校验。", file:"desktop/build.py", fix:"创建 scripts/release.py 统一构建流程，含产物完整性校验。", effort:"medium", hours:8, impact:"medium", quickWin:false },
  { id:"F13", sev:"medium", cat:"Security", title:"速率限制器多 worker 下无效", desc:"RateLimiter 为进程内 dict，多 worker 部署时实际限制 = max × workers。", file:"api/middleware/security.py", fix:"文档说明单 worker 限制；或引入 Redis 共享存储。", effort:"medium", hours:8, impact:"low", quickWin:false },
  { id:"F14", sev:"medium", cat:"Quality", title:"无 pre-commit hooks", desc:"开发者可在不运行 lint/test 的情况下提交，所有问题只能等 CI 反馈。", file:"项目根目录", fix:"添加 .pre-commit-config.yaml 配置 ruff + black + 基础检查。", effort:"low", hours:1, impact:"medium", quickWin:true },
  { id:"F15", sev:"medium", cat:"Testing", title:"无 API 契约测试", desc:"20+ 路由变更可能在不通知前端的情况下破坏 API，无 schema 快照。", file:"api/app.py", fix:"添加 OpenAPI JSON schema snapshot test，CI 检测 schema 变更。", effort:"medium", hours:6, impact:"medium", quickWin:false },
  { id:"F16", sev:"medium", cat:"Security", title:"无依赖安全漏洞扫描", desc:"30+ 第三方依赖无 dependabot/renovate/pip-audit 扫描，CVE 可能持续存在。", file:"pyproject.toml", fix:"启用 Dependabot，CI 添加 pip-audit 扫描已知 CVE。", effort:"low", hours:2, impact:"medium", quickWin:true },
  { id:"F17", sev:"low", cat:"Workflow", title:"无标准化开发环境", desc:"新贡献者需手动阅读 CONTRIBUTING.md 安装依赖，无 devcontainer 定义。", file:"CONTRIBUTING.md", fix:"添加 .devcontainer/devcontainer.json 定义 Python 3.12 + Node 22 环境。", effort:"medium", hours:4, impact:"low", quickWin:false },
  { id:"F18", sev:"low", cat:"Testing", title:"全局状态重置粒度粗", desc:"conftest.py 仅在 session teardown 重置 app_state，无法隔离单个测试的状态污染。", file:"tests/conftest.py", fix:"评估 function 级别隔离；对有副作用的测试使用 autouse function fixture。", effort:"low", hours:2, impact:"low", quickWin:false },
];

const SEV: Record<string, { bg:string; ring:string; text:string; label:string }> = {
  critical: { bg:"#fef2f2", ring:"#fca5a5", text:"#dc2626", label:"Critical" },
  high:     { bg:"#fff7ed", ring:"#fdba74", text:"#ea580c", label:"High" },
  medium:   { bg:"#fefce8", ring:"#fde047", text:"#ca8a04", label:"Medium" },
  low:      { bg:"#f0fdf4", ring:"#86efac", text:"#16a34a", label:"Low" },
};

const IMPACT_MAP: Record<string,number> = { low:1, medium:2, high:3 };

const CAT_SCORES: { cat:string; score:string; color:string }[] = [
  { cat:"Security", score:"B", color:"#22c55e" },
  { cat:"Build", score:"C", color:"#eab308" },
  { cat:"Quality", score:"C", color:"#eab308" },
  { cat:"Workflow", score:"C-", color:"#f97316" },
  { cat:"CI/CD", score:"D+", color:"#f97316" },
  { cat:"Testing", score:"D", color:"#ef4444" },
];

const STRENGTHS = [
  "分层验证管线：8 个 verify 脚本 + 3 级 pytest 套件，从因果验证到回归测试层层递进",
  "指标口径白皮书严格区分 train_fit vs heldout，防止评估泄漏",
  "双入口 PyInstaller + MERGE 共享 _internal，打包设计精巧",
  "JWT fail-closed + 路径穿越防护 + 速率限制，安全基线扎实",
  "CONTRIBUTING.md 明确模块边界约束（seed/taiji/neuroplex 依赖方向），架构防腐",
];

const PHASES = [
  { name:"快速收益", time:"1-2 天", color:"#22c55e", items:[
    { text:"ruff + black CI 门禁", ref:"F01" },
    { text:"health 暴露安全中间件状态", ref:"F07" },
    { text:"pre-commit hooks", ref:"F14" },
    { text:"ESLint + Prettier", ref:"F10" },
    { text:"CI 产物上传", ref:"F09" },
  ]},
  { name:"测试基础设施", time:"1 周", color:"#3b82f6", items:[
    { text:"pytest-cov 覆盖率追踪", ref:"F05" },
    { text:"vitest + vue-test-utils", ref:"F02" },
    { text:"统一 verify 输出 schema", ref:"F11" },
    { text:"OpenAPI 契约测试", ref:"F15" },
  ]},
  { name:"CI 增强", time:"1 周", color:"#a855f7", items:[
    { text:"Windows runner", ref:"F08" },
    { text:"构建/打包验证", ref:"F03" },
    { text:"依赖安全扫描", ref:"F16" },
    { text:"版本号统一管理", ref:"F04" },
  ]},
  { name:"工程化", time:"持续", color:"#f97316", items:[
    { text:"静态类型检查", ref:"F06" },
    { text:"自动化构建脚本", ref:"F12" },
    { text:"标准化开发环境", ref:"F17" },
    { text:"多 worker 速率限制", ref:"F13" },
  ]},
];

// ── Helpers ───────────────────────────────────────────────────────────────────

const s = (v: React.CSSProperties) => v;

function ScoreGauge({ score }: { score: string }) {
  // Map letter grade to percentage: A+=95, A=90, B+=85, B=80, C+=75, C=70, D+=65, D=60
  const pctMap: Record<string,number> = { "A+":95, A:90, "B+":85, B:80, "C+":75, C:70, "D+":65, D:60, F:40 };
  const pct = pctMap[score] ?? 50;
  const r = 42, cx = 54, cy = 54;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ * 0.75; // 270° arc
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
    { label:"Critical", value:2, color:"#ef4444" },
    { label:"High", value:5, color:"#f97316" },
    { label:"Medium", value:7, color:"#eab308" },
    { label:"Low", value:4, color:"#22c55e" },
  ];
  const total = data.reduce((a, d) => a + d.value, 0);
  const r = 40, cx = 56, cy = 56, sw = 14;
  const circ = 2 * Math.PI * r;
  const GAP = 3; // gap in px between segments
  let offset = 0;
  return (
    <div style={{ display:"flex", alignItems:"center", gap:16 }}>
      <svg width={112} height={112} viewBox="0 0 112 112">
        {data.map(d => {
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
  const xMax = 44;
  const toX = (h: number) => PAD + (h / xMax) * (W - PAD * 2);
  const toY = (imp: number) => H - PAD - ((imp - 0.5) / 3) * (H - PAD * 2);
  // jitter overlapping dots: group by (hours, impact) and offset
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
      {/* quadrant background */}
      <rect x={PAD} y={PAD} width={(W - PAD * 2) * 0.35} height={(H - PAD * 2) * 0.5}
        rx={4} fill="#22c55e" opacity={0.04} />
      {/* quadrant lines */}
      <line x1={PAD} y1={H / 2} x2={W - PAD} y2={H / 2} stroke="#e5e7eb" strokeWidth={1} strokeDasharray="4 3" />
      <line x1={W * 0.38} y1={PAD} x2={W * 0.38} y2={H - PAD} stroke="#e5e7eb" strokeWidth={1} strokeDasharray="4 3" />
      {/* quadrant labels */}
      <text x={PAD + 4} y={PAD + 10} fontSize={8} fill="#86efac" fontFamily="-apple-system,sans-serif" fontWeight={600}>Quick Wins</text>
      <text x={W - PAD - 4} y={H - PAD - 4} fontSize={7} fill="#e5e7eb" textAnchor="end" fontFamily="-apple-system,sans-serif">慎重考虑</text>
      {/* axis labels */}
      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={9} fill="#9ca3af" fontFamily="-apple-system,sans-serif">投入 (hours) →</text>
      <text x={10} y={H / 2} textAnchor="middle" fontSize={9} fill="#9ca3af" fontFamily="-apple-system,sans-serif"
        transform={`rotate(-90 10 ${H/2})`}>收益 →</text>
      {/* dots */}
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
    <div style={{ display:"flex", gap:8, alignItems:"center" }}>
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
      {/* timeline line */}
      <div style={s({
        position:"absolute", left:7, top:8, bottom:8, width:2,
        background:"linear-gradient(to bottom, #22c55e, #3b82f6, #a855f7, #f97316)",
        borderRadius:2, opacity:0.3,
      })} />
      {PHASES.map((p, i) => (
        <div key={i} style={s({ position:"relative", marginBottom: i < PHASES.length - 1 ? 16 : 0 })}>
          {/* timeline dot */}
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

export default function SeedHarnessReport() {
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
            background:"radial-gradient(circle,rgba(99,102,241,0.15),transparent 70%)",
          })} />
          <div style={s({ display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:16, position:"relative" })}>
            <div style={s({ flex:1, minWidth:240 })}>
              <h1 style={s({ fontSize:22, fontWeight:800, margin:0, letterSpacing:-0.4, lineHeight:1.3 })}>
                Seed — Harness 实践分析
              </h1>
              <p style={s({ fontSize:12, color:"#94a3b8", margin:"4px 0 0", lineHeight:1.5 })}>
                CI/CD · 测试 · 构建 · 安全 · 开发工作流 — 18 项发现
              </p>
              <p style={s({ fontSize:11, color:"#64748b", margin:"8px 0 0", lineHeight:1.5, maxWidth:420 })}>
                后端验证管线和安全基线扎实（B+），但 CI 缺 lint/build 门禁、前端零测试、无覆盖率追踪、无类型检查，综合拉至 C+。
              </p>
            </div>
            <ScoreGauge score="C+" />
          </div>
          {/* stat row */}
          <div style={s({ display:"flex", gap:8, marginTop:18, flexWrap:"wrap" })}>
            {[
              { label:"Critical", val:2, c:"#ef4444" },
              { label:"High", val:5, c:"#f97316" },
              { label:"Medium", val:7, c:"#eab308" },
              { label:"Low", val:4, c:"#22c55e" },
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
            ⚡ Quick Wins — 低投入高收益，立即可做
          </h3>
          <div style={s({ display:"flex", flexWrap:"wrap", gap:6 })}>
            {quickWins.map(f => (
              <span key={f.id} style={s({
                fontSize:11, fontWeight:600, color:"#059669", background:"#fff",
                borderRadius:6, padding:"3px 10px", border:"1px solid #a7f3d0",
                cursor:"default", lineHeight:1.6,
              })}>{f.id} {f.title} <span style={s({ color:"#9ca3af", fontWeight:400 })}· {f.hours}h</span></span>
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
              {/* search */}
              <input
                value={search} onChange={e => setSearch(e.target.value)}
                placeholder="搜索..."
                style={s({
                  fontSize:11, padding:"4px 10px", borderRadius:6, border:"1px solid #d1d5db",
                  outline:"none", width:120, color:"#374151", background:"#fafafa",
                })}
              />
              {/* filter buttons */}
              {["all","critical","high","medium","low"].map(k => {
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
          Seed Harness Analysis · 2026-08-24 · /better-harness
        </div>
      </div>
    </div>
  );
}
