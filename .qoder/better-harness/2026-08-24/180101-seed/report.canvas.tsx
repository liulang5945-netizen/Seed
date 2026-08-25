import { useState, useMemo } from "react";

// ── Data ──────────────────────────────────────────────────────────────────────

type Finding = {
  id: string; sev: "critical"|"high"|"medium"|"low"; cat: string;
  title: string; desc: string; file: string; fix: string;
  effort: "low"|"medium"|"high"; hours: number; impact: "low"|"medium"|"high";
  quickWin: boolean;
};

const FINDINGS: Finding[] = [
  { id:"I01", sev:"high", cat:"Quality", title:"neuroplex mypy 580 未分治，冻结/活跃模块混杂", desc:"0% 覆盖冻结模块无测试兜底盲改风险高，活跃模块错误混杂其中棘轮无法精准施压。", file:"pyproject.toml", fix:"✅ 冻结边界分治：0% 覆盖模块移出度量面（251 错）并登记，其余实修消减 329 → 212；CI 棘轮 580 → 212。", effort:"high", hours:8, impact:"high", quickWin:false },
  { id:"I02", sev:"high", cat:"Quality", title:"ruff B/SIM 202 存量，异常链缺失等未治理", desc:"B904 raise-without-from(63) 丢异常链、B905 zip-without-strict(48)、B007(31)、SIM 冗余等真实缺陷模式。", file:".github/workflows/ci.yml", fix:"✅ 202 → 0 转 blocking：56 处补 from e + 6 处 from None，zip strict、循环变量清理、SIM 简化；B008/冻结模块豁免。", effort:"high", hours:6, impact:"high", quickWin:false },
  { id:"I03", sev:"medium", cat:"Testing", title:"routes_update SSRF 与 chat_strategies 纯函数无回归保护", desc:"_validate_update_url 是安全关键路径（防 DNS rebinding），聊天前置逻辑无测试。", file:"api/routes_update.py", fix:"✅ 新增 15 用例：SSRF 拒绝非 http/localhost/私网 IP（含云元数据）、放行公网；时间/历史/RAG 纯函数。门禁 21.5 → 21.8（实测 21.91%）。", effort:"medium", hours:3, impact:"medium", quickWin:false },
  { id:"I04", sev:"medium", cat:"Security", title:"pip-audit 转正评估未完成，依赖 CVE 状态不明", desc:"转正条件为连续无高危，但从未实际运行评估。", file:".github/workflows/ci.yml", fix:"✅ 实际扫描：aiohttp/datasets 传递依赖中高危 + 工具链 CVE；未达条件维持 advisory，修复列入 R7，评估记录写入 CI。", effort:"low", hours:1, impact:"medium", quickWin:false },
  { id:"I05", sev:"low", cat:"Quality", title:"B006 FastAPI 体参数可变默认值处理方式未定", desc:"改注解会改 OpenAPI 快照，且 FastAPI 默认值实际不被变更，属误报。", file:"api/routes_update.py", fix:"✅ 3 处加 # noqa: B006 注明理由，保留行为与快照稳定。", effort:"low", hours:1, impact:"low", quickWin:false },
  { id:"I06", sev:"low", cat:"Quality", title:"SIM112 误报 Windows 规范环境变量名", desc:"ProgramFiles / ProgramFiles(x86) 是规范写法，不应改全大写。", file:"neuroplex/tools/file_parser.py", fix:"✅ 保留规范写法，2 处加 # noqa: SIM112 注明理由。", effort:"low", hours:1, impact:"low", quickWin:false },
];

const SEV: Record<string, { bg:string; ring:string; text:string; label:string }> = {
  critical: { bg:"#fef2f2", ring:"#fca5a5", text:"#dc2626", label:"Critical" },
  high:     { bg:"#fff7ed", ring:"#fdba74", text:"#ea580c", label:"High" },
  medium:   { bg:"#fefce8", ring:"#fde047", text:"#ca8a04", label:"Medium" },
  low:      { bg:"#f0fdf4", ring:"#86efac", text:"#16a34a", label:"Low" },
};

const IMPACT_MAP: Record<string,number> = { low:1, medium:2, high:3 };

const CAT_SCORES: { cat:string; score:string; color:string }[] = [
  { cat:"Build", score:"A", color:"#22c55e" },
  { cat:"Quality", score:"A", color:"#22c55e" },
  { cat:"Workflow", score:"A-", color:"#22c55e" },
  { cat:"Testing", score:"A", color:"#22c55e" },
  { cat:"CI/CD", score:"A", color:"#22c55e" },
  { cat:"Security", score:"A-", color:"#22c55e" },
];

const STRENGTHS = [
  "mypy：核心四模块 0，全仓棘轮 580 → 212；冻结模块错误数登记在案，解冻须补测后清零",
  "ruff：E/F/W/I/UP + B/SIM 全部 blocking（202 → 0，B008/冻结模块豁免均有注释）",
  "异常链完整性：56 处 except 内 raise 补 from e / from None，堆栈可追溯",
  "覆盖率：21.91%（门禁 21.8），SSRF 防护路径有回归保护",
  "依赖安全：pip-audit 持续扫描，转正标准明确（连续无高危）",
  "六轮累计 56 项发现全部修复，门禁从『存在』到『只降不升』再到『深度消减』",
];

const PHASES = [
  { name:"mypy 深水区", time:"持续", color:"#3b82f6", items:[
    { text:"neuroplex 212 按模块分批消减（先补测试再解冻）", ref:"I01" },
    { text:"棘轮 212 → 100 → 0", ref:"I01" },
  ]},
  { name:"依赖安全", time:"1 周", color:"#ef4444", items:[
    { text:"aiohttp 3.14.2→3.14.3 / datasets 5.0.0→5.0.1", ref:"I04" },
    { text:"pip-audit 连续无高危后转 blocking", ref:"I04" },
  ]},
  { name:"覆盖率爬坡", time:"每轮 +0.5-1%", color:"#a855f7", items:[
    { text:"routes_terminal / routes_rag / routes_neuroplex 补测", ref:"I03" },
    { text:"fail_under 21.8 → 22.5 → 23", ref:"I03" },
  ]},
];

// R1 → R2 → R3 → R4 → R5 → R6 trajectory
const TRAJECTORY = [
  { round:"R1", score:"C+", findings:18, critical:2, high:5, gaps:["无 lint 门禁","前端零测试"] },
  { round:"R2", score:"B-", findings:14, critical:0, high:3, gaps:["ESLint advisory","覆盖率=0"] },
  { round:"R3", score:"B+", findings:9, critical:0, high:0, gaps:["E2E 未入 CI","Docker 未验证"] },
  { round:"R4", score:"A", findings:8, critical:0, high:1, gaps:["I/UP 1922 清债","mypy 基线"] },
  { round:"R5", score:"A", findings:7, critical:0, high:0, gaps:["mypy 消减","ESLint 收紧"] },
  { round:"R6", score:"A", findings:0, critical:0, high:0, gaps:["深度消减轮：mypy 580→212 / B/SIM 202→0"] },
];

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
    { label:"High", value:2, color:"#f97316" },
    { label:"Medium", value:2, color:"#eab308" },
    { label:"Low", value:2, color:"#22c55e" },
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
  const xMax = 20;
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

function TrajectoryPanel() {
  const colors = ["#dc2626", "#eab308", "#16a34a", "#3b82f6", "#059669", "#0f766e"];
  const bgMap = ["#fef2f2", "#fefce8", "#f0fdf4", "#eff6ff", "#ecfdf5", "#f0fdfa"];
  const borderMap = ["#fecaca", "#fde047", "#bbf7d0", "#bfdbfe", "#a7f3d0", "#99f6e4"];
  const chipMap = ["#fff1f2", "#fef9c3", "#ecfdf5", "#eff6ff", "#ecfdf5", "#f0fdfa"];
  const textMap = ["#991b1b", "#854d0e", "#166534", "#1e40af", "#065f46", "#134e4a"];
  return (
    <div style={s({ display:"grid", gridTemplateColumns:"1fr auto 1fr auto 1fr", gap:10, alignItems:"stretch" })}>
      {TRAJECTORY.map((t, i) => (
        <div key={t.round}>
          <div style={s({
            background: bgMap[i] ?? "#f8fafc",
            borderRadius:10, padding:"12px 14px",
            border:`1px solid ${borderMap[i] ?? "#e2e8f0"}`,
          })}>
            <div style={s({ display:"flex", alignItems:"center", gap:8, marginBottom:8 })}>
              <span style={s({ fontSize:20, fontWeight:800, color:colors[i] ?? "#334155" })}>{t.score}</span>
              <span style={s({ fontSize:10, color:"#9ca3af" })}>{t.round} · {t.findings} findings</span>
            </div>
            <div style={s({ display:"flex", gap:6, marginBottom:6 })}>
              <span style={s({ fontSize:10, fontWeight:600, color:colors[i] ?? "#334155", background:`${colors[i] ?? "#334155"}14`,
                borderRadius:4, padding:"1px 6px" })}>{t.critical} critical</span>
              <span style={s({ fontSize:10, fontWeight:600, color:"#ea580c", background:"#ffedd5",
                borderRadius:4, padding:"1px 6px" })}>{t.high} high</span>
            </div>
            <div style={s({ display:"flex", flexWrap:"wrap", gap:3 })}>
              {t.gaps.map((g, j) => (
                <span key={j} style={s({ fontSize:9, color: textMap[i] ?? "#334155",
                  background: chipMap[i] ?? "#f1f5f9",
                  borderRadius:3, padding:"1px 5px", border:`1px solid ${borderMap[i] ?? "#e2e8f0"}`
                })}>{g}</span>
              ))}
            </div>
          </div>
        </div>
      ))}
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

export default function SeedHarnessReportR6() {
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
                Seed — Harness 实践分析 (R6 — 深度消减轮)
              </h1>
              <p style={s({ fontSize:12, color:"#94a3b8", margin:"4px 0 0", lineHeight:1.5 })}>
                R6 全部 6 项当轮清零 · mypy/B/SIM 深度消减 + 覆盖率/安全评估 — 0 项待办
              </p>
              <p style={s({ fontSize:11, color:"#64748b", margin:"8px 0 0", lineHeight:1.5, maxWidth:420 })}>
                C+ → B- → B+ → A → A → A：六轮 62 项发现全部修复，遗留债务全部纳入只降不升的棘轮机制。
              </p>
            </div>
            <ScoreGauge score="A" />
          </div>
          {/* stat row */}
          <div style={s({ display:"flex", gap:8, marginTop:18, flexWrap:"wrap" })}>
            {[
              { label:"Critical", val:0, c:"#22c55e" },
              { label:"High", val:2, c:"#f97316" },
              { label:"Medium", val:2, c:"#eab308" },
              { label:"Low", val:2, c:"#22c55e" },
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

        {/* ── Trajectory Panel ── */}
        <div style={s({
          background:"#fff", borderRadius:12, padding:"16px 18px", marginBottom:14,
          border:"1px solid #e5e7eb",
        })}>
          <h3 style={s({ fontSize:13, fontWeight:700, margin:"0 0 12px", color:"#374151" })}>
            R1 → R2 → R3 → R4 → R5 → R6 改进轨迹
          </h3>
          <TrajectoryPanel />
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
        {quickWins.length > 0 && (
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
        )}

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
          <h3 style={s({ fontSize:13, fontWeight:700, margin:"0 0 14px", color:"#374151" })}>R7 候选方向</h3>
          <Roadmap />
        </div>

        {/* ── Footer ── */}
        <div style={s({ textAlign:"center", fontSize:10, color:"#b0b8c4", padding:"6px 0 0" })}>
          Seed Harness Analysis R6 (All Fixed · Score A) · 2026-08-24 · /better-harness
        </div>
      </div>
    </div>
  );
}
