"""
Linear Programming Solver — Big M Method (Enhanced + Fixed)
Run: python -m streamlit run main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Big M Solver", page_icon="📐", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800;900&family=Space+Mono:wght@400;700&family=Rajdhani:wght@300;400;500;600;700&display=swap');

:root{
  --bg:#020408;
  --bg2:#060b14;
  --card:#080f1c;
  --card2:#0b1424;
  --border:#0e2040;
  --border2:#142a50;
  --border3:#1a3a68;
  --acc:#00d4ff;
  --acc-glow:rgba(0,212,255,0.15);
  --acc-deep:rgba(0,212,255,0.06);
  --acc2:#ff6b00;
  --acc2-glow:rgba(255,107,0,0.15);
  --acc3:#7b2fff;
  --acc3-glow:rgba(123,47,255,0.15);
  --gold:#ffd700;
  --gold-glow:rgba(255,215,0,0.15);
  --key-col:#ff4500;
  --key-col-glow:rgba(255,69,0,0.2);
  --key-row:#00ff88;
  --key-row-glow:rgba(0,255,136,0.15);
  --pivot:#ff0080;
  --pivot-glow:rgba(255,0,128,0.25);
  --txt:#c8dff0;
  --txt2:#5a7a99;
  --good:#00ff88;
  --bad:#ff3344;
  --good-bg:rgba(0,255,136,0.05);
  --bad-bg:rgba(255,51,68,0.08);
}

html,body,[class*="css"]{
  font-family:'Rajdhani',sans-serif;
  background:var(--bg);
  color:var(--txt);
}

/* ── Animated background grid ── */
body::before{
  content:'';
  position:fixed;
  inset:0;
  background-image:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
  background-size:50px 50px;
  pointer-events:none;
  z-index:0;
  animation:gridPulse 8s ease-in-out infinite;
}
@keyframes gridPulse{
  0%,100%{opacity:0.5;}
  50%{opacity:1;}
}

/* Scrollbar */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg2);}
::-webkit-scrollbar-thumb{background:var(--acc);border-radius:2px;box-shadow:0 0 6px var(--acc);}

/* ── HEADER ── */
.hdr{
  text-align:center;
  padding:3.5rem 0 3rem;
  position:relative;
  margin-bottom:3rem;
}
.hdr::before{
  content:'';
  position:absolute;
  inset:0;
  background:radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,212,255,0.08), transparent 70%);
  pointer-events:none;
}
.hdr::after{
  content:'';
  position:absolute;
  bottom:0;left:5%;right:5%;
  height:1px;
  background:linear-gradient(to right,transparent,var(--acc),var(--acc3),var(--acc),transparent);
  box-shadow:0 0 20px var(--acc);
}

.hdr-badge{
  display:inline-flex;
  align-items:center;
  gap:.5rem;
  font-family:'Space Mono',monospace;
  font-size:.65rem;
  letter-spacing:.2em;
  color:var(--acc);
  background:var(--acc-deep);
  border:1px solid rgba(0,212,255,0.3);
  border-radius:2px;
  padding:.35rem 1.2rem;
  margin-bottom:1.5rem;
  text-transform:uppercase;
  box-shadow:0 0 20px var(--acc-glow), inset 0 0 20px rgba(0,212,255,0.03);
  animation:badgePulse 3s ease-in-out infinite;
}
@keyframes badgePulse{
  0%,100%{box-shadow:0 0 15px var(--acc-glow),inset 0 0 10px rgba(0,212,255,0.03);}
  50%{box-shadow:0 0 30px var(--acc-glow),inset 0 0 20px rgba(0,212,255,0.06);}
}

.hdr h1{
  font-family:'Orbitron',sans-serif;
  font-size:3.4rem;
  font-weight:900;
  color:#fff;
  margin:0 0 .6rem;
  letter-spacing:.04em;
  line-height:1;
  text-shadow:0 0 40px rgba(0,212,255,0.4), 0 0 80px rgba(0,212,255,0.15);
}
.hdr h1 .accent{
  color:var(--acc);
  text-shadow:0 0 30px var(--acc), 0 0 60px rgba(0,212,255,0.4);
}
.hdr h1 .accent2{
  color:var(--gold);
  text-shadow:0 0 30px var(--gold), 0 0 60px rgba(255,215,0,0.4);
}
.hdr-sub{
  font-family:'Space Mono',monospace;
  font-size:.75rem;
  color:var(--txt2);
  letter-spacing:.12em;
  margin-top:.8rem;
}
.hdr-sub span{color:var(--acc);opacity:.7;}

/* ── Corner decorations ── */
.corner-tl,.corner-tr,.corner-bl,.corner-br{
  position:absolute;
  width:30px;height:30px;
  border-color:var(--acc);
  border-style:solid;
  opacity:.5;
}
.corner-tl{top:1rem;left:1rem;border-width:2px 0 0 2px;}
.corner-tr{top:1rem;right:1rem;border-width:2px 2px 0 0;}
.corner-bl{bottom:1rem;left:1rem;border-width:0 0 2px 2px;}
.corner-br{bottom:1rem;right:1rem;border-width:0 2px 2px 0;}

/* ── Section headers ── */
.sec{
  display:flex;
  align-items:center;
  gap:1rem;
  margin:2.5rem 0 1.5rem;
  position:relative;
}
.sec-num{
  font-family:'Orbitron',sans-serif;
  font-size:.7rem;
  font-weight:900;
  color:var(--bg);
  background:var(--acc);
  width:2rem;height:2rem;
  border-radius:2px;
  display:flex;align-items:center;justify-content:center;
  flex-shrink:0;
  box-shadow:0 0 15px var(--acc-glow), 0 0 30px var(--acc-glow);
}
.sec-title{
  font-family:'Orbitron',sans-serif;
  font-size:1rem;
  font-weight:700;
  color:#fff;
  letter-spacing:.1em;
  text-transform:uppercase;
  text-shadow:0 0 20px rgba(255,255,255,0.2);
}
.sec-line{
  flex:1;height:1px;
  background:linear-gradient(to right,var(--border3),transparent);
  box-shadow:0 0 8px rgba(0,212,255,0.1);
}

/* ── Cards ── */
.card{
  background:var(--card);
  border:1px solid var(--border2);
  border-radius:4px;
  padding:1.6rem;
  margin-bottom:1rem;
  position:relative;
  box-shadow:0 4px 30px rgba(0,0,0,0.5), inset 0 0 40px rgba(0,212,255,0.015);
}
.card::before{
  content:'';
  position:absolute;
  top:0;left:0;right:0;
  height:1px;
  background:linear-gradient(to right,transparent,var(--acc),transparent);
  opacity:.4;
}

/* ── Problem preset card ── */
.preset-box{
  background:linear-gradient(135deg,rgba(0,212,255,0.05),rgba(123,47,255,0.05));
  border:1px solid var(--border3);
  border-radius:4px;
  padding:1.2rem 1.6rem;
  margin-bottom:1.5rem;
  position:relative;
  overflow:hidden;
}
.preset-box::after{
  content:'PRESET LOADED';
  position:absolute;top:.7rem;right:1rem;
  font-family:'Space Mono',monospace;
  font-size:.6rem;letter-spacing:.15em;
  color:var(--acc);opacity:.4;
}
.preset-eq{
  font-family:'Space Mono',monospace;
  font-size:.82rem;
  color:var(--acc);
  margin:.4rem 0;
  line-height:1.8;
}
.preset-constraint{
  font-family:'Space Mono',monospace;
  font-size:.78rem;
  color:var(--txt2);
  margin:.2rem 0;
}

/* ── Pivot info banner ── */
.pivot-banner{
  display:flex;
  align-items:center;
  gap:1.5rem;
  background:linear-gradient(135deg,rgba(0,212,255,0.03),rgba(123,47,255,0.03));
  border:1px solid var(--border3);
  border-left:3px solid var(--acc);
  border-radius:0 4px 4px 0;
  padding:.8rem 1.4rem;
  margin:.6rem 0 .8rem;
  flex-wrap:wrap;
  position:relative;
  overflow:hidden;
}
.pivot-banner::before{
  content:'';
  position:absolute;inset:0;
  background:linear-gradient(90deg,var(--acc-deep),transparent);
  pointer-events:none;
}

.pivot-chip{
  display:inline-flex;
  align-items:center;
  gap:.4rem;
  padding:.3rem .8rem;
  border-radius:2px;
  font-family:'Space Mono',monospace;
  font-size:.78rem;
  font-weight:700;
  position:relative;
  z-index:1;
}
.chip-enter{
  background:var(--key-col-glow);
  border:1px solid var(--key-col);
  color:var(--key-col);
  box-shadow:0 0 12px var(--key-col-glow);
}
.chip-leave{
  background:var(--key-row-glow);
  border:1px solid var(--key-row);
  color:var(--key-row);
  box-shadow:0 0 12px var(--key-row-glow);
}
.chip-pivot-val{
  background:var(--pivot-glow);
  border:1px solid var(--pivot);
  color:var(--pivot);
  box-shadow:0 0 12px var(--pivot-glow);
}
.chip-optimal{
  background:var(--good-bg);
  border:1px solid var(--good);
  color:var(--good);
  box-shadow:0 0 12px rgba(0,255,136,0.2);
}
.arrow-sym{color:var(--txt2);font-size:1.1rem;position:relative;z-index:1;}
.pivot-explain{
  font-family:'Space Mono',monospace;
  font-size:.72rem;
  color:var(--txt2);
  border-left:1px solid var(--border3);
  padding-left:1.2rem;
  margin-left:auto;
  flex:1;
  min-width:220px;
  line-height:1.6;
  position:relative;
  z-index:1;
}

/* ── Iteration header ── */
.iter-hdr{
  display:flex;
  align-items:center;
  gap:1rem;
  margin:2rem 0 .6rem;
}
.iter-num{
  font-family:'Orbitron',sans-serif;
  font-size:.75rem;
  font-weight:700;
  color:var(--acc3);
  background:rgba(123,47,255,0.12);
  border:1px solid rgba(123,47,255,0.35);
  border-radius:2px;
  padding:.3rem 1rem;
  letter-spacing:.1em;
  text-transform:uppercase;
  box-shadow:0 0 12px rgba(123,47,255,0.2);
}
.iter-line{flex:1;height:1px;background:linear-gradient(to right,var(--border2),transparent);}

/* ── Legend ── */
.legend{
  display:flex;
  gap:1.5rem;
  flex-wrap:wrap;
  padding:.9rem 1.4rem;
  background:var(--card);
  border:1px solid var(--border2);
  border-radius:4px;
  margin-bottom:1.2rem;
}
.legend-item{
  display:flex;align-items:center;gap:.6rem;
  font-family:'Space Mono',monospace;
  font-size:.68rem;
  color:var(--txt2);
  letter-spacing:.03em;
}
.legend-dot{width:10px;height:10px;border-radius:1px;flex-shrink:0;}

/* ── Result ── */
.resbox{
  background:linear-gradient(135deg,rgba(0,255,136,0.04),var(--card));
  border:1px solid rgba(0,255,136,0.25);
  border-radius:4px;
  padding:2rem 2.5rem;
  margin-top:.8rem;
  position:relative;
  overflow:hidden;
  box-shadow:0 0 40px rgba(0,255,136,0.06), inset 0 0 40px rgba(0,255,136,0.02);
}
.resbox::before{
  content:'';
  position:absolute;top:0;left:0;right:0;
  height:2px;
  background:linear-gradient(to right,transparent,var(--good),transparent);
  box-shadow:0 0 15px var(--good);
}
.resbox::after{
  content:'✦ OPTIMAL';
  position:absolute;top:1rem;right:1.5rem;
  font-family:'Orbitron',sans-serif;
  font-size:.6rem;
  letter-spacing:.2em;
  color:var(--good);
  opacity:.35;
}
.resbox h3{
  font-family:'Orbitron',sans-serif;
  color:var(--good);
  font-size:1rem;
  font-weight:700;
  margin:0 0 1.2rem;
  letter-spacing:.1em;
  text-shadow:0 0 20px rgba(0,255,136,0.4);
}
.resvar{
  font-family:'Space Mono',monospace;
  font-size:.88rem;
  color:var(--txt);
  margin:.4rem 0;
  display:flex;align-items:center;gap:.6rem;
}
.resvar-name{color:var(--acc);font-weight:700;min-width:3rem;}
.resvar-eq{color:var(--txt2);}
.resvar-val{
  color:var(--gold);
  font-weight:700;
  text-shadow:0 0 12px rgba(255,215,0,0.4);
}
.resobj{
  font-family:'Orbitron',sans-serif;
  font-size:2rem;
  font-weight:900;
  color:var(--gold);
  margin-top:1.2rem;
  letter-spacing:-.01em;
  text-shadow:0 0 30px rgba(255,215,0,0.5);
}
.resobj span{
  font-size:.85rem;
  color:var(--txt2);
  font-weight:400;
  margin-left:.4rem;
  font-family:'Space Mono',monospace;
}

.errbox{
  background:var(--bad-bg);
  border:1px solid rgba(255,51,68,0.3);
  border-radius:4px;
  padding:1.2rem 1.6rem;
  color:var(--bad);
  font-family:'Space Mono',monospace;
  font-size:.82rem;
  box-shadow:0 0 20px rgba(255,51,68,0.08);
}

/* ── Button ── */
.stButton>button{
  background:linear-gradient(135deg,var(--acc),#0099bb)!important;
  color:#020408!important;
  font-family:'Orbitron',sans-serif!important;
  font-weight:700!important;
  font-size:.85rem!important;
  border:none!important;
  border-radius:3px!important;
  padding:.75rem 2.5rem!important;
  letter-spacing:.1em!important;
  text-transform:uppercase!important;
  transition:all .2s!important;
  box-shadow:0 0 25px var(--acc-glow), 0 4px 20px rgba(0,0,0,0.4)!important;
}
.stButton>button:hover{
  transform:translateY(-2px)!important;
  box-shadow:0 0 40px rgba(0,212,255,0.4), 0 8px 30px rgba(0,0,0,0.5)!important;
}

/* ── Inputs ── */
.stSelectbox>div>div,
.stNumberInput>div>div>input{
  background:var(--card2)!important;
  border:1px solid var(--border3)!important;
  color:var(--txt)!important;
  border-radius:3px!important;
  font-family:'Space Mono',monospace!important;
}
.stSelectbox>div>div:focus-within,
.stNumberInput>div>div>input:focus{
  border-color:var(--acc)!important;
  box-shadow:0 0 10px var(--acc-glow)!important;
}

/* ── Expander ── */
.stExpander{
  background:var(--card)!important;
  border:1px solid var(--border2)!important;
  border-radius:4px!important;
}
.stExpander summary{
  font-family:'Orbitron',sans-serif!important;
  font-size:.8rem!important;
  letter-spacing:.05em!important;
}

/* ── Stats row ── */
.stats-row{
  display:flex;gap:1rem;margin-bottom:2rem;flex-wrap:wrap;
}
.stat-card{
  flex:1;min-width:130px;
  background:var(--card2);
  border:1px solid var(--border2);
  border-radius:4px;
  padding:1.2rem 1.4rem;
  text-align:center;
  position:relative;
  overflow:hidden;
  transition:border-color .3s;
}
.stat-card:hover{border-color:var(--acc);}
.stat-card::before{
  content:'';
  position:absolute;top:0;left:0;right:0;
  height:2px;
  background:linear-gradient(to right,var(--acc3),var(--acc));
}
.stat-val{
  font-family:'Orbitron',sans-serif;
  font-size:1.8rem;
  font-weight:900;
  color:var(--acc);
  text-shadow:0 0 15px var(--acc-glow);
}
.stat-lbl{
  font-family:'Space Mono',monospace;
  font-size:.62rem;
  color:var(--txt2);
  letter-spacing:.1em;
  text-transform:uppercase;
  margin-top:.3rem;
}

/* ── Divider ── */
.divider{
  border:none;height:1px;
  background:linear-gradient(to right,transparent,var(--border3),transparent);
  margin:2.5rem 0;
  box-shadow:0 0 8px rgba(0,212,255,0.1);
}

/* ── Vis label ── */
.vis-label{
  font-family:'Space Mono',monospace;
  font-size:.72rem;
  color:var(--txt2);
  margin-bottom:1rem;
  letter-spacing:.08em;
  padding:.5rem 1rem;
  background:var(--card);
  border-left:2px solid var(--acc);
  display:inline-block;
}

/* Input label overrides */
label{
  font-family:'Space Mono',monospace!important;
  font-size:.72rem!important;
  color:var(--txt2)!important;
  letter-spacing:.06em!important;
}

/* constraint row label */
.con-label{
  font-family:'Space Mono',monospace;
  color:var(--acc2);
  font-size:.72rem;
  margin:.8rem 0 .3rem;
  letter-spacing:.08em;
  text-transform:uppercase;
  display:flex;align-items:center;gap:.5rem;
}
.con-label::before{
  content:'';
  display:inline-block;
  width:4px;height:4px;
  background:var(--acc2);
  border-radius:50%;
  box-shadow:0 0 6px var(--acc2);
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SOLVER
# ═══════════════════════════════════════════

class BigMSolver:
    BIG_M = 1e6
    MAX_ITER = 200

    def __init__(self, c, A, b, types, sense="max"):
        self.c_orig = np.array(c, dtype=float)
        self.A_orig = np.array(A, dtype=float)
        self.b_orig = np.array(b, dtype=float)
        self.types  = types
        self.sense  = sense
        self.n_vars = len(c)
        self.m_cons = len(b)
        self.iterations = []
        self.status = None
        self.solution = None

    def _build_tableau(self):
        n, m = self.n_vars, self.m_cons
        extra, col_names, basis = [], [f"x{i+1}" for i in range(n)], []
        sc = ac = 0
        for i, t in enumerate(self.types):
            if t == "<=":
                v = np.zeros(m); v[i] = 1.0
                extra.append(v); col_names.append(f"s{sc+1}"); sc += 1
                basis.append(n + len(extra) - 1)
            elif t == ">=":
                v = np.zeros(m); v[i] = -1.0
                extra.append(v); col_names.append(f"s{sc+1}"); sc += 1
                v = np.zeros(m); v[i] = 1.0
                extra.append(v); col_names.append(f"a{ac+1}"); ac += 1
                basis.append(n + len(extra) - 1)
            elif t == "=":
                v = np.zeros(m); v[i] = 1.0
                extra.append(v); col_names.append(f"a{ac+1}"); ac += 1
                basis.append(n + len(extra) - 1)
        self.col_names = col_names
        self.basis = basis
        em = np.column_stack(extra) if extra else np.empty((m, 0))
        body = np.hstack([self.A_orig, em])
        total = body.shape[1]
        self.n_total = total
        cb = self.c_orig.copy() if self.sense == "max" else -self.c_orig.copy()
        cf = np.concatenate([cb, np.zeros(total - n)])
        for j, nm in enumerate(col_names):
            if nm.startswith("a"):
                cf[j] = -self.BIG_M
        self.c_full = cf
        self.tableau = np.hstack([body, self.b_orig.reshape(-1, 1)])
        self.rhs_idx = total
        return self._zrow()

    def _zrow(self):
        z = np.zeros(self.n_total + 1)
        for j in range(self.n_total):
            z[j] = sum(self.c_full[self.basis[i]] * self.tableau[i, j]
                       for i in range(self.m_cons)) - self.c_full[j]
        z[self.rhs_idx] = sum(self.c_full[self.basis[i]] * self.tableau[i, self.rhs_idx]
                              for i in range(self.m_cons))
        return z

    def _snap(self, z, it, pc=None, pr=None):
        bnames = [self.col_names[b] for b in self.basis]
        sv = np.zeros(self.n_total)
        for i, b in enumerate(self.basis):
            sv[b] = self.tableau[i, self.rhs_idx]
        sol_col = [self.tableau[i, self.rhs_idx] for i in range(self.m_cons)] + [z[self.rhs_idx]]
        data = np.vstack([self.tableau, z])
        df = pd.DataFrame(data, index=bnames + ["z_j-c_j"], columns=self.col_names + ["RHS"])
        df.insert(len(df.columns), "Solution", sol_col)
        pivot_val = float(self.tableau[pr, pc]) if (pr is not None and pc is not None) else None
        self.iterations.append({
            "iter": it, "df": df,
            "pc": self.col_names[pc] if pc is not None else None,
            "pr": bnames[pr] if pr is not None else None,
            "pr_idx": pr,
            "pc_idx": pc,
            "pivot_val": pivot_val,
            "sv": sv.copy(),
            "basis_names": bnames[:]
        })

    def solve(self):
        z = self._build_tableau()
        for it in range(self.MAX_ITER):
            if min(z[:self.n_total]) >= -1e-8:
                self._snap(z, it + 1)
                self._extract(z)
                return
            pc = int(np.argmin(z[:self.n_total]))
            ratios = [self.tableau[i, self.rhs_idx] / self.tableau[i, pc]
                      if self.tableau[i, pc] > 1e-8 else np.inf
                      for i in range(self.m_cons)]
            if all(r == np.inf for r in ratios):
                self._snap(z, it + 1, pc)
                self.status = "unbounded"; return
            pr = int(np.argmin(ratios))
            self._snap(z, it + 1, pc, pr)
            pe = self.tableau[pr, pc]
            self.tableau[pr] /= pe
            for i in range(self.m_cons):
                if i != pr:
                    self.tableau[i] -= self.tableau[i, pc] * self.tableau[pr]
            self.basis[pr] = pc
            z = self._zrow()
        self.status = "max_iterations"

    def _extract(self, z):
        for i, b in enumerate(self.basis):
            if self.col_names[b].startswith("a") and abs(self.tableau[i, self.rhs_idx]) > 1e-6:
                self.status = "infeasible"; return
        self.status = "optimal"
        sol = {}
        for j in range(self.n_vars):
            col = self.tableau[:, j]
            ones = np.where(np.abs(col - 1) < 1e-8)[0]
            zeros = np.where(np.abs(col) < 1e-8)[0]
            sol[f"x{j+1}"] = float(self.tableau[ones[0], self.rhs_idx]) \
                if len(ones) == 1 and len(zeros) == self.m_cons - 1 else 0.0
        zi = z[self.rhs_idx]
        self.solution = {"vars": sol, "obj": float(zi if self.sense == "max" else -zi)}


# ═══════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════

BG="#020408"; CARD="#080f1c"; MUT="#5a7a99"
ACC="#00d4ff"; ACC2="#ff6b00"; ACC3="#7b2fff"; GOOD="#00ff88"; GOLD="#ffd700"
KEY_COL="#ff4500"; KEY_ROW="#00ff88"

def dark_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.xaxis.label.set_color(MUT); ax.yaxis.label.set_color(MUT)
    ax.title.set_color("#c8dff0")
    for s in ax.spines.values(): s.set_edgecolor("#0e2040")

def make_visualization(c, A, b, types, sense, solution, n_vars, iterations):
    c = np.array(c, dtype=float); A = np.array(A, dtype=float); b = np.array(b, dtype=float)
    if   n_vars == 2: return _vis2(c, A, b, types, sense, solution)
    elif n_vars == 3: return _vis3(c, A, b, types, sense, solution)
    else:             return _visN(c, A, b, types, sense, solution, n_vars, iterations)

def _vis2(c, A, b, types, sense, sol):
    hi = max(float(b.max()) * 1.6, 20)
    x1 = np.linspace(0, hi, 150); x2 = np.linspace(0, hi, 150)
    X1, X2 = np.meshgrid(x1, x2)
    Z = c[0]*X1 + c[1]*X2
    feas = (X1 >= 0) & (X2 >= 0)
    for i in range(len(b)):
        lhs = A[i,0]*X1 + A[i,1]*X2
        if types[i]=="<=":  feas &= (lhs <= b[i]+1e-6)
        elif types[i]==">=": feas &= (lhs >= b[i]-1e-6)
        else:                feas &= (np.abs(lhs-b[i]) <= b[i]*0.02+0.1)
    Zm = np.where(feas, Z, np.nan)
    fig = plt.figure(figsize=(14,6), facecolor=BG); fig.patch.set_facecolor(BG)
    ax3 = fig.add_subplot(121, projection="3d"); ax3.set_facecolor(BG)
    sf = ax3.plot_surface(X1, X2, Zm, cmap="plasma", alpha=0.78, linewidth=0)
    if sol:
        ox,oy,oz = sol["vars"]["x1"],sol["vars"]["x2"],sol["obj"]
        ax3.scatter([ox],[oy],[oz],color=GOLD,s=200,zorder=10)
        ax3.text(ox,oy,oz*1.06,f"  Z*={oz:.3g}",color=GOLD,fontsize=8,fontfamily="monospace")
    ax3.set_xlabel("x₁",color=MUT); ax3.set_ylabel("x₂",color=MUT); ax3.set_zlabel("Z",color=MUT)
    ax3.tick_params(colors=MUT,labelsize=7)
    ax3.set_title("Objective Surface",color="#c8dff0",fontsize=11,pad=10)
    for p in [ax3.xaxis.pane,ax3.yaxis.pane,ax3.zaxis.pane]:
        p.fill=False; p.set_edgecolor("#0e2040")
    fig.colorbar(sf,ax=ax3,shrink=0.45,pad=0.05).ax.tick_params(colors=MUT)
    ax2 = fig.add_subplot(122); dark_ax(ax2)
    ax2.contourf(X1,X2,Z,levels=18,cmap="plasma",alpha=0.35)
    ax2.contourf(X1,X2,feas.astype(float),levels=[0.5,1.5],colors=[GOOD],alpha=0.08)
    ax2.contour(X1,X2,feas.astype(float),levels=[0.5],colors=[GOOD],linewidths=1.5)
    cls=[ACC,ACC2,ACC3,"#f87171","#fb923c"]
    for i in range(len(b)):
        if abs(A[i,1])>1e-10:
            yl=(b[i]-A[i,0]*x1)/A[i,1]; mk=(yl>=0)&(yl<=hi)
            ax2.plot(x1[mk],yl[mk],color=cls[i%len(cls)],linewidth=1.4,alpha=0.7,label=f"C{i+1}")
        elif abs(A[i,0])>1e-10:
            ax2.axvline(b[i]/A[i,0],color=cls[i%len(cls)],linewidth=1.4,alpha=0.7,label=f"C{i+1}")
    if sol:
        ax2.scatter([sol["vars"]["x1"]],[sol["vars"]["x2"]],color=GOLD,s=220,zorder=10,marker="*",
                    label=f"Optimal ({sol['vars']['x1']:.3g},{sol['vars']['x2']:.3g})")
    ax2.set_xlim(0,hi); ax2.set_ylim(0,hi)
    ax2.set_xlabel("x₁"); ax2.set_ylabel("x₂")
    ax2.set_title("Feasible Region & Constraints",color="#c8dff0",fontsize=11)
    ax2.legend(fontsize=7,facecolor=CARD,edgecolor="#0e2040",labelcolor="#c8dff0",loc="upper right")
    plt.tight_layout(pad=2); return fig

def _vis3(c, A, b, types, sense, sol):
    hi = max(float(b.max())*1.4, 20)
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, hi, (8000, 3))
    mask = np.ones(len(pts), dtype=bool)
    for i in range(len(b)):
        lhs = pts @ A[i]
        if types[i]=="<=":  mask &= (lhs <= b[i]+1e-4)
        elif types[i]==">=": mask &= (lhs >= b[i]-1e-4)
        else:                mask &= (np.abs(lhs-b[i]) <= b[i]*0.04+0.2)
    pf = pts[mask]
    fig = plt.figure(figsize=(13,6), facecolor=BG); fig.patch.set_facecolor(BG)
    ax3 = fig.add_subplot(121, projection="3d"); ax3.set_facecolor(BG)
    if len(pf):
        Zp = pf@c
        sc = ax3.scatter(pf[:,0],pf[:,1],pf[:,2],c=Zp,cmap="plasma",s=4,alpha=0.55)
        fig.colorbar(sc,ax=ax3,shrink=0.45,pad=0.05,label="Z").ax.tick_params(colors=MUT)
    if sol:
        ox=[sol["vars"].get(f"x{j+1}",0) for j in range(3)]
        ax3.scatter(*[[v] for v in ox],color=GOLD,s=200,zorder=10,marker="*")
        ax3.text(ox[0],ox[1],ox[2],f"  Z*={sol['obj']:.3g}",color=GOLD,fontsize=8,fontfamily="monospace")
    ax3.set_xlabel("x₁",color=MUT); ax3.set_ylabel("x₂",color=MUT); ax3.set_zlabel("x₃",color=MUT)
    ax3.tick_params(colors=MUT,labelsize=7)
    ax3.set_title("Feasible Points (coloured by Z)",color="#c8dff0",fontsize=10)
    for p in [ax3.xaxis.pane,ax3.yaxis.pane,ax3.zaxis.pane]:
        p.fill=False; p.set_edgecolor("#0e2040")
    ax2 = fig.add_subplot(122); dark_ax(ax2)
    if len(pf):
        Zp = pf@c
        sc2 = ax2.scatter(pf[:,0],pf[:,1],c=Zp,cmap="plasma",s=6,alpha=0.5)
        fig.colorbar(sc2,ax=ax2,shrink=0.7).ax.tick_params(colors=MUT)
    if sol:
        ax2.scatter([sol["vars"].get("x1",0)],[sol["vars"].get("x2",0)],
                    color=GOLD,s=200,zorder=10,marker="*",label=f"Z*={sol['obj']:.3g}")
        ax2.legend(fontsize=8,facecolor=CARD,edgecolor="#0e2040",labelcolor="#c8dff0")
    ax2.set_xlabel("x₁"); ax2.set_ylabel("x₂")
    ax2.set_title("x₁–x₂ Projection",color="#c8dff0",fontsize=10)
    plt.tight_layout(pad=2); return fig

def _visN(c, A, b, types, sense, sol, n_vars, iterations):
    fig = plt.figure(figsize=(14,10), facecolor=BG); fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    vlabels = [f"x{j+1}" for j in range(n_vars)]
    ax1 = fig.add_subplot(gs[0,0]); dark_ax(ax1)
    if sol:
        vals = [sol["vars"].get(f"x{j+1}",0) for j in range(n_vars)]
        clrs = [ACC if v>1e-8 else MUT for v in vals]
        bars = ax1.bar(vlabels, vals, color=clrs, edgecolor="#0e2040", linewidth=0.6)
        mx = max(vals) if max(vals)>0 else 1
        for bar,val in zip(bars,vals):
            if val>1e-8:
                ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mx*0.02,
                         f"{val:.3g}", ha="center", va="bottom",
                         color=GOLD, fontsize=8, fontfamily="monospace")
    ax1.set_title("Optimal Variable Values", fontsize=10)
    ax1.set_ylabel("Value")
    ax2 = fig.add_subplot(gs[0,1]); dark_ax(ax2)
    if iterations:
        zvals = []
        for it in iterations:
            try:
                zv = float(it["df"].loc["z_j-c_j","Solution"])
                zvals.append(zv if sense=="max" else -zv)
            except: zvals.append(0)
        iters = list(range(1, len(zvals)+1))
        ax2.plot(iters, zvals, color=ACC, linewidth=2, marker="o",
                 markersize=6, markerfacecolor=GOLD, markeredgecolor=BG)
        mn = min(zvals); rng2 = max(zvals)-mn if max(zvals)!=mn else 1
        ax2.fill_between(iters, zvals, mn-rng2*0.05, alpha=0.12, color=ACC)
        ax2.set_xlabel("Iteration"); ax2.set_ylabel("Z value")
        ax2.set_title("Z Convergence Across Iterations", fontsize=10)
        ax2.set_xticks(iters)
    ax3 = fig.add_subplot(gs[1,0]); dark_ax(ax3)
    bars3 = ax3.bar(vlabels, c, color=ACC3, edgecolor="#0e2040", linewidth=0.6)
    rng3 = max(c)-min(c) if max(c)!=min(c) else 1
    for bar,val in zip(bars3,c):
        ax3.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+rng3*0.02 if val>=0 else bar.get_height()-rng3*0.06,
                 f"{val:.3g}", ha="center", va="bottom",
                 color=ACC3, fontsize=8, fontfamily="monospace")
    ax3.axhline(0, color=MUT, linewidth=0.6, linestyle="--")
    ax3.set_title("Objective Coefficients (cⱼ)", fontsize=10)
    ax3.set_ylabel("cⱼ")
    ax4 = fig.add_subplot(gs[1,1]); ax4.set_facecolor(CARD)
    ax4.tick_params(colors=MUT, labelsize=8)
    ax4.set_title("Constraint Satisfaction at Optimum", fontsize=10, color="#c8dff0")
    if sol:
        xopt = np.array([sol["vars"].get(f"x{j+1}",0) for j in range(n_vars)])
        lhsv = A @ xopt
        nc = len(b)
        heat = np.column_stack([lhsv, b])
        rlabels = [f"C{i+1} ({types[i]} {b[i]:.3g})" for i in range(nc)]
        im = ax4.imshow(heat, cmap="YlGnBu", aspect="auto")
        ax4.set_xticks([0,1]); ax4.set_xticklabels(["LHS value","RHS bound"],color=MUT,fontsize=8)
        ax4.set_yticks(range(nc)); ax4.set_yticklabels(rlabels,color=MUT,fontsize=7)
        for i in range(nc):
            for j in range(2):
                ax4.text(j,i,f"{heat[i,j]:.3g}",ha="center",va="center",
                         color="#020408",fontsize=8,fontweight="bold")
        fig.colorbar(im,ax=ax4,shrink=0.7).ax.tick_params(colors=MUT)
    fig.suptitle(f"LPP Analysis Dashboard  —  {n_vars} Variables, {len(b)} Constraints",
                 color="#c8dff0", fontsize=13, y=1.01)
    return fig


# ═══════════════════════════════════════════
# ENHANCED TABLEAU RENDERER  — BUG FIXED
# ═══════════════════════════════════════════

def render_tableau(df, pc_name=None, pr_name=None, pr_idx=None, pc_idx=None, pivot_val=None):
    cols = list(df.columns)

    html = '''<div style="overflow-x:auto;margin:.6rem 0;border-radius:4px;overflow:hidden;border:1px solid #0e2040;box-shadow:0 0 30px rgba(0,212,255,0.05);">
<table style="border-collapse:collapse;font-family:'Space Mono',monospace;font-size:.78rem;width:100%;color:#c8dff0;min-width:500px;">'''

    # Header
    html += '<thead><tr style="background:#06091a;border-bottom:2px solid #0e2040;">'
    html += '<th style="padding:.6rem 1rem;text-align:left;color:#5a7a99;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;font-weight:500;white-space:nowrap;">Basis</th>'

    for col in cols:
        is_key_col = (col == pc_name) and pc_name is not None
        is_sol = (col == "Solution")
        is_rhs = (col == "RHS")

        if is_key_col:
            hs = f"color:#ff4500;background:rgba(255,69,0,0.12);font-weight:700;border-bottom:2px solid #ff4500;box-shadow:0 0 10px rgba(255,69,0,0.2);"
            indicator = " ▼"
        elif is_sol:
            hs = f"color:#00ff88;background:rgba(0,255,136,0.06);font-weight:700;"
            indicator = ""
        elif is_rhs:
            hs = "color:#c8dff0;font-weight:600;"
            indicator = ""
        else:
            hs = f"color:#00d4ff;"
            indicator = ""
        html += f'<th style="padding:.6rem .9rem;text-align:center;{hs}white-space:nowrap;">{col}{indicator}</th>'
    html += "</tr></thead><tbody>"

    for ri, (idx, row) in enumerate(df.iterrows()):
        is_z = str(idx).startswith("z")
        is_key_row = (str(idx) == pr_name) and pr_name is not None

        # ── BUG FIX: always initialise row_label_color before use ──
        if is_z:
            row_bg = "#04081a"
            row_label_color = "#7b2fff"
        elif is_key_row:
            row_bg = "rgba(0,255,136,0.06)"
            row_label_color = "#00ff88"
        elif ri % 2 == 0:
            row_bg = "#080f1c"
            row_label_color = "#c8dff0"
        else:
            row_bg = "#060c18"
            row_label_color = "#c8dff0"

        html += f'<tr style="background:{row_bg};border-bottom:1px solid #0e2040;">'

        border_left = f"border-left:3px solid #00ff88;box-shadow:inset 3px 0 10px rgba(0,255,136,0.1);" if is_key_row else ""
        html += f'<td style="padding:.5rem 1rem;color:{row_label_color};font-weight:600;{border_left}white-space:nowrap;">'
        if is_key_row:
            html += f'<span style="color:#00ff88;margin-right:.4rem;">►</span>{idx}'
        else:
            html += str(idx)
        html += '</td>'

        for ci, col in enumerate(cols):
            is_key_col_cell = (col == pc_name) and pc_name is not None
            is_pivot = is_key_row and is_key_col_cell
            is_sol_col = (col == "Solution")

            v = row[col]
            try:
                n = float(v)
                if abs(n) > 1e5:
                    txt = f"{'−' if n < 0 else ''}M"
                    cs_val = "color:#7b2fff;font-style:italic;"
                elif abs(n) < 1e-8:
                    txt = "0"
                    cs_val = "color:#1a2a40;"
                else:
                    txt = f"{n:.4f}"
                    cs_val = ""
            except:
                txt = str(v)
                cs_val = ""

            if is_pivot:
                cell_bg = "background:rgba(255,0,128,0.2);border:1px solid #ff0080;"
                cell_color = "color:#ff0080;font-weight:800;"
                extra = f'title="PIVOT = {txt}"'
                txt = f"★ {txt}"
            elif is_key_row and not is_sol_col:
                cell_bg = "background:rgba(0,255,136,0.07);"
                cell_color = "color:#00ff88;font-weight:600;"
                extra = ""
            elif is_key_col_cell and not is_z:
                cell_bg = "background:rgba(255,69,0,0.08);"
                cell_color = "color:#ff4500;font-weight:600;"
                extra = ""
            elif is_key_col_cell and is_z:
                cell_bg = "background:rgba(255,69,0,0.04);"
                cell_color = "color:rgba(255,69,0,0.6);"
                extra = ""
            elif is_sol_col:
                if is_z:
                    cell_bg = ""
                    cell_color = "color:#ffd700;font-weight:700;text-shadow:0 0 8px rgba(255,215,0,0.4);"
                else:
                    cell_bg = "background:rgba(0,255,136,0.04);"
                    cell_color = "color:#00ff88;font-weight:700;"
                extra = ""
            elif col == "RHS":
                cell_bg = ""
                cell_color = "color:#c8dff0;"
                extra = ""
            elif is_z:
                cell_bg = ""
                cell_color = cs_val or "color:#c8dff0;"
                extra = ""
            else:
                cell_bg = ""
                cell_color = cs_val or "color:#c8dff0;"
                extra = ""

            html += f'<td {extra} style="padding:.5rem .9rem;text-align:center;{cell_bg}{cell_color}white-space:nowrap;">{txt}</td>'
        html += "</tr>"

    html += "</tbody></table></div>"
    return html


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

# Default problem from the image
DEFAULT_PROBLEM = {
    "n_vars": 5,
    "n_cons": 5,
    "sense": "Maximisation (Z_max)",
    "obj": [4.0, 3.0, 6.0, 2.0, 5.0],
    "A": [
        [1, 2, 1, 1, 1],
        [2, 1, 3, 1, 2],
        [1, 3, 2, 2, 1],
        [3, 1, 2, 1, 4],
        [2, 4, 1, 3, 2],
    ],
    "b": [20.0, 25.0, 30.0, 40.0, 35.0],
    "types": ["<=", ">=", "=", "<=", ">="],
}

def main():
    # Header
    st.markdown('''
    <div class="hdr">
      <div class="corner-tl"></div>
      <div class="corner-tr"></div>
      <div class="corner-bl"></div>
      <div class="corner-br"></div>
      <div class="hdr-badge">⬡ Operations Research · Linear Programming</div>
      <h1>Big <span class="accent">M</span> Method <span class="accent2">Solver</span></h1>
      <div class="hdr-sub">
        <span>//</span> Simplex Algorithm &nbsp;·&nbsp; Step-by-Step Tableaux &nbsp;·&nbsp; Interactive Visualization <span>//</span>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    # ── Preset problem ──
    st.markdown('<div class="sec"><div class="sec-num">0</div><div class="sec-title">Quick Load</div><div class="sec-line"></div></div>', unsafe_allow_html=True)

    use_preset = st.checkbox("📋 Load preset problem from image (5 variables, 5 constraints)", value=True)

    if use_preset:
        st.markdown('''
        <div class="preset-box">
          <div class="preset-eq">Z = 4x₁ + 3x₂ + 6x₃ + 2x₄ + 5x₅ &nbsp; → &nbsp; Maximize</div>
          <div class="preset-constraint">C1: &nbsp; x₁ + 2x₂ +  x₃ +  x₄ +  x₅  ≤  20</div>
          <div class="preset-constraint">C2: 2x₁ +  x₂ + 3x₃ +  x₄ + 2x₅  ≥  25</div>
          <div class="preset-constraint">C3: &nbsp; x₁ + 3x₂ + 2x₃ + 2x₄ +  x₅  =  30</div>
          <div class="preset-constraint">C4: 3x₁ +  x₂ + 2x₃ +  x₄ + 4x₅  ≤  40</div>
          <div class="preset-constraint">C5: 2x₁ + 4x₂ +  x₃ + 3x₄ + 2x₅  ≥  35</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Section 1: Problem Setup ──
    st.markdown('<div class="sec"><div class="sec-num">1</div><div class="sec-title">Problem Setup</div><div class="sec-line"></div></div>', unsafe_allow_html=True)

    def_nv = DEFAULT_PROBLEM["n_vars"] if use_preset else 2
    def_nc = DEFAULT_PROBLEM["n_cons"] if use_preset else 2
    def_sense = DEFAULT_PROBLEM["sense"] if use_preset else "Maximisation (Z_max)"

    ca, cb, cc = st.columns(3)
    with ca: n_vars = st.number_input("Decision variables", min_value=1, max_value=10, value=def_nv, step=1)
    with cb: n_cons = st.number_input("Constraints",        min_value=1, max_value=10, value=def_nc, step=1)
    with cc: sense  = st.selectbox("Objective", ["Maximisation (Z_max)", "Minimisation (Z_min)"],
                                   index=0 if def_sense == "Maximisation (Z_max)" else 1)
    sense_key = "max" if "Max" in sense else "min"

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Section 2: Objective ──
    st.markdown('<div class="sec"><div class="sec-num">2</div><div class="sec-title">Objective Function</div><div class="sec-line"></div></div>', unsafe_allow_html=True)

    nv = int(n_vars)
    nc = int(n_cons)

    oc = st.columns(nv); obj = []
    for j in range(nv):
        with oc[j]:
            def_val = DEFAULT_PROBLEM["obj"][j] if (use_preset and j < len(DEFAULT_PROBLEM["obj"])) else 1.0
            obj.append(st.number_input(f"c{j+1}  (x{j+1})", value=def_val, key=f"o{j}", format="%.2f"))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Section 3: Constraints ──
    st.markdown('<div class="sec"><div class="sec-num">3</div><div class="sec-title">Constraints</div><div class="sec-line"></div></div>', unsafe_allow_html=True)

    A = []; b = []; ctypes = []
    type_opts = ["<=", ">=", "="]

    for i in range(nc):
        st.markdown(f'<div class="con-label">Constraint {i+1}</div>', unsafe_allow_html=True)
        rc = st.columns([1]*nv + [1, 1]); row = []
        for j in range(nv):
            with rc[j]:
                def_a = DEFAULT_PROBLEM["A"][i][j] if (use_preset and i < len(DEFAULT_PROBLEM["A"]) and j < nv) else 1.0
                row.append(st.number_input(f"a{i+1}{j+1}", value=float(def_a), key=f"a{i}{j}",
                                           format="%.2f", label_visibility="collapsed"))
        A.append(row)
        def_t = DEFAULT_PROBLEM["types"][i] if (use_preset and i < len(DEFAULT_PROBLEM["types"])) else "<="
        def_b = DEFAULT_PROBLEM["b"][i] if (use_preset and i < len(DEFAULT_PROBLEM["b"])) else 4.0
        with rc[nv]:
            ctypes.append(st.selectbox("", type_opts, index=type_opts.index(def_t), key=f"t{i}"))
        with rc[nv+1]:
            b.append(st.number_input("RHS", value=float(def_b), key=f"b{i}", format="%.2f", label_visibility="collapsed"))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    sc2, _ = st.columns([1, 4])
    with sc2: go = st.button("⚙  SOLVE")

    if not go:
        st.markdown('''
        <div class="card" style="text-align:center;padding:3rem;margin-top:1rem;">
          <div style="font-family:Orbitron,sans-serif;font-size:2rem;color:#0e2040;margin-bottom:1rem;">◈</div>
          <div style="font-family:Space Mono,monospace;font-size:.78rem;color:#5a7a99;letter-spacing:.08em;">
            Configure your LP problem above and press<br>
            <span style="color:#00d4ff;">⚙ SOLVE</span> to run the Big M simplex algorithm.
          </div>
        </div>
        ''', unsafe_allow_html=True)
        return

    errs = [f"RHS of constraint {i+1} is negative ({rv:.2f}). Multiply by −1 and flip the sign."
            for i, rv in enumerate(b) if rv < 0]
    if errs:
        for e in errs:
            st.markdown(f'<div class="errbox">⚠ {e}</div>', unsafe_allow_html=True)
        return

    solver = BigMSolver(obj, A, b, ctypes, sense_key)
    solver.solve()

    # ── Stats ──
    n_iters = len(solver.iterations)
    status_color = "#00ff88" if solver.status == "optimal" else "#ff3344"
    st.markdown(f'''
    <div class="stats-row">
      <div class="stat-card"><div class="stat-val">{nv}</div><div class="stat-lbl">Decision Vars</div></div>
      <div class="stat-card"><div class="stat-val">{nc}</div><div class="stat-lbl">Constraints</div></div>
      <div class="stat-card"><div class="stat-val">{n_iters}</div><div class="stat-lbl">Iterations</div></div>
      <div class="stat-card"><div class="stat-val" style="color:{status_color};font-size:1.1rem;">{solver.status.upper()}</div><div class="stat-lbl">Status</div></div>
    </div>
    ''', unsafe_allow_html=True)

    # ── Section 4: Iterations ──
    st.markdown('<div class="sec"><div class="sec-num">4</div><div class="sec-title">Simplex Iterations</div><div class="sec-line"></div></div>', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#ff4500;box-shadow:0 0 6px #ff4500;"></div>Key Column — Entering Variable</div>
      <div class="legend-item"><div class="legend-dot" style="background:#00ff88;box-shadow:0 0 6px #00ff88;"></div>Key Row — Leaving Variable</div>
      <div class="legend-item"><div class="legend-dot" style="background:#ff0080;box-shadow:0 0 6px #ff0080;"></div>Pivot Element ★</div>
      <div class="legend-item"><div class="legend-dot" style="background:#00ff88;opacity:.5;"></div>Solution Values</div>
      <div class="legend-item"><div class="legend-dot" style="background:#7b2fff;box-shadow:0 0 6px #7b2fff;"></div>z_j − c_j Row</div>
    </div>
    ''', unsafe_allow_html=True)

    with st.expander("▾ Show / Hide All Tableaux", expanded=True):
        for it in solver.iterations:
            st.markdown(f'''
            <div class="iter-hdr">
              <div class="iter-num">Iteration {it["iter"]}</div>
              <div class="iter-line"></div>
            </div>
            ''', unsafe_allow_html=True)

            if it["pc"] and it["pr"]:
                entering = it["pc"]
                leaving  = it["pr"]
                pivot_v  = it["pivot_val"]
                pv_str   = f"{pivot_v:.4f}" if pivot_v is not None else "—"

                def var_type(name):
                    if name.startswith("x"):   return "decision var"
                    elif name.startswith("s"): return "slack/surplus"
                    elif name.startswith("a"): return "artificial"
                    return "variable"

                e_type = var_type(entering)
                l_type = var_type(leaving)

                st.markdown(f'''
                <div class="pivot-banner">
                  <div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;">
                    <div class="pivot-chip chip-enter">↑ {entering} enters</div>
                    <div class="arrow-sym">⇄</div>
                    <div class="pivot-chip chip-leave">↓ {leaving} leaves</div>
                    <div class="pivot-chip chip-pivot-val">✦ pivot = {pv_str}</div>
                  </div>
                  <div class="pivot-explain">
                    <strong style="color:#c8dff0;">{entering}</strong> ({e_type}) enters — most negative z_j−c_j.<br>
                    <strong style="color:#c8dff0;">{leaving}</strong> ({l_type}) leaves — minimum ratio test.<br>
                    Pivot element <strong style="color:#ff0080;">{pv_str}</strong> is at their intersection (★).
                  </div>
                </div>
                ''', unsafe_allow_html=True)
            elif it["iter"] == solver.iterations[-1]["iter"]:
                st.markdown(f'''
                <div class="pivot-banner">
                  <div class="pivot-chip chip-optimal">✓ All z_j−c_j ≥ 0 — Optimality reached</div>
                  <div class="pivot-explain" style="color:#00ff88;opacity:.7;">
                    No negative reduced costs remain. Current basis is optimal.
                  </div>
                </div>
                ''', unsafe_allow_html=True)

            st.markdown(
                render_tableau(
                    it["df"],
                    pc_name=it["pc"],
                    pr_name=it["pr"],
                    pr_idx=it.get("pr_idx"),
                    pc_idx=it.get("pc_idx"),
                    pivot_val=it.get("pivot_val")
                ),
                unsafe_allow_html=True
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Section 5: Result ──
    st.markdown('<div class="sec"><div class="sec-num">5</div><div class="sec-title">Result</div><div class="sec-line"></div></div>', unsafe_allow_html=True)

    if solver.status == "optimal":
        sol = solver.solution
        vhtml = "".join(
            f'<div class="resvar"><span class="resvar-name">{k}</span><span class="resvar-eq">=</span><span class="resvar-val">{v:.6g}</span></div>'
            for k, v in sol["vars"].items()
        )
        zl = "Z<sub>max</sub>" if sense_key == "max" else "Z<sub>min</sub>"
        st.markdown(f'''
        <div class="resbox">
          <h3>✓ OPTIMAL SOLUTION FOUND</h3>
          <div style="margin:.8rem 0;">{vhtml}</div>
          <div class="resobj">{zl} = {sol["obj"]:.6g} <span>optimal value</span></div>
        </div>
        ''', unsafe_allow_html=True)
    elif solver.status == "unbounded":
        st.markdown('<div class="errbox"><strong>⚠ UNBOUNDED SOLUTION</strong><br>The objective can grow indefinitely. Check your constraints.</div>', unsafe_allow_html=True)
    elif solver.status == "infeasible":
        st.markdown('<div class="errbox"><strong>⚠ INFEASIBLE PROBLEM</strong><br>No feasible solution — constraints are contradictory.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="errbox">Status: {solver.status}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Section 6: Visualization ──
    labels = {
        2: "3-D Objective Surface + 2-D Feasible Region",
        3: "3-D Feasible Point Cloud + x₁–x₂ Projection"
    }
    subtitle = labels.get(nv, f"4-Panel Analysis Dashboard ({nv} variables)")

    st.markdown('<div class="sec"><div class="sec-num">6</div><div class="sec-title">Visualization</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="vis-label">{subtitle}</div>', unsafe_allow_html=True)

    try:
        fig = make_visualization(
            obj, A, b, ctypes, sense_key,
            solver.solution if solver.status == "optimal" else None,
            nv, solver.iterations
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as ex:
        st.warning(f"Visualization error: {ex}")


if __name__ == "__main__":
    main()
