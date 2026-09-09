"""Part 3 backtest: position-wise combinations and exact prize rules.

Validation (2023-2024) is used for strategy selection; final (2025-2026) is
reported only as a frozen out-of-sample result.  Every ticket is a 5-digit
number and is checked against all 27 full-prize rows of the day.
"""
from __future__ import annotations

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/kqxsmb_all_prizes_2007_2026.csv"
TARGETS = ROOT / "data/processed/daily_digit_targets.csv"
OUT = ROOT / "artifacts/p3_strategies"
POS = ("ten_thousands", "thousands", "hundreds", "tens", "units")
FOLDS = {"validation_2023_2024": ("2023-01-01", "2024-12-31"), "final_test_2025_2026": ("2025-01-01", "2026-12-31")}
PAYOUT = {"Đặc biệt":25_000_000,"Giải nhất":10_000_000,"Giải nhì":5_000_000,"Giải ba":1_000_000,"Giải tư":400_000,"Giải năm":200_000,"Giải sáu":100_000,"Giải bảy":40_000}
COST = 10_000
LAYOUTS = {"2x2x2x2x2":(2,2,2,2,2), "3x3x2x1x1":(3,3,2,1,1), "3x2x2x2x1":(3,2,2,2,1)}

def load():
    raw=pd.read_csv(RAW,dtype={"number":str}); raw.date=pd.to_datetime(raw.date); raw.number=raw.number.str.strip()
    targets=pd.read_csv(TARGETS,parse_dates=["date"]).sort_values("date")
    return raw,targets

def probabilities(hist, model="rolling365"):
    out=[]
    for p in POS:
        cols=[f"{p}_d{i}" for i in range(10)]
        h=hist.tail(365) if model=="rolling365" else hist
        out.append(((h[cols].sum()+1)/(len(h)+2)).to_numpy())
    return np.asarray(out)

def tickets(p, layout):
    choices=[list(np.argsort(-p[i])[:layout[i]]) for i in range(5)]
    return ["".join(map(str,x)) for x in itertools.product(*choices)]

def score_ticket(ticket, day):
    hits=[]
    lengths={"Đặc biệt":5,"Giải nhất":5,"Giải nhì":5,"Giải ba":5,"Giải tư":4,"Giải năm":4,"Giải sáu":3,"Giải bảy":2}
    for _,r in day.iterrows():
        length=lengths[r.prize]; n=str(r.number).zfill(length)
        if ticket[-length:]==n[-length:]: hits.append((r.prize,PAYOUT[r.prize]))
    return hits

def run():
    raw,targets=load(); rows=[]
    for fold,(start,end) in FOLDS.items():
        dates=pd.date_range(start,end)
        for _,row in targets[targets.date.isin(dates)].iterrows():
            hist=targets[targets.date<row.date]
            p=probabilities(hist)
            for name,layout in LAYOUTS.items():
                selected=tickets(p,layout); total=0; hit_count=0; prize_count=0
                day=raw[raw.date.eq(row.date)]
                for ticket in selected:
                    hits=score_ticket(ticket,day); hit_count += bool(hits); prize_count += len(hits); total += sum(v for _,v in hits)
                cost=len(selected)*COST; rows.append({"date":row.date,"fold":fold,"strategy":name,"n_tickets":len(selected),"hit_days":int(hit_count>0),"prize_count":prize_count,"payout":total,"cost":cost,"profit":total-cost,"roi":(total-cost)/cost})
    result=pd.DataFrame(rows).sort_values(["fold","date","strategy"]); result["cum_profit"]=result.groupby("strategy").profit.cumsum()
    OUT.mkdir(parents=True,exist_ok=True); result.to_csv(OUT/"strategy_daily_results.csv",index=False)
    summary=result.groupby(["fold","strategy"],as_index=False).agg(days=("date","nunique"),total_tickets=("n_tickets","sum"),hit_days=("hit_days","sum"),prize_count=("prize_count","sum"),payout=("payout","sum"),cost=("cost","sum"),profit=("profit","sum"),roi=("profit",lambda x:x.sum()/result.loc[x.index,"cost"].sum()))
    summary["hit_rate"]=summary.hit_days/summary.days; summary.to_csv(OUT/"strategy_summary.csv",index=False)
    for fold in FOLDS:
        q=result[result.fold.eq(fold)]
        plt.figure(figsize=(11,5))
        for s,g in q.groupby("strategy"): plt.plot(g.groupby("date").profit.sum().cumsum().to_numpy(),label=s)
        plt.axhline(0,color="black",lw=.8); plt.title(f"P3 cumulative profit — {fold}"); plt.ylabel("VND"); plt.xlabel("day"); plt.legend(); plt.tight_layout(); plt.savefig(OUT/f"cumulative_profit_{fold}.png",dpi=180); plt.close()
    summary.to_csv(OUT/"strategy_summary.csv",index=False); print(summary.to_string(index=False))
if __name__=="__main__": run()
