"""P3-G7: Top-m G7 endings + one high-probability 3-digit prefix.

For each day and m=1..10, choose the m most frequent G7 two-digit endings
from history, then attach the same top digit at hundred/thousand/ten-thousand
positions from the full 27-result pool.  The final period is never used to
choose the configuration.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/raw/kqxsmb_all_prizes_2007_2026.csv'
OUT=ROOT/'artifacts/p3_strategies/g7_topm_prefix'
PAYOUT={"Đặc biệt":25_000_000,"Giải nhất":10_000_000,"Giải nhì":5_000_000,"Giải ba":1_000_000,"Giải tư":400_000,"Giải năm":200_000,"Giải sáu":100_000,"Giải bảy":40_000}
LENGTH={"Đặc biệt":5,"Giải nhất":5,"Giải nhì":5,"Giải ba":5,"Giải tư":4,"Giải năm":4,"Giải sáu":3,"Giải bảy":2}
COST=10_000
POS=("ten_thousands","thousands","hundreds")
FOLDS={"validation_2023_2024":("2023-01-01","2024-12-31"),"final_test_2025_2026":("2025-01-01","2026-12-31")}

def load():
    raw=pd.read_csv(RAW,dtype={"number":str},encoding='utf-8-sig'); raw.date=pd.to_datetime(raw.date); raw.number=raw.number.str.strip()
    target=pd.read_csv(ROOT/'data/processed/daily_digit_targets.csv',parse_dates=['date']).sort_values('date')
    return raw,target

def prefix(target_hist, date):
    h=target_hist[target_hist.date<date].tail(365)
    return ''.join(str(int(np.argmax([(h[f'{p}_d{d}'].sum()+1)/(len(h)+2) for d in range(10)]))) for p in POS)

def endings(raw_hist, date):
    h=raw_hist[(raw_hist.date<date)&raw_hist.prize.eq('Giải bảy')]
    counts=h.number.str.zfill(2).str[-2:].value_counts()
    return sorted([f'{i:02d}' for i in range(100)], key=lambda x:(-counts.get(x,0),x))

def payout(ticket,day):
    total=0; prizes=[]
    for _,r in day.iterrows():
        n=str(r.number).zfill(LENGTH[r.prize]); length=LENGTH[r.prize]
        if ticket[-length:]==n[-length:]:
            total+=PAYOUT[r.prize]; prizes.append(r.prize)
    return total,prizes

def run():
    raw,target=load(); rows=[]
    for fold,(start,end) in FOLDS.items():
        dates=target.loc[target.date.between(start,end),'date'].drop_duplicates().sort_values()
        for date in dates:
            hist=raw[raw.date<date]; ph=prefix(target,date); ranked=endings(raw,date); day=raw[raw.date.eq(date)]
            for m in range(1,11):
                tickets=[ph+x for x in ranked[:m]]; pay=0; prize_count=0; hit_g7=0; hit_any=0
                for ticket in tickets:
                    amount,prizes=payout(ticket,day); pay+=amount; prize_count+=len(prizes); hit_g7+=int('Giải bảy' in prizes); hit_any+=int(bool(prizes))
                cost=m*COST; rows.append({'date':date,'fold':fold,'m':m,'prefix':ph,'tickets':' '.join(tickets),'hit_g7_tickets':hit_g7,'hit_any_tickets':hit_any,'prize_count':prize_count,'payout':pay,'cost':cost,'profit':pay-cost,'roi':(pay-cost)/cost})
    result=pd.DataFrame(rows); result['cum_profit']=result.groupby(['fold','m']).profit.cumsum(); OUT.mkdir(parents=True,exist_ok=True)
    result.to_csv(OUT/'daily_results.csv',index=False)
    summary=result.groupby(['fold','m'],as_index=False).agg(days=('date','nunique'),hit_g7_tickets=('hit_g7_tickets','sum'),hit_any_tickets=('hit_any_tickets','sum'),prize_count=('prize_count','sum'),payout=('payout','sum'),cost=('cost','sum'),profit=('profit','sum'),roi=('profit',lambda x:x.sum()/result.loc[x.index,'cost'].sum()))
    summary['g7_hit_rate']=summary.hit_g7_tickets/summary.days; summary['any_hit_rate']=summary.hit_any_tickets/summary.days; summary.to_csv(OUT/'summary.csv',index=False)
    for fold in FOLDS:
        q=summary[summary.fold.eq(fold)]; plt.figure(figsize=(10,5)); plt.plot(q.m,q.profit,marker='o'); plt.axhline(0,color='black',lw=.8); plt.title(f'G7 Top-m + fixed prefix — {fold}'); plt.xlabel('m vé/ngày'); plt.ylabel('Total profit (VND)'); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(OUT/f'profit_by_m_{fold}.png',dpi=180); plt.close()
        q=result[result.fold.eq(fold)].groupby(['date','m']).profit.sum().unstack(); plt.figure(figsize=(11,5)); plt.plot(q.cumsum()); plt.axhline(0,color='black',lw=.8); plt.title(f'Cumulative profit G7 Top-m — {fold}'); plt.xlabel('day'); plt.ylabel('VND'); plt.legend(title='m',ncol=2); plt.tight_layout(); plt.savefig(OUT/f'cumulative_profit_{fold}.png',dpi=180); plt.close()
    print(summary.to_string(index=False))
if __name__=='__main__': run()
