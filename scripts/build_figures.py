"""Render Nature figures from the accompanying frozen aggregate source data."""
from pathlib import Path
import csv
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch

PKG = Path(__file__).resolve().parents[1]
SRC, OUT = PKG / 'source_data', PKG / 'figures'
OUT.mkdir(exist_ok=True)
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--only', help='Render only the named figure stem')
_args = _parser.parse_args()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
    'axes.titlesize':10,'axes.titleweight':'bold','axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'axes.linewidth':0.7,'xtick.labelsize':8,'ytick.labelsize':9})
GREEN, BLUE, GREY, ORANGE = '#087F6D','#3568A0','#626C77','#B77529'

def save(fig, name):
    if _args.only and name != _args.only:
        plt.close(fig)
        return
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=350, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def style(ax, title, letter):
    ax.set_title(title,loc='left',pad=15)
    ax.text(-0.04,1.15,letter,transform=ax.transAxes,fontweight='bold',fontsize=12)
    ax.grid(axis='x',color='#E5E9EC',lw=.6,zorder=0)
    ax.tick_params(axis='y',length=0)
    ax.spines['left'].set_visible(False)

def forest(ax, labels, estimates, colors, xlim, xlabel):
    for i, ((point, lo, hi), color) in enumerate(zip(estimates, colors)):
        ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',color=color,
                    capsize=3,ms=5,elinewidth=1.4,zorder=3)
    ax.set_yticks(range(len(labels)),labels)
    ax.set_ylim(len(labels)-.5,-.6)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel,labelpad=8)

# Figure 1: explicitly conceptual, constructed as vector geometry.
fig = plt.figure(figsize=(7.1,5.0))
ax = fig.add_axes([0.03,.52,.94,.43]); ax.set(xlim=(0,10),ylim=(0,4)); ax.axis('off')
ax.text(0,3.8,'a',fontweight='bold',fontsize=12)
ax.text(.45,3.8,'A fixed coordinate need not follow moving anatomy',fontweight='bold',fontsize=10)
for cx, label, moving in [(2.1,'End-diastole',False),(6.5,'End-systole',True)]:
    ex, ey = (cx+.45,1.65) if moving else (cx,1.75)
    ax.add_patch(Ellipse((ex,ey),2.6 if moving else 3.0,1.5 if moving else 2,
                        angle=-20 if moving else 0,fc='#EAF3F6',ec=BLUE,lw=1.5))
    ax.plot([cx-1.55,cx+1.55],[1.75,1.75],color=GREY,lw=1.5,ls='--')
    if moving:
        ax.plot([ex-1.13,ex+1.13],[ey+.41,ey-.41],color=GREEN,lw=2.1)
    else:
        ax.plot([cx-1.3,cx+1.3],[1.75,1.75],color=GREEN,lw=2.1)
    ax.text(cx,3.07,label,ha='center',fontsize=9)
ax.add_patch(FancyArrowPatch((3.9,1.85),(4.8,1.85),arrowstyle='->',mutation_scale=12,color=GREY))
ax.text(4.35,2.35,'Motion',ha='center',fontsize=8,color=GREY)
ax.plot([.7,1.3],[.15,.15],color=GREY,ls='--'); ax.text(1.4,.15,'ED-fixed axis',va='center',fontsize=8)
ax.plot([4.9,5.5],[.15,.15],color=GREEN,lw=2); ax.text(5.6,.15,'Relocated line',va='center',fontsize=8)
ax = fig.add_axes([.03,.02,.94,.44]); ax.set(xlim=(0,10),ylim=(0,4)); ax.axis('off')
ax.text(0,3.8,'b',fontweight='bold',fontsize=12)
ax.text(.45,3.8,'Complementary tests beyond endpoint supervision',fontweight='bold',fontsize=10)
cards = [(.25,'Sampling principle','Manual CMR comparison\nSame cine, readers and ES\nFunctional proxy association'),
         (3.55,'Ultrasound geometry','Two AI implementations\nAxes at the same frame\nAnatomical accuracy unproven'),
         (6.85,'Clinical association','AI and recorded LVEF\nIndependent biomarker\nPaired differences uncertain')]
for x,title,body in cards:
    ax.add_patch(FancyBboxPatch((x,.8),2.9,2.35,boxstyle='round,pad=0.1',fc='#F4F7F9',ec='#C9D3DA',lw=.8))
    ax.text(x+1.45,2.65,title,ha='center',fontweight='bold',fontsize=9)
    ax.text(x+1.45,1.65,body,ha='center',va='center',linespacing=1.65,fontsize=8)
ax.text(5,.17,'Functional information and absolute agreement are separate outcomes',ha='center',fontsize=9,color=GREEN)
save(fig,'figure1_measurement_framework')

rows=list(csv.DictReader((SRC/'three_reader_cmr_agreement_results.csv').open()))
raw={r['method']:r for r in rows if r['stage']=='raw'}
contr=list(csv.DictReader((SRC/'three_reader_bootstrap_contrasts.csv').open()))
contr={r['contrast']:r for r in contr if r['metric']=='delta_pearson_r'}
fig,(a,b)=plt.subplots(2,1,figsize=(7.1,4.7),gridspec_kw={'height_ratios':[1.2,1]},layout='constrained')
forest(a,['Static','Peak-to-peak','Dynamic'],
    [tuple(float(raw[k][v]) for v in ['pearson_r','pearson_ci_low','pearson_ci_high']) for k in ['static','peak_to_peak','dynamic']],
    [GREY,ORANGE,GREEN],(-.2,.65),'Pearson correlation with short-axis functional proxy')
style(a,'Raw association  |  152 examinations','a'); a.axvline(0,color=GREY,ls='--',lw=.8)
forest(b,['Dynamic − static','Dynamic − peak-to-peak'],
    [tuple(float(contr[k][v]) for v in ['observed','ci_low','ci_high']) for k in ['dynamic_minus_static','dynamic_minus_peak_to_peak']],
    [GREEN,GREEN],(-.1,.65),'Paired difference in Pearson correlation')
style(b,'Paired comparisons  |  95% confidence intervals','b'); b.axvline(0,color=GREY,ls='--',lw=.8)
save(fig,'figure2_cmr_signal')

geo=json.loads((SRC/'plax_geometry_summary.json').read_text())
cut=json.loads((SRC/'static_cutin_summary.json').read_text())
fig,(a,b)=plt.subplots(2,1,figsize=(7.1,5.2),gridspec_kw={'height_ratios':[1.2,1]},layout='constrained')
forest(a,[f"{r['set']}: {'F04' if r['model']=='f04' else 'EchoNet-LVH'}" for r in geo],
    [(r['median'],r['q1'],r['q3']) for r in geo],[GREEN,BLUE,GREEN,BLUE],(0,30),
    'Perpendicular offset / ED line length (%)')
style(a,'Same-ES geometry  |  A: 91 pairs; B: 94 pairs','a')
labels=[r['label'] for r in cut['categories']]
values=[100*r['n']/cut['denominator'] for r in cut['categories']]
b.barh(range(4),values,color=[GREY,BLUE,ORANGE,GREEN],height=.58,zorder=3)
b.set_yticks(range(4),labels); b.set_ylim(3.6,-.6); b.set_xlim(0,80)
b.set_xlabel('Videos with the indicated static marker category (%)',labelpad=8)
for i,r in enumerate(cut['categories']):
    b.text(values[i]+1.2,i,f"{r['n']}/{cut['denominator']}",va='center',fontsize=9)
style(b,'Static cut-in records  |  96 ES-valid videos','b')
b.set_xticks([0,20,40,60,80])
save(fig,'figure3_plax_geometry')

c=json.loads((SRC/'clinical_association_summary.json').read_text())
fig,(a,b)=plt.subplots(2,1,figsize=(7.1,4.7),gridspec_kw={'height_ratios':[1.2,1]},layout='constrained')
order=['lvef_interval_midpoint','echonet_lvh','f04']
forest(a,['Recorded LVEF','EchoNet-LVH','F04'],
       [(c['models'][k]['partial_r']['point'],*c['models'][k]['partial_r']['ci95']) for k in order],
       [GREY,BLUE,GREEN],(-.85,.05),'Partial correlation with log10 NT-proBNP')
style(a,'Adjusted associations  |  45 examinations; 39 patients','a'); a.axvline(0,color=GREY,ls='--',lw=.8)
order=['lvef_interval_midpoint','echonet_lvh']
forest(b,['Recorded − F04','EchoNet-LVH − F04'],
       [(c['contrasts'][k]['partial_r_other_minus_f04']['point'],*c['contrasts'][k]['partial_r_other_minus_f04']['ci95']) for k in order],
       [GREEN,GREEN],(-.10,.32),'Paired difference in partial correlation')
style(b,'Paired uncertainty  |  95% confidence intervals','b'); b.axvline(0,color=GREY,ls='--',lw=.8)
save(fig,'figure4_clinical_associations')
print(f"Rendered {_args.only or 'all 4 figures'} as PDF, SVG and 350 dpi PNG")
