"""Generate revised manuscript, measured-data figures and reviewer responses.

The submitted AIH Paper.docx is preserved. The output remains a working draft
while hardware extensions requested by Reviewer 2 are outstanding.
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm,LogNorm
from matplotlib.patches import Patch,FancyBboxPatch
from docx import Document
from docx.shared import Inches,Pt
from docx.text.paragraph import Paragraph
from docx.enum.style import WD_STYLE_TYPE
from revision_analysis import ROOT,RESULTS,REPORTS,pair_lookup,MODEL_FILES

FIG=REPORTS/'figures'
CORE_MODELS=['Logistic Regression','LightGBM','DistilBERT','PubMedBERT','BERT-tiny']
COLORS=dict(zip(CORE_MODELS,['#2463aa','#19876d','#e48c22','#9455a5','#c34c58']))
COLORS.update({'DistilBERT CPU FP32':'#cf8b47','DistilBERT CPU INT8':'#996019','PubMedBERT CPU FP32':'#bb8cc6','PubMedBERT CPU INT8':'#683b84','BERT-tiny CPU FP32':'#e6959b','BERT-tiny CPU INT8':'#8f2733'})
plt.rcParams.update({'font.size':11,'axes.labelsize':11,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10})


def save_figure(fig,path):
    fig.savefig(path,dpi=300,bbox_inches='tight')
    fig.savefig(Path(path).with_suffix('.pdf'),bbox_inches='tight')


def line_chart(path,panels,title):
    fig,axes=plt.subplots(1,len(panels),figsize=(8.3,4.8),squeeze=False)
    for ax,(heading,series,xlabel,ylabel) in zip(axes[0],panels):
        for label,x,y,kwargs in series:ax.plot(x,y,label=label,**kwargs)
        ax.set(title=heading,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=9.5)
    fig.suptitle(title,fontsize=12);fig.tight_layout();save_figure(fig,path);plt.close(fig)


def diagram(path,title,columns,footer=''):
    fig,ax=plt.subplots(figsize=(8.3,5));ax.axis('off');ax.set_title(title,weight='bold',pad=15)
    width=.88/len(columns)
    for i,(heading,steps) in enumerate(columns):
        x=.06+i*width
        ax.text(x+width/2,.91,textwrap.fill(heading,20),ha='center',va='center',weight='bold',fontsize=11)
        for j,step in enumerate(steps):
            y=.76-j*.20
            ax.add_patch(FancyBboxPatch((x+.012,y-.083),width-.024,.166,boxstyle='round,pad=.008',
                         facecolor=['#edf4fc','#eef8f0','#fcf4e8'][i%3],edgecolor='#9aaaba',transform=ax.transAxes))
            ax.text(x+width/2,y,textwrap.fill(step,24),ha='center',va='center',fontsize=9.5)
            if j<len(steps)-1:ax.annotate('',(x+width/2,y-.13),(x+width/2,y-.07),arrowprops={'arrowstyle':'->','color':'#6b7c8f'})
    ax.text(.5,.02,textwrap.fill(footer,125),ha='center',va='bottom',fontsize=8,color='#495768')
    fig.tight_layout();save_figure(fig,path);plt.close(fig)


def make_figures(summary,primary):
    FIG.mkdir(parents=True,exist_ok=True)
    diagram(FIG/'figure1_pipeline.png','ECC-MS: development selection and separate external assessment',[
        ('Data and fitting',['PsyTAR: 6,003 sentences, 886 reviews','Review-disjoint train / calibration / development audit','Fit classifiers on training; recalibrators on calibration']),
        ('Primary selection',['Review-cluster calibration intervals on development audit','Filter on calibration upper bound and scoped measured energy','Compare feasible arms with paired cluster AUROC intervals']),
        ('Outputs and external assessment',['Choose lowest-energy practically equivalent feasible arm','Report infeasibility when the feasible set is empty','CADEC diagnostic only: calibration, AUROC, workload; no primary selection feedback'])],
        'CADEC-assisted selection is reported separately as a sensitivity analysis, not as independent validation of that assisted decision.')
    diagram(FIG/'figure2_groups.png','Corpus harmonisation and review-disjoint partitioning',[
        ('PsyTAR',['ADR sentence label; withdrawal kept separate','Train 3,626 sentences / 531 reviews; calibration 1,188 / 177','Audit 1,189 / 178; every pair of review sets is disjoint']),
        ('CADEC',['1,248 non-empty posts; 7,823 sentences','Rule A overlap: 2,865 positives; Rule B containment: 2,863','Six sentences touch crossing segments; exactly two labels change'])],
        'Post identifiers reconstructed from original files are checked against every submitted text and label. 7,409 span segments belong to 6,318 annotated ADR entities.')
    diagram(FIG/'figure3_energy.png','Measured energy domains and their limits',[
        ('Original Linux measurements',['Intel RAPL package-0 on i5-8500','Raw text → TF-IDF vectorisation → prediction','7 repeats; coupled median power, throughput and J/1,000']),
        ('Revision Colab measurements',['NVIDIA T4 board power only','Raw-text pipeline timing; 3 encoders × 15 batch/length settings','3 repeats; host CPU/RAM and wall energy unmeasured']),
        ('Returned Linux extension',['Three encoders in FP32 and dynamic INT8; seed-42 checkpoints','Raw-text pipeline package energy; 7 repeats per configuration','Tokenizer component repeats retained, including flagged original'])],
        'Separate-host package and board counters cannot be summed into a measured complete system. The wall-power instrumentation request remains unmet.')
    diagram(FIG/'figure4_selection.png','Formal selection policy',[
        ('Steps 1–2',['Feasible set F: U95(ECE_source) ≤ tau and measured scoped J/1,000 ≤ E','Return no selection if F is empty','Leader: highest source AUROC among feasible arms']),
        ('Steps 3–4',['Retain arms with paired source 95% AUROC interval contained in [−delta,+delta]','Choose minimum energy; resolve equal costs by higher AUROC, then lower ECE upper bound','Report CADEC calibration/equivalence separately; delta, tau and E are illustrative'])],
        'All recalibration variants are tested as distinct arms. Interval inclusion is conservative relative to conventional 90% CI TOST at alpha=0.05.')
    fig,axes=plt.subplots(1,2,figsize=(8.3,4.8))
    for ax,domain in zip(axes,['test','cadec']):
        for arm in primary['catalogue']:
            model=arm['model'];energy=arm['energy_j_1000']
            if energy is None:continue
            method=arm['name'].split(' + ')[1];marker={'Uncalibrated':'o','TempScale':'^','Isotonic':'s'}[method]
            ax.scatter(energy,primary['domains'][domain]['metrics'][arm['name']]['auroc'],c=COLORS.get(model,'#555555'),marker=marker,s=60)
        ax.set(xscale='log',xlabel='Scoped gross energy (J / 1,000)',ylabel='AUROC',title='PsyTAR development audit' if domain=='test' else 'CADEC external diagnostic');ax.grid(alpha=.2)
    handles=[Patch(color=c,label=m) for m,c in COLORS.items() if f'{m} + Uncalibrated' in primary['domains']['test']['metrics']];fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=7)
    fig.suptitle('Discrimination versus measured-domain energy (primary grouped split)');fig.tight_layout(rect=[0,.17,1,1]);save_figure(fig,FIG/'figure5_pareto.png');plt.close(fig)
    models=[m for m in CORE_MODELS if f'{m} + Uncalibrated' in primary['domains']['test']['metrics']]
    fig,axes=plt.subplots(1,2,figsize=(8.3,4.8))
    for ax,domain in zip(axes,['test','cadec']):
        for j,method in enumerate(['Uncalibrated','TempScale','Isotonic']):
            vals=[[r['domains'][domain]['metrics'][f'{m} + {method}']['ece'] for r in summary['runs'] if f'{m} + {method}' in r['domains'][domain]['metrics']] for m in models]
            ax.bar(np.arange(len(models))+(j-1)*.24,[np.mean(v) for v in vals],.24,yerr=[np.std(v) for v in vals],label=method,capsize=3,color=['#9caabe','#39a8db','#2355a1'][j])
        ax.axhline(.07,color='#bd3232',ls='--',label='Illustrative tau = 0.07');ax.set_xticks(range(len(models)),[{'Logistic Regression':'LR','LightGBM':'LGBM','DistilBERT':'Distil','PubMedBERT':'PubMed','BERT-tiny':'Tiny'}[m] for m in models]);ax.set(ylabel='Adaptive ECE (mean ± split SD)',title='PsyTAR audit' if domain=='test' else 'CADEC');ax.legend(fontsize=8)
    fig.tight_layout();save_figure(fig,FIG/'figure6_calibration.png');plt.close(fig)
    panels=[]
    for domain in ['test','cadec']:
        series=[];first=None
        for model in models:
            curve=primary['domains'][domain]['metrics'][f'{model} + Isotonic']['decision_curve'];first=curve
            series.append((model,curve['thresholds'],curve['net_benefit_model'],{'color':COLORS[model]}))
        series.extend([('Review all',first['thresholds'],first['net_benefit_all'],{'ls':':','color':'#777777'}),('Review none',first['thresholds'],first['net_benefit_none'],{'color':'#aaaaaa'})])
        panels.append(('PsyTAR audit' if domain=='test' else 'CADEC',series,'Decision threshold','Net benefit'))
    line_chart(FIG/'figure7_decision_curves.png',panels,'Decision curves: isotonic arms; native corpus prevalence')
    tau=summary['provenance']['config']['tau_grid'];energy=summary['provenance']['config']['energy_grid_j_per_1000'];grid=primary['grid']
    names=sorted({r['selected'] for r in grid if r['selected']});palette=['#efefef']+list(plt.get_cmap('tab20').colors[:len(names)])
    code={None:0,**{n:i+1 for i,n in enumerate(names)}};lookup={(r['tau'],r['energy_budget']):r for r in grid}
    z=np.array([[code[lookup[t,e]['selected']] for e in energy] for t in tau])
    fig,ax=plt.subplots(figsize=(10,7));cmap=ListedColormap(palette);ax.imshow(z,cmap=cmap,norm=BoundaryNorm(np.arange(len(palette)+1)-.5,len(palette)),aspect='auto')
    ax.set_xticks(range(len(energy)),[str(e) for e in energy],rotation=45);ax.set_yticks(range(len(tau)),[f'{t:.2f}' for t in tau]);ax.set(xlabel='Illustrative energy budget (scoped J / 1,000)',ylabel='Illustrative ECE upper-bound limit',title='Complete source-only ECC-MS grid; all cells evaluated')
    for i,t in enumerate(tau):
        for j,e in enumerate(energy):ax.text(j,i,str(lookup[t,e]['feasible']),ha='center',va='center',fontsize=10)
    ax.legend(handles=[Patch(color=palette[0],label='Evaluated: no feasible arm')]+[Patch(color=palette[i+1],label=n) for i,n in enumerate(names)],bbox_to_anchor=(.5,-.25),loc='upper center',fontsize=11,ncol=2)
    if any(r.get('energy_status')=='incomplete_missing_energy' for r in grid):
        ax.set_title('Incomplete energy catalogue: counts/choices cover measured arms only')
    fig.tight_layout();save_figure(fig,FIG/'figure8_regime.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.3,4.5))
    for ax,domain in zip(axes,['test','cadec']):
        rows=[r for r in primary['workload_pairs'] if r['domain']==domain and r['target_sensitivity']==.9 and r.get('threshold_policy')=='retrospective_matched_sensitivity']
        ax.scatter([abs(r['delta_auroc_b_minus_a']) for r in rows],[abs(r['relative_flagged_review_change_pct']) for r in rows],s=14,alpha=.5)
        ax.axvline(.015,ls='--',color='#bd3232');ax.axhline(2,ls=':',color='#bd3232');ax.set(xlabel='Absolute AUROC difference',ylabel='Absolute relative flagged-review change (%)',title=domain)
    fig.suptitle('Retrospective workload at ≥90% sentence sensitivity');fig.tight_layout();save_figure(fig,FIG/'supplement_workload.png');plt.close(fig)
    surface_rows=[]
    for model in ['DistilBERT','PubMedBERT','BERT-tiny']:
        cells=json.loads((RESULTS/'gpu_energy'/f'{MODEL_FILES[model]}_surface.json').read_text())['cells']
        for cell in cells:
            surface_rows.append({'model':model,'checkpoint_seed':42,'batch_size':cell['batch_size'],'max_length':cell['max_length'],'gross_board_j_1000':cell['gross_j_1000'],'throughput_sps':cell['throughput_sps'],'average_board_w':cell['load_w'],'cv_pct':cell['cv_pct'],'repeatability_pass':cell['repeatability_pass'],'repeats':len(cell['repeats']),'scope':cell['scope'],'complete_system_energy_measured':False})
    frame=pd.DataFrame(surface_rows);frame.to_csv(RESULTS/'gpu_operating_grid.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained',sharey=True);norm=LogNorm(vmin=frame.gross_board_j_1000.min(),vmax=frame.gross_board_j_1000.max())
    for ax,model in zip(axes,['DistilBERT','PubMedBERT','BERT-tiny']):
        surface=frame[frame.model==model].pivot(index='max_length',columns='batch_size',values='gross_board_j_1000').sort_index().sort_index(axis=1)
        im=ax.imshow(surface.to_numpy(),norm=norm,cmap='viridis',aspect='auto');ax.set_xticks(range(len(surface.columns)),surface.columns);ax.set_yticks(range(len(surface.index)),surface.index);ax.set(title=model,xlabel='Batch size')
        for i in range(len(surface.index)):
            for j in range(len(surface.columns)):
                value=surface.iloc[i,j];ax.text(j,i,f'{value:.1f}',ha='center',va='center',fontsize=9,color='black' if norm(value)>.65 else 'white')
    axes[0].set_ylabel('Maximum inference length');fig.colorbar(im,ax=axes,label='GPU-board J / 1,000 (log scale)',shrink=.8)
    fig.suptitle('Tesla T4 board energy during raw-text pipeline timing\nPrimary checkpoints; three repeats per setting; host and wall energy unmeasured',fontsize=11);save_figure(fig,FIG/'supplement_gpu_energy_surface.png');plt.close(fig)


def replace_table(doc,old,headers,rows):
    if 'Table Grid' not in doc.styles:
        doc.styles.add_style('Table Grid',WD_STYLE_TYPE.TABLE)
    table=doc.add_table(rows=1,cols=len(headers));table.style='Table Grid'
    for c,t in zip(table.rows[0].cells,headers):c.text=str(t)
    for row in rows:
        for c,t in zip(table.add_row().cells,row):c.text=str(t)
    for i,row in enumerate(table.rows):
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:run.font.size=Pt(8);run.bold=(i==0)
    old.addprevious(table._tbl);old.getparent().remove(old)


def picture(doc,element,path):
    p=Paragraph(element,doc._body)
    for drawing in list(element.xpath('.//w:drawing')):drawing.getparent().remove(drawing)
    if p.text.strip():
        # Caption notes and the Step-4 paragraph originally contained drawings.
        paragraph=doc.add_paragraph();element.addnext(paragraph._p);p=paragraph
    p.add_run().add_picture(str(path),width=Inches(6.3))


def cpu_families(primary):
    return [m for m in ['DistilBERT','PubMedBERT','BERT-tiny'] if m+' CPU FP32 + Uncalibrated' in primary['domains']['test']['metrics']]


def hardware_rows(primary):
    old=json.loads((ROOT/'results/frozen_split_reconciled.json').read_text())['energy_by_model'];rows=[]
    for model in ['Logistic Regression','LightGBM']:
        d=old[model];rows.append([model,'CPU package',f"{d['load_w']:.2f}",f"{d['throughput_sps']:.2f}",f"{d['inf_j_gross']:.3f}",f"{d['inf_j_net']:.3f}",f"{d['energy_cv_pct']:.2f}",7])
    for model in ['DistilBERT','PubMedBERT','BERT-tiny']:
        path=RESULTS/'gpu_energy'/f'{MODEL_FILES[model]}_surface.json'
        cells=json.loads(path.read_text())['cells'];d=next(c for c in cells if (c['batch_size'],c['max_length'])==(64,128))
        rows.append([model+' GPU FP16','GPU board',f"{d['load_w']:.2f}",f"{d['throughput_sps']:.2f}",f"{d['gross_j_1000']:.3f}",'Unmeasured',f"{d['cv_pct']:.2f}",len(d['repeats'])])
    audit=json.loads((RESULTS/'linux_artifact_audit.json').read_text())
    for model in cpu_families(primary):
        for precision in ['FP32','INT8']:
            stem=MODEL_FILES[model+' CPU '+precision];r=next(r for r in audit['cpu_predictions'] if r['model']==stem)
            d=json.loads((ROOT/r['energy_path']).read_text())['cells'][0]
            rows.append([model+' CPU '+precision,'CPU package',f"{d['load_w']:.2f}",f"{d['throughput_sps']:.2f}",f"{d['gross_j_1000']:.3f}",f"{d['net_j_1000']:.3f}",f"{d['cv_pct']:.2f}",len(d['repeats'])])
    return rows


def quantization_evidence(primary):
    audit=json.loads((RESULTS/'linux_artifact_audit.json').read_text());records={r['model']:r for r in audit['cpu_predictions']}
    lookups={domain:pair_lookup(primary['domains'][domain]['paired_auroc']) for domain in ['test','cadec']};rows=[]
    for model in cpu_families(primary):
        fp_name=model+' CPU FP32 + Uncalibrated';q_name=model+' CPU INT8 + Uncalibrated'
        fp=records[MODEL_FILES[model+' CPU FP32']];q=records[MODEL_FILES[model+' CPU INT8']]
        with np.load(ROOT/fp['prediction_path'],allow_pickle=False) as a,np.load(ROOT/q['prediction_path'],allow_pickle=False) as b:
            maximum=max(float(np.max(np.abs(a[f'probs_{d}_uncal'][:,1]-b[f'probs_{d}_uncal'][:,1]))) for d in ['calib','test','cadec'])
        row={'model':model,'fp32_package_j_1000':fp['gross_j_1000'],'int8_package_j_1000':q['gross_j_1000'],'energy_reduction_pct':100*(1-q['gross_j_1000']/fp['gross_j_1000']),'maximum_uncalibrated_probability_change':maximum}
        for domain in ['test','cadec']:
            fm=primary['domains'][domain]['metrics'][fp_name];qm=primary['domains'][domain]['metrics'][q_name];lo,hi=lookups[domain][q_name,fp_name]
            row.update({f'{domain}_delta_auroc':qm['auroc']-fm['auroc'],f'{domain}_delta_ci_lo':lo,f'{domain}_delta_ci_hi':hi,f'{domain}_fp32_ece':fm['ece'],f'{domain}_int8_ece':qm['ece']})
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS/'cpu_quantization_comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    x=np.arange(len(rows))
    for offset,key,label,color in [(-.18,'fp32_package_j_1000','FP32','#557eaa'),(.18,'int8_package_j_1000','Dynamic INT8','#dda34d')]:
        bars=axes[0].bar(x+offset,[r[key] for r in rows],width=.36,label=label,color=color)
        axes[0].bar_label(bars,labels=[f'{r[key]:.0f}' for r in rows],fontsize=8,padding=3)
    axes[0].set_xticks(x,[r['model'] for r in rows]);axes[0].set(yscale='log',ylabel='CPU-package J / 1,000 (log)',title='Same-checkpoint measured package energy');axes[0].legend(fontsize=8)
    axes[0].set_ylim(top=max(r['fp32_package_j_1000'] for r in rows)*1.7)
    for offset,domain,label,color in [(-.12,'test','Source audit','#2463aa'),(.12,'cadec','External CADEC','#ad493b')]:
        y=x+offset;delta=np.array([r[f'{domain}_delta_auroc'] for r in rows]);low=np.array([r[f'{domain}_delta_ci_lo'] for r in rows]);high=np.array([r[f'{domain}_delta_ci_hi'] for r in rows])
        axes[1].errorbar(delta,y,xerr=np.vstack([delta-low,high-delta]),fmt='o',capsize=3,label=label,color=color)
    axes[1].axvline(0,color='#777777',lw=1);axes[1].set_yticks(x,[r['model'] for r in rows]);axes[1].set(xlabel='INT8 minus FP32 AUROC; paired 95% cluster interval',title='Uncalibrated discrimination change');axes[1].legend(fontsize=8)
    fig.suptitle('Primary seed 42 only; component energy and predictive effects require joint interpretation',fontsize=10);fig.tight_layout();save_figure(fig,FIG/'supplement_cpu_quantization.png');plt.close(fig)
    return rows


def integrated_text(summary,primary):
    seed_count=len(summary['runs']);count=len(primary['catalogue']);common=summary['common_catalogue'];config=summary['provenance']['config']
    frequencies=[r for r in common['selection_frequencies'] if r['tau']==.07 and r['energy_budget']==120]
    freq='; '.join(f"{r['arm']}: {r['count']}/{r['splits']}" for r in frequencies)
    strict=[r['feasible'] for r in primary['grid'] if r['tau']==.03]
    audit=json.loads((RESULTS/'linux_artifact_audit.json').read_text());by_stem={r['model']:r for r in audit['cpu_predictions']}
    reductions=[]
    for m in cpu_families(primary):
        fp=by_stem[MODEL_FILES[m+' CPU FP32']]['gross_j_1000'];q=by_stem[MODEL_FILES[m+' CPU INT8']]['gross_j_1000']
        reductions.append(f'{m}: {100*(1-q/fp):.1f}%')
    source=primary['domains']['test']['metrics'];external=primary['domains']['cadec']['metrics']
    pair=pair_lookup(primary['domains']['test']['paired_auroc'])[('PubMedBERT + Uncalibrated','DistilBERT + Uncalibrated')]
    high=[n for n in common['arm_names'] if np.mean([r['domains']['cadec']['metrics'][n]['ece'] for r in summary['runs']])>.07]
    ppv=[source[n]['clinical_utility']['prevalence_adjusted']['1.0%']['ppv'] for n in common['arm_names']]
    return {
      0:'Original research article — revision draft with returned CPU/GPU experiments integrated',
      3:f'Energy- and calibration-constrained model selection (ECC-MS) is an auditable policy over a finite catalogue of adverse-drug-reaction classifiers. Five classifier families contribute 15 model–recalibration configurations evaluated over {seed_count} review-disjoint PsyTAR splits, with CADEC reserved for external diagnostics. Twelve additional CPU FP32/INT8 configurations are evaluated on the primary split. Source feasibility requires a cluster-bootstrap calibration upper bound and a scoped energy budget; paired source AUROC interval containment within an illustrative tolerance precedes minimum-energy selection. Adaptive ECE uses 10 bins and 2,000 whole-review/post bootstrap draws, with bin-count and equivalence-margin sensitivity. A complete 165-cell constraint grid is evaluated for each split. New measurements cover three GPU encoders over 15 batch/length settings and same-checkpoint CPU FP32/INT8 comparisons. These component measurements show configuration-dependent costs; they do not measure complete-system energy. Workload at sensitivity-targeted thresholds is measured directly, and no clinical guarantee is inferred from an AUROC tolerance. The policy can report no feasible arm and does not guarantee target-corpus calibration under distribution shift.',
      10:'The revision evaluates five classifier families on two English corpora. The common catalogue has fifteen recalibration configurations over ten grouped splits. Four CPU execution variants add twelve recalibration configurations on primary seed 42 only. Review identifiers establish review-disjoint partitions, not verified patient identities. Primary source selection, repeated-split core selection and the primary CPU extension have separate candidate catalogues and denominators.',
      42:'Table 1 describes logistic regression, LightGBM, DistilBERT, PubMedBERT and compact BERT-tiny (prajjwal1/bert-tiny). The three encoders are trained for three epochs with AdamW, learning rate 2e-5, batch 64 and maximum length 128 on each of ten grouped training splits. GPU inference uses FP16. Saved seed-42 DistilBERT and BERT-tiny checkpoints are evaluated on CPU in FP32 and dynamic INT8 without retraining; INT8 replaces Linear modules and uses the available FBGEMM backend on the measured host. Recalibrators are refitted on the reserved calibration probabilities of each execution variant. The planned CPU extension covers DistilBERT and BERT-tiny; PubMedBERT remains a GPU baseline.',
      44:'Table 1. Classifier families, execution variants and training configuration; CPU extensions use the primary checkpoint only.',
      55:'The returned measurements cover disclosed CPU-package and GPU-board domains. Figure 3 separates those domains; no wall-power or complete-system measurement is available.',
      57:'New accelerator measurements use an NVIDIA Tesla T4 and FP16. Raw text is tokenised and converted to probabilities within the elapsed-time workload. Board power is sampled with actual timestamps and integrated over each window. Five batches (8,16,32,64,128) and three maximum inference lengths (64,128,256) yield fifteen settings per encoder, each with three repeats after warm-up. The representative table uses batch 64 and maximum length 128. Host CPU and RAM energy are outside the measured board domain. Longer inference lengths are sensitivity settings; the checkpoints were trained at length 128.',
      58:'The returned Linux extension uses a bare-metal Intel i5-8500, six CPU threads and readable package-0 RAPL counters. DistilBERT and BERT-tiny are evaluated in FP32 and dynamic INT8 at batch 64 and maximum length 128, with 60-second warm-up, seven windows of at least 20 seconds and 15-second cooldown. Gross energy, throughput and load power are taken from the same median-energy repeat; net package energy subtracts the mean pre/post idle power. Tokenizer-only diagnostics are retained separately. The initial PubMedBERT tokenizer CV was 10.81%, above the configured 10% threshold; its separately returned follow-up CV was 0.38%. Both raw runs remain available. Neither Linux package measurements nor Colab board measurements include whole-system consumption, and values from different hosts are not added into a measured system total.',
      60:'Figure 3. Measured CPU-package and GPU-board scopes, returned extensions and the remaining whole-system instrumentation gap.',
      65:f'All thresholds are illustrative experimental values, not elicited pharmacovigilance deployment requirements. The main grid crosses {len(config["tau_grid"])} calibration limits and {len(config["energy_grid_j_per_1000"])} energy budgets (0.1–300 J/1,000), with ten equivalence margins. Repeated selection uses the same fifteen-arm catalogue across all {seed_count} grouped splits. Primary seed 42 additionally evaluates {count} arms. A separate CPU-only grid includes 18 package-measured arms and extends illustrative budgets to 4,000 J/1,000 because the larger CPU encoders exceed the main grid. This is a sensitivity extension, not a claimed deployment specification. An external-assisted sensitivity uses CADEC intervals to alter a decision and therefore forfeits independent CADEC evaluation of that assisted choice.',
      69:'4.1 Measured Power, Throughput, Energy and CPU Quantisation',
      70:'Table 2 combines the original package-measured classical operating points with the returned GPU raw-text timing and CPU encoder operating points. The operating point is batch 64 and length 128 for encoders; classical throughput includes TF-IDF processing. Gross GPU board energy is 75.473, 152.233 and 11.028 J/1,000 for DistilBERT, PubMedBERT and BERT-tiny. Dynamic INT8 reduces CPU-package energy relative to each FP32 checkpoint ('+'; '.join(reductions)+'). Quantisation also changes discrimination and individual probabilities; its predictive effects and paired source/external intervals are reported in Supplementary Table S2. Similar AUROC does not imply interchangeable individual risks. CPU measurements use one primary checkpoint; GPU energy is not independently remeasured for every split. Calibration-transform energy is not separately resolved.',
      72:'Table 2. Measured component energy and throughput at the canonical operating point; counter domains are explicitly different.',
      74:'CV is the sample coefficient of variation across repeated measurement windows. CPU values cover package energy and raw-text processing; GPU values cover board energy during raw-text pipeline timing. Net GPU energy and all whole-system values are unmeasured. Costs are held fixed across grouped-split model-selection analyses and do not represent independent hardware repeats across all seeds.',
      76:f'Table 3 reports source and external metrics for the {count} primary configurations. The common five-family catalogue contributes fifteen configurations; four CPU precision variants contribute twelve more. All predictive values are recomputed from the returned artifacts. Model and calibration rankings are empirical results for these corpora and execution settings, not architecture-wide claims.',
      78:f'Table 3. Discrimination and calibration for all {count} primary-split configurations; ECE intervals resample complete reviews/posts.',
      81:'Figure 5. AUROC against measured-domain energy for primary configurations; CPU package and GPU board costs are not whole-system equivalents.',
      82:'4.3 Stability Across Grouped Splits',
      83:f'Table 4 reports descriptive means and population standard deviations over {seed_count} grouped splits for the common fifteen-arm catalogue. These reflect both partition and training variation, not isolated optimiser variation on one fixed split. CPU precision variants have only the primary checkpoint and are excluded from these multi-split summaries. External mean ECE exceeds the illustrative 0.07 point limit for {len(high)} of the fifteen common arms; feasibility still uses each split’s source upper confidence bound.',
      84:'Repeated-split differences in external calibration vary by architecture and recalibration method. No universal capacity–stability relationship or general superiority under shift is inferred from these two English corpora.',
      85:f'Table 4. Mean ± population SD across {seed_count} grouped splits for the common fifteen configurations.',
      87:'Seeds are 42, 123, 456 and 1001–1007. CPU FP32/INT8 configurations are primary-seed diagnostics and have no ten-split standard deviation.',
      89:'Figure 6. Adaptive ECE across ten grouped splits for the five common classifier families; bars show descriptive mean ± split SD.',
      91:'Table 5 recalculates sensitivity, specificity, likelihood ratios and prevalence-adjusted predictive values at threshold 0.5 for all fifteen common configurations. These operating characteristics are distinct from the calibration-frozen thresholds targeting 85%, 90% and 95% sentence sensitivity used in the workload supplement. Achieved evaluation sensitivity can be lower than a calibration target.',
      92:f'Under an illustrative 1% ADR-sentence prevalence, adjusted PPV for these fifteen configurations ranges from {min(ppv):.3f} to {max(ppv):.3f}. This calculation assumes sensitivity and specificity are preserved under a prevalence change; it is not an observed deployment result and does not guarantee transport under other distribution shifts. Review workload is measured directly, rather than inferred from an AUROC difference.',
      93:'Table 5. Source operating characteristics at threshold 0.5 and illustrative prevalence-adjusted PPV for the common fifteen configurations.',
      97:'Figure 7. Decision curves for isotonic arms of the five common classifier families on the source audit and external corpus.',
      99:f'Table 6 and Figure 8 use every evaluated cell in the 165-cell primary grid with {count} available arms. Repeated-split frequencies separately apply the same fifteen-arm catalogue on all {seed_count} splits. At illustrative tau=0.07 and E=120 J/1,000, the common-catalogue choices are {freq}. Different catalogue cohorts are not pooled into one frequency denominator. The CPU-only expanded-budget grid is supplied separately.',
      100:f'At tau=0.03, the number of primary feasible configurations ranges from {min(strict)} to {max(strict)} over the evaluated main energy budgets. These are outcomes of the present source intervals and operating costs. The calibration threshold, interval method, binning and sample size all affect feasibility; the observed result is not an intrinsic property of ADR screening.',
      101:'As the energy budget increases at fixed calibration tolerance, feasible counts cannot decrease. The selected arm can nevertheless change because a newly affordable arm alters the source AUROC leader and its equivalence set. The regime map is generated from the measured inputs, with evaluated infeasible cells distinguished from missing measurements.',
      102:f'The primary PubMedBERT-minus-DistilBERT uncalibrated source AUROC interval is [{pair[0]:+.4f}, {pair[1]:+.4f}]. Its relation to zero and containment within a chosen margin answer separate statistical questions. A margin of 0.015 is illustrative; it is not a clinical threshold or a guarantee about workload. Decisions in Table 6 use the actual calibration variants and scoped costs, not a blanket tie assigned to all variants.',
      104:'Table 6. Representative cells from the full primary source-only constraint grid.',
      108:'Figure 8. Primary source-only ECC-MS grid generated from all measured inputs; numbers are feasible counts, colours are selected arms.',
      113:'The harmonisation audit distinguishes six sentences touched by crossing segments from exactly two sentence-label changes under segment containment (2,865 versus 2,863 positives). This is a limited annotation-rule diagnostic. Revised uncertainty intervals and every selection cell have not been recomputed under alternative annotation schemes, so full selection invariance is not asserted.',
      114:'Sentence prediction and post-level max-pooling define different estimands and prevalences. Historical post-level figures in the submission are archived rather than treated as results from the returned checkpoints. The revised workload analysis directly counts distinct flagged reviews/posts; it does not substitute post-level classification metrics for sentence-level calibration.',
      120:'The revised tables show recalibration-specific source and external ECE estimates and cluster uncertainty. Isotonic regression can fit a more flexible monotone probability map than a single temperature, while also introducing score ties. The observed values do not establish that one method universally repairs distribution shift or that lower average ECE guarantees a safer operating threshold.',
      121:'Recalibration is fitted using source calibration reviews only. Changes in prevalence, annotation and review style can alter its behaviour after transfer. Method choice therefore requires target-representative validation; mechanistic explanations about a model’s confidence shape are not inferred solely from a few aggregate ECE values.',
      124:'Quantisation, compact architecture, recalibration and operating point affect different dimensions of the evaluated system. The returned CPU energy reductions are accompanied by configuration-specific AUROC and calibration changes. These measured trade-offs support joint evaluation, without implying negligible clinical impact or a universal architecture ranking.',
      127:'Prevalence-adjusted predictive values and directly measured review workloads are complementary. The former assumes fixed sensitivity and specificity under prior shift; the latter counts the review/post burden at an explicitly declared threshold policy. Neither establishes prospective analyst time, verified adverse events or clinical safety.',
      129:'Energy remains limited to package-0 RAPL on the Linux CPU host and board power on the Colab T4. Host tokenisation is included in elapsed pipeline timing, but Colab host CPU/RAM energy is unmeasured; CPU package counters exclude the rest of the system. There is no GPU-equipped instrumented Linux host or wall-power meter. The complete-system measurement requested by Reviewer 2 therefore remains unmet. The four CPU precision variants use only seed 42; their results do not establish split-robust quantisation effects. Checkpoint weight files were not returned locally: matching corpus arrays, prediction hashes and close GPU-FP16/CPU-FP32 probabilities provide compatibility evidence, not a cryptographic proof of identical weights. The supplied repository runner reproduces the planned CPU extension.',
      130:'The label-mapping audit covers one sentence splitter and a narrow overlap-versus-containment distinction. Its two label disagreements are not a broad stress test of annotation choices, languages or therapeutic areas. A different splitter, span convention or label taxonomy can produce larger changes and requires a fresh evaluation.',
      135:'ECC-MS specifies source-calibration uncertainty, scoped energy and practical AUROC tolerance as explicit finite-catalogue selection inputs. The revision provides mathematical definitions, review/post resampling, complete constraint grids, directly measured workload and ten grouped-split evaluations of a common fifteen-arm catalogue. Twelve primary CPU calibration configurations and GPU batch/length surfaces extend the measured trade-offs. External calibration is diagnosed separately, with no shift guarantee.',
      136:'Within the evaluated English corpora, choices depend on the declared tolerances, execution setting, measured domain and candidate catalogue. Some constraint combinations admit no arm. Calibration changes and INT8 energy savings require predictive and operational evaluation together; no result establishes clinical equivalence or transfer to untested settings.',
      137:'Future work should elicit operational tolerances independently, evaluate additional therapeutic areas, languages and platforms, measure prospective analyst workload and subgroup calibration, and obtain complete-system energy with suitable instrumentation. The present component measurements and illustrative margins do not replace those requirements.'
    }


def write_manuscript(summary,primary):
    original=ROOT/'AIH Paper.docx';doc=Document(original);blocks=list(doc._element.body)
    config=summary['provenance']['config'];seed_count=len(summary['runs']);pairs=pair_lookup(primary['domains']['test']['paired_auroc']);external=pair_lookup(primary['domains']['cadec']['paired_auroc'])
    pair=('PubMedBERT + Uncalibrated','DistilBERT + Uncalibrated');slo,shi=pairs[pair];elo,ehi=external[pair]
    frequencies=[r for r in summary['selection_frequencies'] if r['tau']==.07 and r['energy_budget']==120]
    freq='; '.join(f"{r['arm']}: {r['count']}/{r['splits']} splits" for r in frequencies)
    text={
      0:'Original research article — revision working draft; additional hardware experiments pending',
      6:'Patient-authored medicine reviews contain descriptions of suspected adverse reactions that motivate pharmacovigilance research. Harpaz et al. [1] survey data-mining approaches, Leaman et al. [2] demonstrate extraction from user-generated health-network posts, and Sarker and Gonzalez [3] examine multi-corpus classification. Automated screening can help prioritise text at scale; its output remains a candidate signal for human assessment rather than a verified adverse event.',
      7:'Locally hosted screening systems can face hardware, procurement, privacy and latency constraints. A model also needs probabilities whose interpretation is examined on data resembling its intended use. Discrimination, resource cost and probability calibration describe different properties, and favourable development-corpus calibration may not survive a change of corpus.',
      8:'This motivates an explicit decision protocol over an evaluated model catalogue. Multi-objective and constrained learning already address combinations of predictive quality and operational objectives [65–67]. The present study specifies which measured energy and calibration criteria are applied before a discrimination comparison, how uncertainty and a declared tolerance affect a choice, and when no candidate is returned.',
      3:f'Energy- and calibration-constrained model selection (ECC-MS) is an auditable decision policy for a finite catalogue of adverse-drug-reaction classifiers. The present revision reanalyses the original 12 configurations on PsyTAR and CADEC using review-disjoint partitions, 10-bin adaptive calibration error and 2,000 whole-review/post bootstrap resamples. Feasibility requires both a source-corpus calibration upper bound and a measured-domain energy budget to be satisfied. Practically equivalent feasible arms are compared using paired source AUROC intervals and an illustrative margin, followed by minimum-energy selection. CADEC remains outside this primary selection and supplies separate external diagnostics. The complete constraint grid is evaluated over {len(config["tau_grid"])} calibration limits and {len(config["energy_grid_j_per_1000"])} energy budgets, with ten equivalence margins and repeated selection across {seed_count} grouped splits. A strict calibration requirement can leave no feasible arm under the revised intervals. Isotonic recalibration improves the observed external calibration estimates, but these findings do not guarantee reliability under arbitrary distribution shift. Original CPU figures cover RAPL package energy including vectorisation; original GPU figures cover accelerator-board forward-pass energy and exclude host tokenisation. Complete-system energy was not measured. Compact/quantized encoder and batch/length hardware extensions await the supplied Linux and Colab runs.',
      9:'ECC-MS combines established constrained-selection and practical-equivalence ideas into an explicit policy for an evaluated ADR model catalogue. It filters configurations on a calibration upper bound and a scoped energy budget, identifies the best source discrimination, and minimises energy within a declared practical-equivalence band. External calibration and discrimination are then reported without changing this primary source-only choice. The historical CADEC-assisted gate is retained only as a labelled sensitivity analysis; its output cannot also be described as independently validated on CADEC.',
      10:'The original evaluation covers four models on two English corpora. Sentences are nested within reviews, so partitioning uses review identifiers and resampling uses reviews/posts. These identifiers establish review-disjoint partitions, not verified patient-disjoint partitions across separate reviews. The following sections describe prior approaches, methods, revised results and their limits.',
      17:'Compression methods such as distillation [24–26] and quantisation [27] offer additional candidate configurations. Training and inference contribute differently to lifetime energy depending on volume and retraining frequency, so an inference measurement alone does not establish lifecycle savings. RAPL characterisation and validation studies [28,29] motivate reporting measured hardware domains and their limits rather than treating a component counter as whole-system power.',
      25:'Multi-objective selection already addresses competing performance and resource objectives, while cost-sensitive learning attaches costs to predictive decisions. Constrained optimisation provides a general language for policy requirements; energy-aware search and calibration methods address further parts of this problem. ECC-MS contributes their explicit application as a finite-catalogue ADR selection and reporting protocol, including per-arm uncertainty filtering and an allowed empty set. It does not establish a new optimisation algorithm or a universal reliability guarantee. The comparison in the revision supplement distinguishes the decision object, cost definition, uncertainty treatment and role of external data [65–68].',
      28:'Figure 1 separates fitting, source-only selection and nonbinding external diagnostics. Candidate classifiers are trained on PsyTAR training reviews and recalibrators fitted on reserved calibration reviews. The remaining source split is a development audit used to compare and select arms; its selection-associated performance is therefore apparent performance, rather than a new independent test estimate. CADEC provides the separate external evaluation. Analysis scripts, cluster identities, measured-data figures and input checksums are supplied with the revision.',
      32:'PsyTAR supplies 6,003 sentences from 886 reviews. CADEC supplies 7,823 sentences from 1,248 non-empty posts; two empty placeholders are excluded. The annotation parser emits 7,409 continuous span segments from 6,318 ADR entity records, including discontinuous entities. Segments and entity records are distinguished rather than reported as interchangeable annotation counts.',
      34:'Rule A labels a sentence positive on overlap with an ADR segment; Rule B requires containment of a segment. Exactly two sentence labels change (2,865 versus 2,863 positives). Crossing segments touch six sentences in ten sentence-segment events; the affected-sentence count is not the label-disagreement count. Original post identifiers and sentence offsets are reconstructed from raw files and checked elementwise against the submitted harmonised texts and labels. Rule C remains a separate post-level max-pooling diagnostic.',
      38:'A random sentence-level split can put sentences from the same review into both training and evaluation, allowing review-specific information to cross the boundary. Grouped partitioning prevents that overlap. The previously quoted approximate 99% contamination calculation based only on average review length is not used as an empirical estimate.',
      39:'Grouped shuffling assigns 60% of review identifiers to training and splits the rest evenly into calibration and development audit partitions. The primary split contains 3,626, 1,188 and 1,189 sentences, respectively; pairwise review-ID overlap is zero for each evaluated seed. Short generic strings can recur in different reviews, so review independence does not imply exact-string disjointness. Distinct review IDs do not establish distinct patient identities, and no patient-linkage audit is claimed.',
      40:'CADEC enters neither classifier fitting, recalibration fitting nor primary selection. Its labels are used for external metrics, cluster intervals and explicitly retrospective operating-point diagnostics. A separate historical-gate sensitivity uses CADEC to influence a choice and is therefore labelled external-assisted rather than an independent test of that assisted choice.',
      42:'The original catalogue contains the four models and twelve recalibration configurations in Table 1. Classical TF-IDF settings and transformer training settings are unchanged. LightGBM uses its own implementation; logistic regression uses scikit-learn. Additional scripts train a compact BERT-tiny encoder and evaluate same-checkpoint CPU FP32 and dynamic INT8 encoders. Those extension results are pending and are not included as measured results in the present working draft.',
      43:'Generative model comparisons provide context for task-specific evaluation, not direct ADR baselines here. HELM motivates evaluating multiple properties including calibration and efficiency [69], and task-specific classifier comparisons examine encoder-style and generative approaches under a common task definition [70]. Their findings do not establish ADR performance for this dataset. Prompt variance, label definitions, fine-tuning access, privacy and measurement scope would have to be controlled in a direct comparison.',
      48:'Each model contributes uncalibrated, temperature-scaled and isotonic arms. Recalibrators are fitted only on the calibration partition. Their extra energy has not been separately resolved by the present hardware protocol; the revised text does not infer that the cost is zero or proven below meter resolution.',
      49:'Temperature scaling preserves score ordering apart from finite-precision ties; isotonic maps may introduce ties and alter AUROC slightly. Every recalibration variant is therefore treated as a distinct arm in paired comparisons. Fixed-threshold operating characteristics use 0.5. Workload analyses additionally use calibration-fitted sensitivity thresholds and separately labelled retrospective matched-sensitivity thresholds; neither is called a calibration-tuned F1 optimum.',
      51:'Discrimination is assessed with AUROC and AUPRC, alongside Brier score and clipped-probability log loss. Adaptive ECE uses 10 equal-mass rank bins, with deterministic corpus-order handling when ties cross bin boundaries. Bin-count sensitivity covers 5, 10, 15 and 20 bins. Confidence intervals use 2,000 percentile bootstrap draws: reviews are sampled with replacement in PsyTAR, posts in CADEC, and all sentences of every selected cluster are retained, including repeated draws. The same draws are shared across arms for paired AUROC differences. The estimand remains sentence-weighted performance with within-review dependence. Row-bootstrap comparisons use the same percentile method to separate the resampling-unit effect from the original BCa-versus-percentile distinction.',
      52:'Predictive values are evaluated under illustrative prevalences of 1%, 5%, 10%, 20% and 36.12%, assuming sensitivity and specificity remain fixed. This prior-shift calculation does not guarantee transport under covariate or concept shift. Workload is also measured directly as the number of distinct reviews/posts with at least one flagged sentence. Thresholds targeting 85%, 90% and 95% sentence sensitivity are fitted on calibration data and frozen before evaluation; achieved test/external sensitivity is reported and can fall below the target. A second, retrospective diagnostic matches sensitivity on evaluated labels and is explicitly excluded from deployment selection.',
      53:f'Practical equivalence is defined by containment of a paired 95% cluster-bootstrap AUROC interval within [−δ,+δ]. This is a conservative interval-inclusion rule relative to conventional TOST using a 90% interval at α=0.05; it is not inferred from a nonsignificant difference. The reference δ=0.015 is an illustrative decision tolerance, not an independently elicited clinical standard or a documented prespecified workload guarantee. Sensitivity covers δ=0.005 to 0.050 in increments of 0.005. We withdraw the asserted mapping from 0.015 AUROC to less than 2% extra workload. No such general implication follows from AUROC, and the measured operating-point diagnostics do not validate it clinically. The previous minimum-detectable-difference calculation used a heuristic scale factor and is removed; no formal power claim is made from it.',
      55:'The submitted measurements cover distinct hardware domains. Figure 3 identifies their scope and the pending extensions; complete-system energy was not measured.',
      56:'The original CPU measurements use package-0 RAPL counters on a bare-metal Intel i5-8500 host with six physical cores and one socket. The overlapping psys domain and all subzones are excluded; DRAM is outside the chosen CPU-package scope, not a measured whole-system component. Idle measurements before and after benchmarking were 4.29 W and 3.41 W (mean 3.85 W). Each classical arm uses a 60-second warm-up, seven 20-second windows and 15-second cooldowns. A coupled median repeat supplies power, throughput and energy; CVs are 0.69% and 0.51%.',
      57:'The original accelerator benchmark uses an NVIDIA T4, FP16 and a fixed resident batch of size 64 padded to 128 tokens. Board power is sampled using actual timestamps and integrated to estimate average board power over the forward-pass workload. Tokenisation and host-to-device preparation occur before the timed saturated loop, so the reported board J/1,000 and throughput omit these stages. Reported coefficients of variation are averages of within-seed repeat coefficients, not the across-seed coefficient of variation.',
      58:'CPU package measurements including TF-IDF vectorisation and GPU board-only forward-pass measurements do not share a complete raw-text-to-probability energy scope. Their ratios describe the disclosed measured domains on the tested hosts; they are not complete-system savings or portable model constants. New scripts measure Colab board energy during raw-text pipeline timing over batches 8,16,32,64,128 and lengths 64,128,256, and Linux package energy for tokenisation and same-checkpoint FP32/INT8 encoders. The available Linux host has no GPU or wall-power meter. Its tokenizer measurements cannot be added to a different Colab machine and described as measured whole-system energy. This limitation remains outstanding for Reviewer 2’s complete-system request.',
      61:'3.8 Formal ECC-MS Selection and External Diagnostics',
      62:'Let A be the finite set of model–recalibration arms, Dsrc the review-disjoint source development audit, and Dext the held-out external corpus. For arm a, let R̂src(a) be AUROC, Ĉsrc(a) adaptive ECE, U95(Ĉsrc(a)) its cluster-bootstrap upper confidence limit, and Jscope(a) gross measured-domain energy in joules per 1,000 sentences. The user-specified τ, E and δ denote a calibration limit, a scoped energy budget and a practical-equivalence tolerance, respectively.',
      63:'Step 1 defines F(τ,E) = {a ∈ A : U95(Ĉsrc(a)) ≤ τ and Jscope(a) ≤ E}. Missing energy measurements are not replaced by constants. When F is empty, the policy returns no selected arm. These are estimated per-arm constraints, not simultaneous confidence guarantees for a data-selected configuration.',
      64:'Step 2 defines a* ∈ argmax[a ∈ F] R̂src(a). Step 3 forms Sδ = {a ∈ F : L95[Δsrc(a*,a)] ≥ −δ and U95[Δsrc(a*,a)] ≤ δ}, where Δsrc(a*,a) = R̂src(a*) − R̂src(a) and both arms use the same cluster draws. The leader belongs to Sδ by definition. Step 4 returns â ∈ argmin[a ∈ Sδ] Jscope(a). Equal-energy ties are resolved by higher source AUROC, then lower ECE upper bound, then arm name. External Δ intervals and ECE bounds are reported after selection and do not redefine F or Sδ.',
      65:f'The constraint values are illustrative experimental settings rather than requirements elicited from a pharmacovigilance deployment. The full grid crosses {len(config["tau_grid"])} τ values (0.01–0.20) and {len(config["energy_grid_j_per_1000"])} E values (0.1–300 J/1,000). We also evaluate ten δ values and repeat complete selection on all {seed_count} available grouped splits. The energy axis inherits the measured-domain scope. A supplementary external-assisted variant intersects Sδ with the external interval-inclusion condition; it is not the primary rule and has no independent CADEC evaluation after that use.',
      66:'Feasible counts must be non-decreasing as either τ or E increases with the other fixed. Every grid cell is evaluated and both monotonicity properties are asserted before plotting. If no arm satisfies the estimated constraints, the policy reports infeasibility without relaxing requirements. A larger budget cannot remove a previously feasible arm.',
      67:'Figure 4. Formal source-only ECC-MS selection, with external diagnostics kept separate.',
      71:'Per-1,000 figures, throughput and ratios remain conditional on hardware, precision, batch size, sequence length, workload and measurement domain. They cannot be transferred unchanged to another deployment host or interpreted as whole-system savings.',
      74:'CV is the within-run repetition coefficient; GPU values average those coefficients across seeds. CPU throughput includes TF-IDF; original GPU throughput is forward-only.',
      83:'Table 4 reports descriptive means and population standard deviations across the available grouped training/audit splits. These combine split and training variation; they do not isolate optimiser randomness on a fixed split. At the illustrative external ECE point limit of 0.07, uncalibrated logistic regression, temperature-scaled logistic regression and uncalibrated DistilBERT exceed the limit on their multi-split means. Primary feasibility still uses each split’s source upper confidence bound, not these external means.',
      84:'Among the two uncalibrated transformer arms, PubMedBERT has lower observed external ECE variation than DistilBERT in the available grouped splits. This descriptive comparison is specific to these splits and does not establish that calibration stability generally improves with model capacity.',
      99:f'Table 6 shows representative cells from the complete {len(config["tau_grid"])*len(config["energy_grid_j_per_1000"])}-cell grid for the primary split. Figure 8 is generated directly from the full evaluated table. Selection-frequency tables repeat the complete rule across {seed_count} grouped splits with the candidate catalogue and denominator stated explicitly.',
      100:'Under the revised cluster-percentile intervals, feasibility at τ=0.03 is evaluated across every energy budget rather than assumed from a single low-budget cell. The accompanying full-grid table is authoritative: increased energy can only enlarge the feasible set. Differences from the submission reflect both the resampling-unit correction and the explicit change from row BCa to cluster-percentile intervals; the row-percentile control in the supplement isolates the unit effect.',
      101:'Moderate calibration and energy constraints can select a classical arm, while larger energy budgets may admit higher-discrimination encoders. These are empirical transitions in the evaluated candidate set under illustrative budgets, not validated procurement recommendations.',
      102:f'For the uncalibrated transformer comparison on the primary source split, the PubMedBERT-minus-DistilBERT paired 95% cluster interval is [{slo:+.4f}, {shi:+.4f}]. The reference-margin substitution is justified only in the explicitly defined statistical decision policy; it does not establish absence of an operationally important difference. At τ=0.07 and E=120, repeated source-only choices were: {freq}.',
      103:f'On CADEC, the corresponding PubMedBERT-minus-DistilBERT cluster interval is [{elo:+.4f}, {ehi:+.4f}]. Whether it excludes zero and whether it is contained in a chosen practical margin are separate questions. Neither property establishes clinical relevance or irrelevance. Agreement within the declared illustrative statistical tolerance is reported alongside, rather than substituted for, measured workload and achieved sensitivity.',
      109:'Table 7 expands δ sensitivity to ten values at τ=0.07 and E=120 J/1,000. It reports the actual selected arms rather than applying an uncalibrated-model tie to every recalibration variant. The machine-readable supplement also crosses δ with additional τ/E cells. The observed switch is a property of these confidence intervals and candidates, not an independent clinical determination of δ.',
      113:'The mapping audit changes exactly two sentence labels under the segment-containment alternative and produces small point-metric differences. We retain that limited sentence-label diagnostic. The original audit did not recompute every revised cluster-bootstrap selection under Rule B, so the stronger claim that every selection cell is invariant is withdrawn. A different sentence splitter, discontinuous-entity containment convention or annotation scheme could produce a larger change.',
      114:'Post-level max-pooling preserves transformer dominance over the classical tier, but not the complete four-model sentence-level ordering: LightGBM’s original post AUROC is 0.8280 versus logistic regression’s 0.8115. PubMedBERT and DistilBERT reach 0.9589 and 0.9422. At 88.7% post prevalence these are complementary diagnostics; they are not interchangeable with sentence-level calibration.',
      117:'The evaluated policy can decline to select a model when calibration uncertainty and scoped energy requirements leave no feasible candidate. This is a useful reporting behaviour in this study, rather than a new property of constrained optimisation in general. A decision-maker could obtain more representative validation data, consider another model, revise an illustrative requirement with stakeholder input, or provide more computational resources; none is guaranteed to resolve infeasibility.',
      118:'Per-arm upper-bound filtering is more conservative than point-estimate filtering, but its performance depends on clustering, interval construction, binning and available sample size. The revision reports those choices and their sensitivity. It does not turn a bootstrap confidence limit into a finite-sample guarantee of calibration under deployment shift, nor provide simultaneous confidence coverage after searching multiple candidates.',
      120:'On the evaluated corpora, isotonic arms have lower external ECE point estimates than their corresponding temperature-scaled arms. A single temperature adjusts sharpness uniformly, whereas isotonic regression offers a flexible monotone calibration map. These observations do not establish that the linear model’s calibration curve is non-monotonic, that one method universally repairs shift, or that the effect will persist under another prevalence or annotation process.',
      123:'Existing multi-objective and resource-constrained approaches already optimise performance with deployment costs [65–67]; energy-aware successive-halving approaches additionally account for energy during hyperparameter search [68]. ECC-MS addresses a narrower decision: choosing among measured ADR configurations with declared calibration uncertainty and a practical AUROC tolerance. The comparison concerns the decision protocol, not a claim to outperform those optimisation algorithms. Encoder–LLM evaluation is contextual [69,70], and the cited generative ADR work is not used as a numerically comparable baseline.',
      124:'The external calibration results show method- and split-dependent behaviour. Isotonic recalibration improves the observed ECE estimates here, while uncalibrated DistilBERT exceeds the illustrative external 0.07 point limit on its multi-split mean. This empirical pattern supports examining calibration after transfer; it does not establish a transferable architecture ranking or guaranteed negligible-cost repair of distribution shift.',
      126:'A deployment team would first elicit acceptable missed-signal rates, review workload, latency and scoped energy costs with analysts, then validate candidate policies on representative target data. The present τ/E/δ grid illustrates decision sensitivity and is not an elicited deployment specification. A choice can be economical within the measured domain while its whole-system energy remains unknown.',
      129:'Energy figures describe the tested hardware, precision, operating point and measured domains. The original GPU board measurements omit host tokenisation and other system components; CPU RAPL package measurements omit whole-system consumption too. No GPU-equipped Linux host or wall-power meter is available for this revision. Separately measured Linux tokenizer energy is reported as a component diagnostic, not as a measured Colab-system total. Compact/quantized encoder and batch/length results remain pending until the supplied runs are returned.',
      131:'Cluster-percentile intervals and repeated grouped splits reduce specific evaluation defects but do not remove selection optimism, multiple-comparison concerns or limited domain coverage. Source audit labels are used to select the primary arm, so source performance is apparent. CADEC is independent of that primary selection, but ceases to be independent for the separately reported external-assisted variant. Future work requires another held-out target or nested external evaluation before using target labels to optimise deployment choices. No heuristic minimum-detectable-difference or generic clinical-equivalence guarantee is retained.',
      132:'Subgroup calibration was not evaluated. Source reviews cover psychiatric medications; CADEC is drug- and platform-imbalanced. Both corpora are English and use different annotation units. Additional therapeutic areas, languages, platforms, ADR prevalence, review lengths, clinical contexts and annotation policies can change discrimination, calibration, workload and energy. Counts alone do not establish subgroup fairness. Generalisation to those settings requires separate evaluation.',
      135:'ECC-MS operationalises a finite-catalogue selection policy using explicit source-calibration uncertainty, scoped measured energy and a declared practical AUROC tolerance. The revision provides mathematical definitions, whole-review/post resampling, a complete constraint grid, measured workload diagnostics and repeated selection across available grouped splits. External calibration remains a separately measured property rather than a guaranteed constraint under shift.',
      136:'Within the two evaluated English corpora and original model catalogue, some illustrative constraints admit no arm, recalibration choices alter probability reliability, and energy-sensitive choices depend on the tolerance and available resources. These findings motivate target-specific validation; they are not claims about all pharmacovigilance datasets, clinical safety or universal superiority of isotonic regression.',
      137:'Future work should elicit operational tolerances independently with pharmacovigilance analysts, add a second untouched target, assess prospective review workload and subgroup calibration, expand therapeutic and linguistic coverage, and measure whole-system energy with appropriate instrumentation. Pending compact/quantized and operating-point benchmark outputs must be integrated before this working draft is treated as a completed hardware revision.'
    }
    text.update(integrated_text(summary,primary))
    text[48]+=' For budget screening, the three recalibration variants inherit the measured base-model raw-text-to-uncalibrated-probability pipeline cost as a proxy. Their calibrated-pipeline totals were not separately measured, so budget feasibility is conditional on this disclosed component-cost proxy.'
    text[51]+=' The bootstrap is conditional on the fitted classifier and recalibrators: it resamples evaluation clusters without refitting them. The separate grouped-split analyses describe partition/training variation.'
    text[62]+=' In the present measurements, Jscope(a) is the base-model pipeline component-cost proxy assigned to each recalibration variant; calibration-transform and unmeasured host/system energy are not silently set to zero.'
    text[81]+=' Circles denote uncalibrated arms, triangles temperature scaling and squares isotonic recalibration.'
    text[42]=text[42].replace('and uses the available FBGEMM backend on the measured host','using a supported CPU quantisation backend; the original log does not separately record the selected backend')
    source_metrics=primary['domains']['test']['metrics'];external_metrics=primary['domains']['cadec']['metrics']
    present_models=[m for m in COLORS if m+' + Isotonic' in source_metrics]
    iso_external='; '.join(f"{m}: {external_metrics[m+' + Isotonic']['ece']:.4f}" for m in present_models)
    text[77]=f"Recalibration alters probability reliability differently across arms. The largest source isotonic ECE point estimate is {max(source_metrics[m+' + Isotonic']['ece'] for m in present_models):.4f}. External isotonic estimates are {iso_external}. These estimates and their uncertainty depend on the stated rank-bin and tie convention; all are regenerated in this revision. Lower ECE does not imply a universal discrimination or clinical-utility improvement. Figure 5 relates the observed AUROC values to scoped energy."
    common_names=set(summary['common_catalogue']['arm_names'])
    near=[r for run in summary['runs'] for r in run['workload_pairs'] if r['arm_a'] in common_names and r['arm_b'] in common_names and r['target_sensitivity']==.9 and r.get('threshold_policy')=='retrospective_matched_sensitivity' and abs(r['delta_auroc_b_minus_a'])<=.015]
    violations=[r for r in near if abs(r['relative_flagged_review_change_pct'])>2]
    if violations:
        text[109]+=f" Across the grouped splits' source/external matched-sensitivity diagnostics, {len(violations)} of {len(near)} arm-pair/split comparisons within 0.015 AUROC differ by more than 2% in relative flagged-review volume (maximum {max(abs(r['relative_flagged_review_change_pct']) for r in near):.2f}%). This is retrospective evidence against a general workload guarantee, not an independent choice of a replacement margin."
    for i,value in text.items():Paragraph(blocks[i],doc._body).text=value
    replace_table(doc,blocks[45],['Family','Execution','Representation / fitting','Recalibration','Configurations'],[
        ['Logistic Regression','CPU','TF-IDF unigrams/bigrams, max 2,500; LR max_iter=1,000','Uncalibrated / temperature / isotonic',3],
        ['LightGBM','CPU','Same TF-IDF; 100 trees, learning rate 0.05, 31 leaves','Same three',3],
        ['DistilBERT','T4 FP16','3 epochs; AdamW 2e-5; batch 64; length 128','Same three',3],
        ['PubMedBERT','T4 FP16','Same training schedule','Same three',3],
        ['BERT-tiny','T4 FP16','prajjwal1/bert-tiny; same schedule','Same three',3],
        ['DistilBERT / BERT-tiny','CPU FP32 and dynamic INT8','Saved primary checkpoints; no retraining; four execution variants','Same three fitted separately per variant',12]])
    replace_table(doc,blocks[73],['Configuration','Counter domain','Load (W)','Sentences/s','Gross J/1,000','Net J/1,000','CV (%)','Repeats'],hardware_rows(primary))
    names=[a['name'] for a in primary['catalogue']]
    rows=[]
    for name in names:
        a=primary['domains']['test']['metrics'][name];b=primary['domains']['cadec']['metrics'][name];model,method=name.split(' + ')
        rows.append([model,method,f"{a['auroc']:.4f}",f"{a['auprc']:.4f}",f"{a['ece']:.4f} [{a['ece_ci'][0]:.4f}, {a['ece_ci'][1]:.4f}]",f"{a['brier']:.4f}",f"{b['auroc']:.4f}",f"{b['ece']:.4f}"])
    replace_table(doc,blocks[79],['Model','Recalibration','AUROC','AUPRC','ECE [cluster 95% CI]','Brier','CADEC AUROC','CADEC ECE'],rows)
    rows=[]
    for name in [n for n in names if n in summary['common_catalogue']['arm_names']]:
        values=[]
        for domain,key in [('test','auroc'),('test','ece'),('cadec','auroc'),('cadec','ece')]:
            x=[r['domains'][domain]['metrics'][name][key] for r in summary['runs'] if name in r['domains'][domain]['metrics']];values.append(f'{np.mean(x):.4f} ± {np.std(x):.4f}')
        rows.append([*name.split(' + '),*values])
    replace_table(doc,blocks[86],['Model','Recalibration','AUROC (source)','ECE (source)','AUROC (CADEC)','ECE (CADEC)'],rows)
    clinical_rows=[]
    for name in [n for n in names if n in summary['common_catalogue']['arm_names']]:
        d=primary['domains']['test']['metrics'][name]['clinical_utility'];prev=d['prevalence_adjusted']
        native=next(v for v in prev.values() if abs(v['prevalence']-.3612)<1e-6)
        clinical_rows.append([name,d['sensitivity'],d['specificity'],d['lr_plus'],d['lr_minus'],prev['1.0%']['ppv'],prev['10.0%']['ppv'],native['ppv']])
    replace_table(doc,blocks[94],['Configuration','Sensitivity','Specificity','LR+','LR−','PPV 1%','PPV 10%','PPV 36.12%'],clinical_rows)
    cells=[(.03,.5),(.03,60),(.03,120),(.05,60),(.07,10),(.07,60),(.10,.5),(.10,60),(.05,120),(.07,120),(.10,120),(.10,150),(.10,200)]
    rows=[]
    for t,e in cells:
        r=next(r for r in primary['grid'] if r['tau']==t and r['energy_budget']==e)
        rows.append([t,e,r['feasible'],r['leader'] or 'Infeasible',r['selected'] or 'Infeasible'])
    replace_table(doc,blocks[105],['τ','E (scoped J/1,000)','Feasible','Source AUROC leader','ECC-MS source-only choice'],rows)
    delta=[r for r in primary['delta_sensitivity'] if r['tau']==.07 and r['energy_budget']==120]
    replace_table(doc,blocks[111],['δ','Base transformer source equivalence','External diagnostic equivalence','Source-only selected arm','External-assisted sensitivity'],[[r['delta'],r['base_transformer_source_equivalent'],r['base_transformer_external_equivalent'],r['selected'],r['external_assisted_selected']] for r in delta])
    images=[(29,'figure1_pipeline.png'),(35,'figure2_groups.png'),(59,'figure3_energy.png'),(66,'figure4_selection.png'),(80,'figure5_pareto.png'),(88,'figure6_calibration.png'),(96,'figure7_decision_curves.png'),(107,'figure8_regime.png')]
    for i,name in images:picture(doc,blocks[i],FIG/name)
    for citation in [
        '65. Karl F, Pielok T, Moosbauer J, et al. Multi-Objective Hyperparameter Optimization in Machine Learning—An Overview. ACM Trans Evol Learn Optim. 2023;3(4):1–50. doi:10.1145/3610536. https://arxiv.org/abs/2206.07438',
        '66. Elkan C. The Foundations of Cost-Sensitive Learning. Proceedings of IJCAI. 2001. https://cseweb.ucsd.edu/~elkan/rescale.pdf',
        '67. Cotter A, Gupta M, Jiang H, et al. Training Well-Generalizing Classifiers for Fairness Metrics and Other Data-Dependent Constraints. PMLR. 2019;97:1397–1405. https://proceedings.mlr.press/v97/cotter19b.html',
        '68. Geissler D, Zhou B, Suh S, Lukowicz P. Spend More to Save More (SM2): An Energy-Aware Implementation of Successive Halving for Sustainable Hyperparameter Optimization. arXiv:2412.08526. 2024. https://arxiv.org/abs/2412.08526',
        '69. Liang P, Bommasani R, Lee T, et al. Holistic Evaluation of Language Models. arXiv:2211.09110. 2022. https://arxiv.org/abs/2211.09110',
        '70. Ruan Q, Kuznetsov I, Gurevych I. Are Large Language Models Good Classifiers? A Study on Edit Intent Classification in Scientific Document Revisions. EMNLP. 2024. https://aclanthology.org/2024.emnlp-main.839/'
    ]:doc.add_paragraph(citation)
    heading=doc.add_paragraph('Revision supplement: comparison and evidence status')
    for run in heading.runs:run.bold=True;run.font.size=Pt(12)
    doc.add_paragraph('The returned hardware results are integrated. Machine-readable supplements provide the full main grid, a CPU-only expanded-budget grid, common-catalogue frequencies, δ and bin sensitivity, row-versus-cluster intervals, workload, raw traces and artifact audits. Component measurement scope and single-checkpoint CPU coverage remain explicitly disclosed. Complete-system energy remains unmeasured.')
    doc.add_paragraph('Supplementary Table S1. Positioning against established model-selection approaches.')
    placeholder=doc.add_table(rows=1,cols=1)
    replace_table(doc,placeholder._tbl,['Approach','Established criterion','ECC-MS scope'],[
        ['Multi-objective / Pareto [65]','Trade-offs across predictive and resource objectives','A finite measured catalogue with explicit feasibility and tolerance'],
        ['Cost-sensitive [66]','Predictive decision loss under error costs','Hardware joules are distinguished from missed-signal/review costs'],
        ['Resource-constrained [67]','Predictive objective subject to policy constraints','Per-arm source ECE upper bound and scoped energy; no shift guarantee'],
        ['Energy-aware search [68]','Energy-informed search/training resource allocation','Final inference-arm selection, rather than a new search scheduler'],
        ['Calibration-aware [33,40]','Probability calibration and behaviour under shift','Recalibrated arms evaluated with cluster uncertainty; external diagnostics'],
        ['Encoder / LLM context [69,70]','Task-specific multidimensional evaluation','Evaluation context, not direct ADR baseline performance evidence']])
    doc.add_paragraph('Supplementary Table S2. Same-checkpoint INT8 versus FP32 on primary seed 42. Differences are INT8 minus FP32; intervals use paired review/post resampling. Maximum probability changes are uncalibrated changes over the three evaluated partitions, not clinical risk estimates.')
    quant=quantization_evidence(primary);placeholder=doc.add_table(rows=1,cols=1)
    rows=[]
    for r in quant:
        intervals=[f"{r[d+'_delta_auroc']:+.4f} [{r[d+'_delta_ci_lo']:+.4f}, {r[d+'_delta_ci_hi']:+.4f}]" for d in ['test','cadec']]
        rows.append([r['model'],f"{r['fp32_package_j_1000']:.1f} / {r['int8_package_j_1000']:.1f}",f"{r['energy_reduction_pct']:.1f}%",*intervals,f"{r['test_fp32_ece']:.4f} / {r['test_int8_ece']:.4f}",f"{r['cadec_fp32_ece']:.4f} / {r['cadec_int8_ece']:.4f}",f"{r['maximum_uncalibrated_probability_change']:.3f}"])
    replace_table(doc,placeholder._tbl,['Family','Package J/1,000 FP32 / INT8','Saving','Source ΔAUROC [95% CI]','External ΔAUROC [95% CI]','Source ECE FP32 / INT8','External ECE FP32 / INT8','Max |Δp|'],rows)
    doc.add_paragraph('Supplementary Table S3. Tokenizer-only package measurements; the failing original PubMedBERT run and its passing follow-up are both retained.')
    tok=json.loads((RESULTS/'linux_artifact_audit.json').read_text())['tokenizer_measurements'];placeholder=doc.add_table(rows=1,cols=1)
    replace_table(doc,placeholder._tbl,['Tokenizer','Run','Package J/1,000','CV (%)','CV ≤10%'],[[r['model'],'Follow-up' if r.get('run')=='follow-up' else 'Original',f"{r['gross_j_1000']:.3f}",f"{r['cv_pct']:.2f}",r['repeatability_pass']] for r in tok])
    for name,caption in [('supplement_gpu_energy_surface.png','Supplementary Figure S1. Complete measured GPU-board batch/length surface; host and wall energy unmeasured.'),('supplement_cpu_quantization.png','Supplementary Figure S2. CPU-package energy and paired AUROC effects of dynamic INT8 on the primary checkpoints.'),('supplement_workload.png','Supplementary Figure S3. Retrospective review workload at matched sensitivity of at least 90%; an AUROC tolerance does not guarantee a workload bound.')]:
        doc.add_picture(str(FIG/name),width=Inches(6.3));doc.add_paragraph(caption)
    path=ROOT/'AIH Paper - Revised.docx';doc.save(path);return path


def reports(summary,primary):
    config=summary['provenance']['config'];pairs=pair_lookup(primary['domains']['test']['paired_auroc']);ext=pair_lookup(primary['domains']['cadec']['paired_auroc'])
    pair=('PubMedBERT + Uncalibrated','DistilBERT + Uncalibrated');ci=pairs[pair];ec=ext[pair]
    comparison='''# Positioning of ECC-MS in existing methods

| Approach | Decision object and typical criterion | Relation to this revision |
|---|---|---|
| Multi-objective/Pareto optimisation | Search over multiple quality/resource objectives; return trade-offs | ECC-MS applies explicit feasibility and a declared tolerance to a finite measured catalogue; it does not replace Pareto optimisation. [Karl et al.](https://arxiv.org/abs/2206.07438) |
| Cost-sensitive learning | Minimise decision loss from class-dependent error costs | Analyst false-negative/false-positive costs differ from hardware joules; both would be needed for a validated operational policy. [Elkan](https://cseweb.ucsd.edu/~elkan/rescale.pdf) |
| Resource-constrained learning | Optimise a predictive objective subject to policy/resource limits | Constraint formulation is established; independent validation of data-dependent constraints matters. [Cotter et al.](https://proceedings.mlr.press/v97/cotter19b.html) |
| Energy-aware model/hyperparameter search | Allocate search/training resources with energy-aware selection | SM2 adjusts successive-halving resource allocation. ECC-MS's evaluated object is the final inference catalogue, not a new search scheduler. [Geissler et al.](https://arxiv.org/abs/2412.08526) |
| Calibration-aware modelling and assessment | Fit/evaluate probability calibration, including under shift | Recalibration methods supply candidate arms; estimated ECE bounds become an explicitly limited source-corpus filter. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) |
| Multidimensional encoder/LLM evaluation | Compare task outcomes and nonaccuracy properties under a specified protocol | Context for evaluation design, not direct ADR baseline evidence. [HELM](https://arxiv.org/abs/2211.09110), [task-specific classifier comparison](https://aclanthology.org/2024.emnlp-main.839/) |

Contribution claim: an auditable application-level integration of measured-domain inference cost, cluster-aware uncertainty, a declared practical-discrimination tolerance and explicit infeasibility in ADR model selection. No first-of-kind or general optimisation-theory claim is made. CADEC remains outside primary selection; the historical external gate is separately labelled.
'''
    (REPORTS/'POSITIONING_AND_FORMULATION.md').write_text(comparison+r'''
## Mathematical formulation

For candidate arms $a\in\mathcal A$, define source AUROC $\widehat R_s(a)$, adaptive calibration error $\widehat C_s(a)$, cluster-bootstrap limits $L_{.95},U_{.95}$ and measured scoped energy $J(a)$ per 1,000 inputs.

$$\mathcal F(\tau,E)=\{a\in\mathcal A: U_{.95}(\widehat C_s(a))\leq\tau,\ J(a)\leq E\}.$$

If $\mathcal F=\varnothing$, return no choice. Otherwise $a^*\in\arg\max_{a\in\mathcal F}\widehat R_s(a)$ and define the shared-resample paired difference $\Delta_s(a^*,a)=\widehat R_s(a^*)-\widehat R_s(a)$.

$$\mathcal S_\delta=\{a\in\mathcal F: L_{.95}(\Delta_s(a^*,a))\geq-\delta,\ U_{.95}(\Delta_s(a^*,a))\leq\delta\},\qquad \widehat a\in\arg\min_{a\in\mathcal S_\delta}J(a).$$

Equal-cost ties use higher source AUROC, lower calibration upper bound, then name. Every recalibration variant is a distinct arm. Primary selection uses source audit data only. External calibration/equivalence is nonbinding diagnostic evidence; an external-assisted intersection is reported separately and forfeits independent evaluation on that external corpus. The 95% interval-inclusion rule is conservative relative to conventional TOST at alpha=0.05. Per-arm bootstrap intervals provide neither simultaneous postselection coverage nor a shift guarantee. All numerical tolerances are illustrative.
''',encoding='utf-8')
    response=f'''# AIH Round 1 — response draft and evidence ledger

Status: returned GPU and Linux component measurements are validated and integrated. Statistical reanalysis covers {len(summary['runs'])} grouped splits with a common fifteen-arm catalogue and {len(primary['catalogue'])} primary configurations including CPU precision variants. Complete-system energy remains unmeasured, and no independent clinical margin validation is claimed. The manuscript is a revision draft for author review. Original submission artifacts are preserved.

## Reviewer 1

1. **Positioning.** Added a six-family comparison with primary-source references in POSITIONING_AND_FORMULATION.md and revised Sections 2.5/5.3. The contribution is scoped as a finite-catalogue decision protocol integrating established ingredients, without an unsupported claim to a new optimisation method.

2. **Formalisation.** Revised Section 3.8 defines the candidate arms, feasible set, calibration upper-bound constraint, scoped energy constraint, AUROC leader, paired interval-inclusion criterion and final energy minimisation, including tie-breaking and the empty-set outcome.

3. **Constraint origin and sensitivity.** τ and E are explicitly illustrative, not elicited pharmacovigilance requirements. The main grid contains {len(config['tau_grid'])} × {len(config['energy_grid_j_per_1000'])} = {len(primary['grid'])} evaluated cells per split; both feasibility monotonicity properties pass. A separate primary CPU-only grid extends illustrative budgets to 4,000 J/1,000 because the larger CPU encoders exceed the main budgets. See grid.csv, cpu_primary_grid.csv and Figure 8.

4. **Equivalence margin and workload.** Expanded δ to 0.005–0.050 in steps of 0.005. With no independent expert elicitation available, δ=0.015 is retained only as an illustrative reference. The assertion that it guarantees <2% workload change is withdrawn. workload.csv and workload_pairs.csv report flagged reviews/posts at 85%,90%,95% calibration-target sensitivities, achieved evaluation sensitivity, and separate retrospective matched-sensitivity diagnostics. A data-derived mapping is not relabelled as an independent clinical margin.

5. **Statistical versus clinical relevance.** Removed “too small to matter” and similar language. The primary PubMedBERT-minus-DistilBERT 95% cluster intervals are [{ci[0]:+.4f}, {ci[1]:+.4f}] on source and [{ec[0]:+.4f}, {ec[1]:+.4f}] externally. Interval containment within an illustrative tolerance does not establish clinical equivalence or safety.

6. **External calibration scope.** Prominently state that feasibility is a source-corpus constraint and does not guarantee calibration under shift. Primary selection is source-only; the historical binding CADEC gate is now a separately labelled sensitivity. CADEC is independent for the primary source-only choice, not for the external-assisted sensitivity.

7. **Generalisability.** Revised Discussion/Conclusion limit empirical findings to the evaluated English corpora, therapeutic coverage and candidate catalogue. Additional languages, platforms, prevalences, review lengths and annotation processes require separate validation. Subgroup calibration and prospective clinical safety are not asserted.

## Reviewer 2

1. **Figures and regime-map feasibility.** Figures 7/8 are regenerated from measured prediction/selection artifacts and placed beside the correct captions. All grid cells are evaluated. Feasible counts are asserted non-decreasing in either constraint; missing cells are not painted as infeasible.

2. **Clustering, bins, repeated selection and workload.** Use whole-review/post percentile bootstrap with 2,000 shared draws; primary ECE uses 10 bins with 5/10/15/20 sensitivity. Row-percentile controls use the same interval method. Complete selection is repeated on the same fifteen configurations across all ten grouped splits, including primary seed 42; common_selection_frequencies.csv has an explicit denominator of ten. The larger primary CPU catalogue is evaluated separately, rather than pooled into those frequencies. Workload is directly counted at calibration-target sensitivities 85%, 90% and 95%, with achieved evaluation sensitivities and a separate retrospective matched-sensitivity diagnostic. The unsupported general claim that 0.015 AUROC implies less than 2% additional workload is withdrawn; no independent expert assessment is invented.

3. **Energy scope and new baselines — PARTIALLY ADDRESSED; COMPLETE-SYSTEM MEASUREMENT UNMET.** The new T4 run measures board energy during raw-text-to-probability pipeline timing for DistilBERT, PubMedBERT and compact BERT-tiny, across five batch sizes and three maximum lengths, with three repeats per setting. Host tokenisation time is included in elapsed time, but host CPU/RAM energy is unmeasured. Returned Linux runs evaluate seed-42 DistilBERT and BERT-tiny in FP32 and dynamic INT8 at batch 64/length 128, with seven raw package-energy repeats and paired accuracy/calibration comparisons. All four included pipeline measurements pass repeatability. The initial PubMedBERT tokenizer CV failure (10.81%) is retained alongside its passing follow-up (0.38%). See Table 2, Supplementary Tables S2/S3, Figures S1/S2 and LINUX_RESULTS_REVIEW.md. The CPU-only Linux host has no wall meter; separate-host tokenizer and board readings are not added into a fictitious complete-system result. Actual complete-system energy therefore remains unmeasured. CPU quantisation has one trained split, and the planned reproduction runner is supplied.

4. **Encoder–LLM context.** Added multidimensional evaluation and task-specific encoder/generative comparison context, clearly distinguished from direct ADR baseline evidence. No numerically incomparable cross-task performance claims are made.

## Required before final resubmission

The returned runs have been integrated; the author should review the revised manuscript and response, decide whether to resubmit with the explicitly unmet whole-system measurement request or obtain appropriate instrumentation. No further execution of the completed Colab/Linux component experiments is required by the current analysis. No independent expert margin assessment is claimed.
'''
    (REPORTS/'REVIEW_RESPONSE.md').write_text(response,encoding='utf-8')
    status={'statistical_reanalysis':'completed','grouped_splits_evaluated':len(summary['runs']),'grid_cells_per_split':len(primary['grid']),
            'cluster_bootstrap_repeats':config['bootstrap_repeats'],'hardware_extensions':'returned Colab and Linux component experiments validated and integrated',
            'common_candidate_arms':len(summary['common_catalogue']['arm_names']),'primary_candidate_arms':len(primary['catalogue']),
            'gpu_energy_settings':45,'cpu_pipeline_configurations':4,'cpu_pipeline_repeats':28,'cpu_tokenizer_repeats':28,
            'extra_pubmed_cpu':'excluded from paper and selection per author scope; original PubMedBERT GPU retained',
            'manuscript_results_status':'regenerated with returned GPU and CPU results','review_response_status':'regenerated with returned GPU and CPU results',
            'complete_system_energy':'not available: no GPU-equipped instrumented host / wall meter','independent_clinical_delta_validation':'not available; unsupported claim withdrawn',
            'submission_ready':False,'manuscript':'AIH Paper - Revised.docx','response':'reports/revision/REVIEW_RESPONSE.md'}
    (REPORTS/'revision_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')


def main():
    summary=json.loads((RESULTS/'revision_analysis.json').read_text());primary=next(r for r in summary['runs'] if r['seed']==42)
    make_figures(summary,primary);path=write_manuscript(summary,primary);reports(summary,primary)
    print(f'Revised draft: {path}. Returned component experiments integrated; complete-system energy remains unmeasured.')


if __name__=='__main__':main()
