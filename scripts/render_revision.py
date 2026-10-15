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
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch,FancyBboxPatch
from docx import Document
from docx.shared import Inches,Pt
from docx.text.paragraph import Paragraph
from docx.enum.style import WD_STYLE_TYPE
from revision_analysis import ROOT,RESULTS,REPORTS,pair_lookup

FIG=REPORTS/'figures'
COLORS={'Logistic Regression':'#2463aa','LightGBM':'#19876d','DistilBERT':'#e48c22','PubMedBERT':'#9455a5'}
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
        ('Original Colab measurements',['NVIDIA T4 board power only','Pre-tokenised resident padded batches → forward pass','Host tokenisation, CPU/RAM and whole-system power are excluded']),
        ('Requested extensions',['Colab: raw-text timing, batch × length surface, compact encoder','Linux: same-checkpoint FP32/INT8 and tokenizer package energy','Hardware runs pending; no wall-power meter is available'])],
        'Different hosts and measured domains cannot be summed and described as a measured complete system. Original energy figures remain explicitly scope-limited.')
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
    handles=[Patch(color=c,label=m) for m,c in COLORS.items()];fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=9)
    fig.suptitle('Discrimination versus measured-domain energy (primary grouped split)');fig.tight_layout(rect=[0,.07,1,1]);save_figure(fig,FIG/'figure5_pareto.png');plt.close(fig)
    models=[m for m in COLORS if f'{m} + Uncalibrated' in primary['domains']['test']['metrics']]
    fig,axes=plt.subplots(1,2,figsize=(8.3,4.8))
    for ax,domain in zip(axes,['test','cadec']):
        for j,method in enumerate(['Uncalibrated','TempScale','Isotonic']):
            vals=[[r['domains'][domain]['metrics'][f'{m} + {method}']['ece'] for r in summary['runs'] if f'{m} + {method}' in r['domains'][domain]['metrics']] for m in models]
            ax.bar(np.arange(len(models))+(j-1)*.24,[np.mean(v) for v in vals],.24,yerr=[np.std(v) for v in vals],label=method,capsize=3,color=['#9caabe','#39a8db','#2355a1'][j])
        ax.axhline(.07,color='#bd3232',ls='--',label='Illustrative tau = 0.07');ax.set_xticks(range(len(models)),['LR','LGBM','Distil','PubMed']);ax.set(ylabel='Adaptive ECE (mean ± split SD)',title='PsyTAR audit' if domain=='test' else 'CADEC');ax.legend(fontsize=9.5)
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
    source_metrics=primary['domains']['test']['metrics'];external_metrics=primary['domains']['cadec']['metrics']
    iso_external='; '.join(f"{m}: {external_metrics[m+' + Isotonic']['ece']:.4f}" for m in COLORS)
    text[77]=f"Recalibration alters probability reliability differently across arms. The largest source isotonic ECE point estimate is {max(source_metrics[m+' + Isotonic']['ece'] for m in COLORS):.4f}. External isotonic estimates are {iso_external}. These estimates and their uncertainty depend on the stated rank-bin and tie convention; all are regenerated in this revision. Lower ECE does not imply a universal discrimination or clinical-utility improvement. Figure 5 relates the observed AUROC values to scoped energy."
    near=[r for run in summary['runs'] for r in run['workload_pairs'] if r['target_sensitivity']==.9 and r.get('threshold_policy')=='retrospective_matched_sensitivity' and abs(r['delta_auroc_b_minus_a'])<=.015]
    violations=[r for r in near if abs(r['relative_flagged_review_change_pct'])>2]
    if violations:
        text[109]+=f" Across the grouped splits' source/external matched-sensitivity diagnostics, {len(violations)} of {len(near)} arm-pair/split comparisons within 0.015 AUROC differ by more than 2% in relative flagged-review volume (maximum {max(abs(r['relative_flagged_review_change_pct']) for r in near):.2f}%). This is retrospective evidence against a general workload guarantee, not an independent choice of a replacement margin."
    for i,value in text.items():Paragraph(blocks[i],doc._body).text=value
    # Table 2 retains the original measured values with corrected scope notes.
    names=[a['name'] for a in primary['catalogue']]
    rows=[]
    for name in names:
        a=primary['domains']['test']['metrics'][name];b=primary['domains']['cadec']['metrics'][name];model,method=name.split(' + ')
        rows.append([model,method,f"{a['auroc']:.4f}",f"{a['auprc']:.4f}",f"{a['ece']:.4f} [{a['ece_ci'][0]:.4f}, {a['ece_ci'][1]:.4f}]",f"{a['brier']:.4f}",f"{b['auroc']:.4f}",f"{b['ece']:.4f}"])
    replace_table(doc,blocks[79],['Model','Recalibration','AUROC','AUPRC','ECE [cluster 95% CI]','Brier','CADEC AUROC','CADEC ECE'],rows)
    rows=[]
    for name in names:
        values=[]
        for domain,key in [('test','auroc'),('test','ece'),('cadec','auroc'),('cadec','ece')]:
            x=[r['domains'][domain]['metrics'][name][key] for r in summary['runs'] if name in r['domains'][domain]['metrics']];values.append(f'{np.mean(x):.4f} ± {np.std(x):.4f}')
        rows.append([*name.split(' + '),*values])
    replace_table(doc,blocks[86],['Model','Recalibration','AUROC (source)','ECE (source)','AUROC (CADEC)','ECE (CADEC)'],rows)
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
    doc.add_paragraph('The added positioning comparison and reviewer responses are supplied in reports/revision. Machine-readable supplements provide the complete τ/E grid, δ sensitivity, row-versus-cluster interval widths, bin sensitivity, repeated-split selection frequencies, and both workload threshold policies. Hardware extensions are pending; no new hardware result is implied by a prepared script.')
    doc.add_paragraph('Supplementary Table S1. Positioning against established model-selection approaches.')
    placeholder=doc.add_table(rows=1,cols=1)
    replace_table(doc,placeholder._tbl,['Approach','Established criterion','ECC-MS scope'],[
        ['Multi-objective / Pareto [65]','Trade-offs across predictive and resource objectives','A finite measured catalogue with explicit feasibility and tolerance'],
        ['Cost-sensitive [66]','Predictive decision loss under error costs','Hardware joules are distinguished from missed-signal/review costs'],
        ['Resource-constrained [67]','Predictive objective subject to policy constraints','Per-arm source ECE upper bound and scoped energy; no shift guarantee'],
        ['Energy-aware search [68]','Energy-informed search/training resource allocation','Final inference-arm selection, rather than a new search scheduler'],
        ['Calibration-aware [33,40]','Probability calibration and behaviour under shift','Recalibrated arms evaluated with cluster uncertainty; external diagnostics'],
        ['Encoder / LLM context [69,70]','Task-specific multidimensional evaluation','Evaluation context, not direct ADR baseline performance evidence']])
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

Status: statistical reanalysis and manuscript corrections completed for {len(summary['runs'])} available grouped splits; new hardware/compact-model runs remain pending. This is not a claim that all requested experiments have been completed. Original submission files and result artifacts are preserved.

## Reviewer 1

1. **Positioning.** Added a six-family comparison with primary-source references in POSITIONING_AND_FORMULATION.md and revised Sections 2.5/5.3. The contribution is scoped as a finite-catalogue decision protocol integrating established ingredients, without an unsupported claim to a new optimisation method.

2. **Formalisation.** Revised Section 3.8 defines the candidate arms, feasible set, calibration upper-bound constraint, scoped energy constraint, AUROC leader, paired interval-inclusion criterion and final energy minimisation, including tie-breaking and the empty-set outcome.

3. **Constraint origin and sensitivity.** τ and E are explicitly illustrative, not elicited pharmacovigilance requirements. The full grid contains {len(config['tau_grid'])} × {len(config['energy_grid_j_per_1000'])} = {len(primary['grid'])} evaluated cells per split; both feasibility monotonicity properties pass. See grid.csv and Figure 8.

4. **Equivalence margin and workload.** Expanded δ to 0.005–0.050 in steps of 0.005. With no independent expert elicitation available, δ=0.015 is retained only as an illustrative reference. The assertion that it guarantees <2% workload change is withdrawn. workload.csv and workload_pairs.csv report flagged reviews/posts at 85%,90%,95% calibration-target sensitivities, achieved evaluation sensitivity, and separate retrospective matched-sensitivity diagnostics. A data-derived mapping is not relabelled as an independent clinical margin.

5. **Statistical versus clinical relevance.** Removed “too small to matter” and similar language. The primary PubMedBERT-minus-DistilBERT 95% cluster intervals are [{ci[0]:+.4f}, {ci[1]:+.4f}] on source and [{ec[0]:+.4f}, {ec[1]:+.4f}] externally. Interval containment within an illustrative tolerance does not establish clinical equivalence or safety.

6. **External calibration scope.** Prominently state that feasibility is a source-corpus constraint and does not guarantee calibration under shift. Primary selection is source-only; the historical binding CADEC gate is now a separately labelled sensitivity. CADEC is independent for the primary source-only choice, not for the external-assisted sensitivity.

7. **Generalisability.** Revised Discussion/Conclusion limit empirical findings to the evaluated English corpora, therapeutic coverage and candidate catalogue. Additional languages, platforms, prevalences, review lengths and annotation processes require separate validation. Subgroup calibration and prospective clinical safety are not asserted.

## Reviewer 2

1. **Figures and regime-map feasibility.** Figures 7/8 are regenerated from measured prediction/selection artifacts and placed beside the correct captions. All grid cells are evaluated. Feasible counts are asserted non-decreasing in either constraint; missing cells are not painted as infeasible.

2. **Clustering, bins, repeated selection and workload.** Use whole-review/post percentile bootstrap with 2,000 shared draws; primary ECE uses 10 bins with 5/10/15/20 sensitivity. Report row-percentile controls and cluster-percentile intervals rather than imposing an invalid requirement that every clustered interval be wider. Complete selection is repeated across the three existing grouped splits. Ten-split encoder scripts are prepared, but additional split results are pending. Matched-sensitivity workload is explicitly retrospective; frozen calibration thresholds do not guarantee 90% sensitivity after transfer.

3. **Energy scope and new baselines — PARTIALLY ADDRESSED; HARDWARE RESULTS PENDING.** Corrected the original claim of common raw-text-to-probability energy scope. Original GPU values measure board forward-pass energy and omit host tokenisation; CPU values cover package energy including TF-IDF. The available Linux host is CPU-only and lacks a wall-power meter, and Colab does not expose measured host-package energy. Therefore actual complete-system energy has not been measured. Prepared Colab batch×length sweeps and compact BERT-tiny training, plus Linux same-checkpoint FP32/INT8 accuracy/package-energy and tokenizer benchmarks. Returned outputs must be integrated before asserting completion. Separate-host component figures will not be presented as a measured whole system. The complete-system portion of this request remains unmet with current instrumentation.

4. **Encoder–LLM context.** Added multidimensional evaluation and task-specific encoder/generative comparison context, clearly distinguished from direct ADR baseline evidence. No numerically incomparable cross-task performance claims are made.

## Required before final resubmission

Run the Colab notebook, then Linux benchmarks using those saved checkpoints; return both result ZIPs and rerun the analysis with explicit artifact directories. Integrate the compact/quantized baselines and energy surfaces. State the remaining whole-system instrumentation limitation explicitly. The revised DOCX is labelled a working draft while these measurements are pending. No independent expert margin assessment is claimed.
'''
    (REPORTS/'REVIEW_RESPONSE.md').write_text(response,encoding='utf-8')
    status={'statistical_reanalysis':'completed','grouped_splits_evaluated':len(summary['runs']),'grid_cells_per_split':len(primary['grid']),
            'cluster_bootstrap_repeats':config['bootstrap_repeats'],'hardware_extensions':'pending external Colab and Linux execution',
            'complete_system_energy':'not available: no GPU-equipped instrumented host / wall meter','independent_clinical_delta_validation':'not available; unsupported claim withdrawn',
            'submission_ready':False,'manuscript':'AIH Paper - Revised.docx','response':'reports/revision/REVIEW_RESPONSE.md'}
    (REPORTS/'revision_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')


def main():
    summary=json.loads((RESULTS/'revision_analysis.json').read_text());primary=next(r for r in summary['runs'] if r['seed']==42)
    make_figures(summary,primary);path=write_manuscript(summary,primary);reports(summary,primary)
    print(f'Revised working draft: {path}. Statistical supplements: {RESULTS}. Hardware runs remain pending.')


if __name__=='__main__':main()
