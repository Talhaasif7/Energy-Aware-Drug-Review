# Positioning of ECC-MS in existing methods

| Approach | Decision object and typical criterion | Relation to this revision |
|---|---|---|
| Multi-objective/Pareto optimisation | Search over multiple quality/resource objectives; return trade-offs | ECC-MS applies explicit feasibility and a declared tolerance to a finite measured catalogue; it does not replace Pareto optimisation. [Karl et al.](https://arxiv.org/abs/2206.07438) |
| Cost-sensitive learning | Minimise decision loss from class-dependent error costs | Analyst false-negative/false-positive costs differ from hardware joules; both would be needed for a validated operational policy. [Elkan](https://cseweb.ucsd.edu/~elkan/rescale.pdf) |
| Resource-constrained learning | Optimise a predictive objective subject to policy/resource limits | Constraint formulation is established; independent validation of data-dependent constraints matters. [Cotter et al.](https://proceedings.mlr.press/v97/cotter19b.html) |
| Energy-aware model/hyperparameter search | Allocate search/training resources with energy-aware selection | SM2 adjusts successive-halving resource allocation. ECC-MS's evaluated object is the final inference catalogue, not a new search scheduler. [Geissler et al.](https://arxiv.org/abs/2412.08526) |
| Calibration-aware modelling and assessment | Fit/evaluate probability calibration, including under shift | Recalibration methods supply candidate arms; estimated ECE bounds become an explicitly limited source-corpus filter. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) |
| Multidimensional encoder/LLM evaluation | Compare task outcomes and nonaccuracy properties under a specified protocol | Context for evaluation design, not direct ADR baseline evidence. [HELM](https://arxiv.org/abs/2211.09110), [task-specific classifier comparison](https://aclanthology.org/2024.emnlp-main.839/) |

Contribution claim: an auditable application-level integration of measured-domain inference cost, cluster-aware uncertainty, a declared practical-discrimination tolerance and explicit infeasibility in ADR model selection. No first-of-kind or general optimisation-theory claim is made. CADEC remains outside primary selection; the historical external gate is separately labelled.

## Mathematical formulation

For candidate arms $a\in\mathcal A$, define source AUROC $\widehat R_s(a)$, adaptive calibration error $\widehat C_s(a)$, cluster-bootstrap limits $L_{.95},U_{.95}$ and measured scoped energy $J(a)$ per 1,000 inputs.

$$\mathcal F(\tau,E)=\{a\in\mathcal A: U_{.95}(\widehat C_s(a))\leq\tau,\ J(a)\leq E\}.$$

If $\mathcal F=\varnothing$, return no choice. Otherwise $a^*\in\arg\max_{a\in\mathcal F}\widehat R_s(a)$ and define the shared-resample paired difference $\Delta_s(a^*,a)=\widehat R_s(a^*)-\widehat R_s(a)$.

$$\mathcal S_\delta=\{a\in\mathcal F: L_{.95}(\Delta_s(a^*,a))\geq-\delta,\ U_{.95}(\Delta_s(a^*,a))\leq\delta\},\qquad \widehat a\in\arg\min_{a\in\mathcal S_\delta}J(a).$$

Equal-cost ties use higher source AUROC, lower calibration upper bound, then name. Every recalibration variant is a distinct arm. Primary selection uses source audit data only. External calibration/equivalence is nonbinding diagnostic evidence; an external-assisted intersection is reported separately and forfeits independent evaluation on that external corpus. The 95% interval-inclusion rule is conservative relative to conventional TOST at alpha=0.05. Per-arm bootstrap intervals provide neither simultaneous postselection coverage nor a shift guarantee. All numerical tolerances are illustrative.
