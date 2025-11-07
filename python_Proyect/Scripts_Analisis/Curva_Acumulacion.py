import pandas as pd
import openpyxl

# --- Paso 1. Asegurar formato de fecha ---
Registros['FECHA'] = pd.to_datetime(Registros['FECHA'])

# --- Paso 2. Crear rangos semanales ---
Registros['RANGO_FECHA'] = Registros['FECHA'].dt.to_period('W')

# --- Paso 3. Crear tabla de abundancia ---
# Agrupamos por especie y rango, sumando el número de individuos
tabla_abundancia = (
    Registros
    .groupby(['ESPECIE', 'RANGO_FECHA'])['INDIVIDUOS']
    .sum()
    .unstack(fill_value=0)   # Filas = especies, columnas = rangos
)

# --- Paso 4. Exportar a Excel ---
ruta_salida = 'Tabla_Abundancia_Semanal.xlsx'
with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
    tabla_abundancia.to_excel(writer, sheet_name='Abundancia_Semanal')

print('✅ Tabla de abundancia creada y guardada en:', ruta_salida)
print('\\nVista previa:')
print(tabla_abundancia.head())


#--------------------Calculo de Estimadores EstimateS---------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
estimadores_estimates_like.py

Replica (aproximadamente) la salida de EstimateS (Diversity Statistics)
para una tabla de abundancia (filas=especies, columnas=muestras/semanas).
Genera una hoja Excel con la tabla completa similar a EstimateS.

Ajustes:
- Permutaciones/runs: 100 (por defecto)
- Ruta de input: Tabla_Abundancia_Semanal.xlsx (hoja: Abundancia_Semanal)
- Ruta de salida: D:\CORPONOR 2025\Backet\python_Proyect\Resultados\Estimadores_Especie.xlsx
"""

import os
import numpy as np
import pandas as pd
from math import comb
from scipy.optimize import curve_fit
from skbio.diversity.alpha import shannon, simpson, fisher_alpha
from openpyxl import load_workbook

# ---------- Parámetros ----------
RUNS = 100  # número de permutaciones (tal como pediste)
input_xlsx = r"Tabla_Abundancia_Semanal.xlsx"  # archivo generado por tu paso anterior
input_sheet = "Abundancia_Semanal"
output_folder = r"D:\CORPONOR 2025\Backet\python_Proyect\Resultados"
os.makedirs(output_folder, exist_ok=True)
output_xlsx = os.path.join(output_folder, "Estimadores_Especie.xlsx")

# ---------- Funciones auxiliares ----------

def chao1_estimator(S_obs, f1, f2):
    """Chao1 point estimator with small-sample correction if f2==0"""
    if f2 == 0:
        # correction suggested in literature
        return S_obs + (f1 * (f1 - 1)) / (2 * (f2 + 1))
    else:
        return S_obs + (f1 * f1) / (2.0 * f2)

def chao1_variance(f1, f2):
    """Analytical variance for Chao1 (Chao 1987). Returns variance (not SD)."""
    # Use standard approximate formula for Var(Chao1)
    if f2 == 0:
        # if f2 == 0 variance formula unstable; return NaN so we'll fallback to bootstrap SD
        return np.nan
    try:
        term1 = f2 * (( (f1 / f2) ** 4 ) / 4.0)
        term2 = (f1 ** 3) / (4.0 * (f2 ** 2))
        var = term1 + term2
        return var
    except Exception:
        return np.nan

def ace_estimator_from_abundances(abundances, threshold=10):
    """
    ACE estimator for a pooled abundance vector (list/array of counts per species).
    Returns S_ace (point) and C_ace, gamma_sq (internal).
    """
    abund = np.array(abundances, dtype=int)
    S_abund = np.sum(abund > 0)
    # rare species: abund <= threshold and >0
    rare_mask = (abund > 0) & (abund <= threshold)
    abund_rare = abund[rare_mask]
    S_rare = len(abund_rare)
    N_rare = int(np.sum(abund_rare))
    f1 = int(np.sum(abund_rare == 1))
    # compute C_ACE
    if N_rare == 0:
        C_ace = 1.0
        gamma_sq = 0.0
    else:
        C_ace = 1.0 - (f1 / N_rare)
        # compute gamma^2
        # sum i(i-1) f_i  where f_i = number of rare species with abundance i
        fi = {}
        for v in abund_rare:
            fi[v] = fi.get(v, 0) + 1
        numerator = 0.0
        for i, count in fi.items():
            numerator += i * (i - 1) * count
        if C_ace == 0 or N_rare <= 1:
            gamma_sq = 0.0
        else:
            gamma_sq = max((S_rare * numerator) / (C_ace * N_rare * (N_rare - 1.0)) - 1.0, 0.0)
    # ACE formula
    if C_ace == 0:
        S_ace = S_abund  # degenerate
    else:
        S_abund_total = S_abund
        S_ace = S_abund_total + (S_rare / C_ace) + (f1 / C_ace) * gamma_sq
    return S_ace, C_ace, gamma_sq

def jackknife1_samplebased(S_obs, Q1, t):
    """Jackknife 1 for sample-based data, formula using T=t samples."""
    if t == 0:
        return S_obs
    return S_obs + Q1 * ( (t - 1) / t )

def jackknife2_samplebased(S_obs, Q1, Q2, t):
    if t <= 1:
        return S_obs
    return S_obs + Q1 * ( (2*t - 3) / t ) - Q2 * ( ((t - 2)**2) / (t * (t - 1)) )

def coleman_rarefaction_expected(N_total, abundances, t_samples_equiv_individuals):
    """
    Coleman rarefaction (approx): expected number of species in m pooled samples.
    EstimateS implements Coleman (1981) sample-based rarefaction; approximate with
    individual-based: expected species among m individuals:
      S(m) = sum_i (1 - comb(N - n_i, m) / comb(N, m))
    Note: comb may be large; use logs if needed. We'll use scipy.comb via math.comb for ints.
    """
    # here we assume t_samples_equiv_individuals is an integer m (number of individuals)
    m = int(round(t_samples_equiv_individuals))
    N = int(np.sum(abundances))
    if m <= 0 or N <= 0:
        return 0.0
    exp_s = 0.0
    for n_i in abundances:
        n_i = int(n_i)
        if n_i == 0:
            continue
        if N - n_i < m:
            # then term = 1 - 0 = 1
            exp_s += 1.0
        else:
            # compute comb ratio safely using logs or with fallback to multiplicative
            # Use multiplicative expression for comb ratio: C(N-n_i, m)/C(N, m) = prod_{j=0..m-1} (N-n_i - j)/(N - j)
            num = 1.0
            denom = 1.0
            prod = 1.0
            for j in range(m):
                prod *= (N - n_i - j) / (N - j)
            exp_s += (1.0 - prod)
    return exp_s

def fit_mm_model(x, y):
    """
    Fit Michaelis-Menten S(t) = Smax * t / (B + t) using non-linear least squares.
    Returns Smax (asymptote) and B.
    """
    def mm(t, Smax, B):
        return (Smax * t) / (B + t)
    try:
        p0 = [max(y)*1.2, np.median(x)+1.0]
        popt, pcov = curve_fit(mm, x, y, p0=p0, maxfev=10000)
        Smax, B = popt
        return Smax, B
    except Exception:
        return np.nan, np.nan

# ---------- Carga de datos ----------
print("Cargando tabla de abundancia desde:", input_xlsx)
df_ab = pd.read_excel(input_xlsx, sheet_name=input_sheet, index_col=0)  # index_col=0 => primera columna = especies
# Ensure integer
df_ab = df_ab.fillna(0).astype(int)
species = df_ab.index.tolist()
samples = df_ab.columns.tolist()
T = len(samples)
N_total_full = int(df_ab.values.sum())

print(f"N especies: {len(species)}, N muestras (T): {T}, N individuos totales: {N_total_full}")

# Prepare result storage: we'll build a list of dicts for each t
results = []

# For reproducibility
rng = np.random.default_rng(12345)

# ---------- Loop over t = 1..T (sample-based accumulation) ----------
for t in range(1, T+1):
    # computed individuals (EstimateS convention for sample-based abundance): (t/T) * N_total
    individuals_computed = (t / T) * N_total_full

    # We'll collect per-run statistics in lists
    per_run = {
        'S_est_run': [], 'Sobs_run': [], 'singletons_run': [], 'doubletons_run': [],
        'uniques_run': [], 'duplicates_run': [], 'ACE_run': [], 'ICE_run': [],
        'Chao1_run': [], 'Chao2_run': [], 'Jack1_run': [], 'Jack2_run': [], 'Bootstrap_run': [],
        'MMRuns_run': [], 'Cole_run': [], 'Alpha_run': [], 'Shannon_run': [], 'ShannonExp_run': [], 'SimpsonInv_run': []
    }

    # Also compute analytic S(est) for the sample-based pooled samples? EstimateS provides analytical MaoTau for rarefaction,
    # but we will approximate S_est analytical by averaging over runs (EstimateS earlier versions did MaoTau analytical).
    S_est_analytic_values = []

    for run in range(RUNS):
        # permute columns order and take first t columns
        perm = rng.permutation(T)
        chosen_cols = [samples[i] for i in perm[:t]]
        pooled = df_ab[chosen_cols].sum(axis=1).astype(int)  # pooled abundances per species in first t samples

        # Sobs
        Sobs = int((pooled > 0).sum())

        # singletons f1, doubletons f2 (from pooled abundances)
        f1 = int((pooled == 1).sum())
        f2 = int((pooled == 2).sum())

        # uniques (Q1) and duplicates (Q2) in sample-based sense: species occurring in exactly 1 sample among the t samples (incidence)
        # For that compute presence/absence across chosen_cols
        sub = (df_ab[chosen_cols] > 0).astype(int)
        incidence_counts = sub.sum(axis=1)  # number of samples (among t) where species occurs
        Q1 = int((incidence_counts == 1).sum())
        Q2 = int((incidence_counts == 2).sum())

        # ACE (from pooled abundances)
        S_ace, C_ace, gamma_sq = ace_estimator_from_abundances(pooled.values, threshold=10)

        # Chao1 (abundance-based)
        if f2 == 0:
            # Use bias-corrected variant
            ch1 = Sobs + (f1 * (f1 - 1)) / (2.0 * (f2 + 1))
        else:
            ch1 = Sobs + (f1 * f1) / (2.0 * f2)

        # Chao2 (sample-based) using Q1,Q2
        if Q2 == 0:
            chao2 = Sobs + (Q1 * (Q1 - 1)) / (2.0 * (Q2 + 1))
        else:
            chao2 = Sobs + (Q1 * Q1) / (2.0 * Q2)

        # Jackknife sample-based using t samples
        jack1 = jackknife1_samplebased(Sobs, Q1, t)
        jack2 = jackknife2_samplebased(Sobs, Q1, Q2, t)

        # Bootstrap (sample-based): resample with replacement t columns and take Sobs in that resample
        # We'll perform a small internal bootstrap (single replicate per run) to get a bootstrap richness
        resample_idx = rng.integers(0, t, size=t)  # indices into chosen_cols (0..t-1)
        sampled_cols = [chosen_cols[i] for i in resample_idx]
        pooled_boot = df_ab[sampled_cols].sum(axis=1)
        bootstrap_S = int((pooled_boot > 0).sum())

        # MM (Michaelis-Menten) - we approximate by fitting Sobs vs m across run-levels only if we have enough points.
        # Here we just compute Sobs for this t as MM run value placeholder
        mmrun_val = float(Sobs)

        # Coleman rarefaction (approx individual-based) using individuals_computed rounded
        cole_val = coleman_rarefaction_expected(N_total_full, df_ab.sum(axis=1).values, individuals_computed)

        # Diversity indices using pooled abundances (skbio uses counts per sample; works with pooled species counts)
        abund_vector = pooled.values.astype(float)
        # remove zeros for fisher_alpha which requires positive counts
        abund_nonzero = abund_vector[abund_vector > 0]
        try:
            alpha_val = float(fisher_alpha(abund_nonzero)) if len(abund_nonzero) > 0 else np.nan
        except Exception:
            alpha_val = np.nan
        try:
            sh = float(shannon(abund_vector, base=np.e))  # natural log
            sh_exp = np.exp(sh) if not np.isnan(sh) else np.nan
        except Exception:
            sh = np.nan
            sh_exp = np.nan
        try:
            sim = float(simpson(abund_vector))
            # EstimateS reports Simpson inverse sometimes; compute 1 / D where D = sum p_i^2
            sim_inv = (1.0 / sim) if sim > 0 else np.nan
        except Exception:
            sim_inv = np.nan

        # Save per-run
        per_run['S_est_run'].append(float(Sobs))  # placeholder: using Sobs per run (EstimateS uses analytical MaoTau for rarefaction)
        per_run['Sobs_run'].append(Sobs)
        per_run['singletons_run'].append(f1)
        per_run['doubletons_run'].append(f2)
        per_run['uniques_run'].append(Q1)
        per_run['duplicates_run'].append(Q2)
        per_run['ACE_run'].append(S_ace)
        per_run['ICE_run'].append(np.nan)  # ICE requires incidence coverage estimator implementation
        per_run['Chao1_run'].append(ch1)
        per_run['Chao2_run'].append(chao2)
        per_run['Jack1_run'].append(jack1)
        per_run['Jack2_run'].append(jack2)
        per_run['Bootstrap_run'].append(bootstrap_S)
        per_run['MMRuns_run'].append(mmrun_val)
        per_run['Cole_run'].append(cole_val)
        per_run['Alpha_run'].append(alpha_val)
        per_run['Shannon_run'].append(sh)
        per_run['ShannonExp_run'].append(sh_exp)
        per_run['SimpsonInv_run'].append(sim_inv)

        # for analytic S(est) placeholder, append Sobs (we'll compute mean/SD across runs below)
        S_est_analytic_values.append(Sobs)

    # Aggregate per t: means and SDs
    def mean_sd(arr):
        a = np.array(arr, dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) if len(a) > 1 else np.nan)

    Sest_mean, Sest_sd = mean_sd(per_run['S_est_run'])
    Sobs_mean, Sobs_sd = mean_sd(per_run['Sobs_run'])
    single_mean, single_sd = mean_sd(per_run['singletons_run'])
    double_mean, double_sd = mean_sd(per_run['doubletons_run'])
    uniques_mean, uniques_sd = mean_sd(per_run['uniques_run'])
    dup_mean, dup_sd = mean_sd(per_run['duplicates_run'])
    ACE_mean, ACE_sd = mean_sd(per_run['ACE_run'])
    Chao1_mean, Chao1_sd_run = mean_sd(per_run['Chao1_run'])
    Chao2_mean, Chao2_sd_run = mean_sd(per_run['Chao2_run'])
    Jack1_mean, Jack1_sd = mean_sd(per_run['Jack1_run'])
    Jack2_mean, Jack2_sd = mean_sd(per_run['Jack2_run'])
    Bootstrap_mean, Bootstrap_sd = mean_sd(per_run['Bootstrap_run'])
    MMRuns_mean, MMMs_mean = mean_sd(per_run['MMRuns_run'])
    Cole_mean, Cole_sd = mean_sd(per_run['Cole_run'])
    Alpha_mean, Alpha_sd = mean_sd(per_run['Alpha_run'])
    Shannon_mean, Shannon_sd = mean_sd(per_run['Shannon_run'])
    ShannonExp_mean, ShannonExp_sd = mean_sd(per_run['ShannonExp_run'])
    SimpsonInv_mean, SimpsonInv_sd = mean_sd(per_run['SimpsonInv_run'])

    # Analytical Chao1 SD if possible: compute using the pooled full-sample f1,f2 (use full data pooled at t=T? We'll estimate per-run average f1,f2)
    # We'll use the average f1,f2 across runs for variance formula
    avg_f1 = np.nanmean(per_run['singletons_run'])
    avg_f2 = np.nanmean(per_run['doubletons_run'])
    # compute analytic var and sd for Chao1 if possible
    if np.isnan(avg_f2) or avg_f2 == 0:
        chao1_var_analytic = np.nan
        chao1_sd_analytic = np.nan
    else:
        try:
            # Use accepted var formula approximation (Chao 1987)
            f1_ = avg_f1
            f2_ = avg_f2
            chao1_var_analytic = (f2_ * (( (f1_ / f2_) ** 4 ) / 4.0) + ( (f1_ ** 3) / (4.0 * (f2_ ** 2)) ))
            chao1_sd_analytic = np.sqrt(chao1_var_analytic) if chao1_var_analytic >= 0 else np.nan
        except Exception:
            chao1_var_analytic = np.nan
            chao1_sd_analytic = np.nan

    # 95% CI for S(est) and for Chao1 (we use mean +/- 1.96 * SD where SD is the run-based SD,
    # and for Chao1 if analytic sd available use it)
    Sest_ci_lower = Sest_mean - 1.96 * Sest_sd if not np.isnan(Sest_sd) else np.nan
    Sest_ci_upper = Sest_mean + 1.96 * Sest_sd if not np.isnan(Sest_sd) else np.nan

    Chao1_ci_lower = Chao1_mean - 1.96 * (chao1_sd_analytic if not np.isnan(chao1_sd_analytic) else Chao1_sd_run)
    Chao1_ci_upper = Chao1_mean + 1.96 * (chao1_sd_analytic if not np.isnan(chao1_sd_analytic) else Chao1_sd_run)

    # Chao2 SD/CI approximate using run-based SD
    Chao2_ci_lower = Chao2_mean - 1.96 * Chao2_sd_run
    Chao2_ci_upper = Chao2_mean + 1.96 * Chao2_sd_run

    # Compose output row matching EstimateS header style (subset; many columns)
    row = {
        "Samples": t,
        "Individuals (computed)": individuals_computed,
        "S(est)": Sest_mean,
        "S(est) 95% CI Lower Bound": Sest_ci_lower,
        "S(est) 95% CI Upper Bound": Sest_ci_upper,
        "S(est) SD": Sest_sd,
        "S Mean (runs)": Sobs_mean,
        "Singletons Mean": single_mean,
        "Singletons SD (runs)": single_sd,
        "Doubletons Mean": double_mean,
        "Doubletons SD (runs)": double_sd,
        "Uniques Mean": uniques_mean,
        "Uniques SD (runs)": uniques_sd,
        "Duplicates Mean": dup_mean,
        "Duplicates SD (runs)": dup_sd,
        "ACE Mean": ACE_mean,
        "ACE SD (runs)": ACE_sd,
        "ICE Mean": np.nan,  # placeholder
        "ICE SD (runs)": np.nan,
        "Chao 1 Mean": Chao1_mean,
        "Chao 1 95% CI Lower Bound": Chao1_ci_lower,
        "Chao 1 95% CI Upper Bound": Chao1_ci_upper,
        "Chao 1 SD (analytical)": chao1_sd_analytic if not np.isnan(chao1_sd_analytic) else Chao1_sd_run,
        "Chao 2 Mean": Chao2_mean,
        "Chao 2 95% CI Lower Bound": Chao2_ci_lower,
        "Chao 2 95% CI Upper Bound": Chao2_ci_upper,
        "Chao 2 SD (analytical/run)": Chao2_sd_run,
        "Jack 1 Mean": Jack1_mean,
        "Jack 1 SD (runs)": Jack1_sd,
        "Jack 2 Mean": Jack2_mean,
        "Jack 2 SD (runs)": Jack2_sd,
        "Bootstrap Mean": Bootstrap_mean,
        "Bootstrap SD (runs)": Bootstrap_sd,
        "MMRuns Mean": MMRuns_mean,
        "MMMeans (1 run)": MMMs_mean,
        "Cole Rarefaction": Cole_mean,
        "Cole SD (analytical)": Cole_sd,
        "Alpha Mean": Alpha_mean,
        "Alpha SD (analytical/run)": Alpha_sd,
        "Shannon Mean": Shannon_mean,
        "Shannon SD (runs)": Shannon_sd,
        "Shannon Exponential Mean": ShannonExp_mean,
        "Shannon Exponential SD (runs)": ShannonExp_sd,
        "Simpson (Inverse) Mean": SimpsonInv_mean,
        "Simpson (Inverse) SD (runs)": SimpsonInv_sd
    }

    results.append(row)

# ---------- Convertir resultados a DataFrame y guardar ----------
df_res = pd.DataFrame(results)

# Orden de columnas preferido (intentar replicar EstimateS)
cols_order = [
    "Samples", "Individuals (computed)", "S(est)", "S(est) 95% CI Lower Bound", "S(est) 95% CI Upper Bound", "S(est) SD",
    "S Mean (runs)", "Singletons Mean", "Singletons SD (runs)", "Doubletons Mean", "Doubletons SD (runs)",
    "Uniques Mean", "Uniques SD (runs)", "Duplicates Mean", "Duplicates SD (runs)",
    "ACE Mean", "ACE SD (runs)", "ICE Mean", "ICE SD (runs)",
    "Chao 1 Mean", "Chao 1 95% CI Lower Bound", "Chao 1 95% CI Upper Bound", "Chao 1 SD (analytical)",
    "Chao 2 Mean", "Chao 2 95% CI Lower Bound", "Chao 2 95% CI Upper Bound", "Chao 2 SD (analytical/run)",
    "Jack 1 Mean", "Jack 1 SD (runs)", "Jack 2 Mean", "Jack 2 SD (runs)",
    "Bootstrap Mean", "Bootstrap SD (runs)",
    "MMRuns Mean", "MMMeans (1 run)", "Cole Rarefaction", "Cole SD (analytical)",
    "Alpha Mean", "Alpha SD (analytical/run)",
    "Shannon Mean", "Shannon SD (runs)", "Shannon Exponential Mean", "Shannon Exponential SD (runs)",
    "Simpson (Inverse) Mean", "Simpson (Inverse) SD (runs)"
]
# Keep only available columns in this order
cols_present = [c for c in cols_order if c in df_res.columns]
df_res = df_res[cols_present]

# Save to Excel
df_res.to_excel(output_xlsx, index=False)
print("✅ Estimadores calculados y guardados en:", output_xlsx)




