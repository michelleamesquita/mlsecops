# MLSecOps · RF Findings Classifier

Pipeline MLSecOps de ponta a ponta para classificar findings de SAST como **`is_risky`** (binário), treinando um **Random Forest** sobre o dataset `all_findings_flat.csv` (~1,2 M registros gerados por Bandit em repos Python).

Baseado no **OpenSSF MLSecOps Whitepaper (2025)**, no capítulo 18 de Sotiropoulos (*Adversarial AI*, Packt) e nos princípios do [`Agent.md`](Agent.md).

> **Scripts genéricos** — todos os controles em `scripts/` funcionam com qualquer dataset tabular e qualquer modelo sklearn serializado. Para adaptar ao seu projeto, ajuste `DATA_CSV` e `TARGET_COLUMN` nos workflows.

---

## Estrutura do projeto

```
mlsecops/
├── Agent.md                                # System prompt MLSecOps Engineer
├── all_findings_flat.csv                   # Dataset SAST Bandit (1,2 M linhas)
├── train_rf.py                             # Treino RF: early stopping + KFold + MLflow
├── requirements.txt                        # Dependências Python
├── policy.yml                              # Policy machine-readable (lida pelo compliance gate)
├── POLICY.md                               # Policy humana (OWASP MLSVS · MITRE ATLAS)
├── promptfoo.yaml                          # Red-team config LLM (threshold: 0.95)
│
├── scripts/                                # Controles MLSecOps genéricos
│   │
│   │   ── Estágio 01 · Data ──────────────────────────────────────────
│   ├── integrity_check.py                  # SHA256 + row-level (split-view) + manifesto SLSA
│   ├── ge_validate.py                      # Schema + distribuição (GE-style)
│   ├── poison_detection.py                 # KS · Chi² · JSD · Isolation Forest · Chaff Detection
│   ├── label_noise_check.py                # Cleanlab label noise
│   ├── drift_report.py                     # Evidently + PSI fallback
│   │
│   │   ── Estágio 02 · Model Training ────────────────────────────────
│   ├── model_scan.py                       # ModelScan (ProtectAI) + pickle audit
│   ├── lineage.py                          # Lineage manifest: provenance chain completa
│   ├── generate_mlbom.py                   # CycloneDX ML BOM + Model Card (1.6 spec)
│   ├── fetch_owasp_aibom.py                # OWASP AIBOM Generator (HuggingFace models)
│   │
│   │   ── Estágio 03 · Inference ──────────────────────────────────────
│   ├── model_behavioral_baseline.py        # Fingerprint comportamental do modelo
│   ├── adversarial_eval.py                 # FGSM/PGD via IBM ART (tabular + images)
│   ├── input_sanitization.py               # FeatureSqueezing + GaussianAug (defesa automática)
│   ├── binary_input_detector.py            # BinaryInputDetector: porteiro limpo/adversarial
│   ├── membership_inference.py             # Membership inference via IBM ART
│   ├── model_extraction_test.py            # Knockoff extraction test (AML.T0044)
│   │
│   │   ── Governance ────────────────────────────────────────────────
│   ├── compliance_check.py                 # InSpec para MLSecOps (lê policy.yml)
│   ├── generate_ci_data.py                 # Dataset sintético para CI/CD (fallback)
│   └── lineage.py                          # build + verify: SLSA Level 2
│
├── .github/workflows/
│   ├── main-pipeline.yml                   # ★ Orquestrador (chama os 3 estágios)
│   ├── data-validation.yml                 # Estágio 01 · Data
│   ├── secure-experiment.yml               # Estágio 02 · Model Training
│   └── adversarial-validation.yml          # Estágio 03 · Inference (jobs paralelos)
│
├── model/                                  # Artefatos (runtime)
│   ├── rf_model.pkl
│   ├── feature_names.json
│   ├── behavioral_baseline.json            # Fingerprint do modelo aprovado
│   ├── binary_detector.pkl                 # Detector de inputs adversariais
│   └── checksums.sha256
│
└── results/                                # Relatórios de cada gate (runtime)
    ├── lineage_manifest.json
    ├── adversarial_fgsm.json
    ├── adversarial_pgd.json
    ├── sanitization_fgsm.json
    ├── sanitization_pgd.json
    ├── binary_detector_report.json
    └── compliance_report.json
```

---

## Pipeline principal

O workflow `main-pipeline.yml` orquestra os 3 estágios do ciclo de vida ML em sequência — baseado na **Figura 4-1** de *Adversarial AI* (Packt): **Data → Model Training → Inference**.

```
push → main
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  01 · data-validation.yml                                           │
│                                                                     │
│  integrity_check (SHA256 + row-level chunks)                        │
│  → ge_validate → poison_detection (+ chaff)                         │
│  → label_noise_check → drift_report                                 │
│                                                                     │
│  🔒 Data Quality Gate                                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ needs: data-validation
┌──────────────────────────▼──────────────────────────────────────────┐
│  02 · secure-experiment.yml                                         │
│                                                                     │
│  pip-audit → NB scan → train_rf (seed + KFold + early stop)         │
│  → model_scan → lineage.py (build manifest)                         │
│  → Sigstore SLSA → CycloneDX ML BOM + OWASP AIBOM                   │
│                                                                     │
│  🔒 Supply-Chain Gate                                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ needs: secure-experiment
┌──────────────────────────▼──────────────────────────────────────────┐
│  03 · adversarial-validation.yml                                    │
│                                                                     │
│  setup ──┬── fgsm-eval ──┐                                          │
│          │  (FGSM +      │                                          │
│          │   sanitiz. +  ├──▶ post-eval                             │
│          │   detector)   │    (membership · extraction ·            │
│          └── pgd-eval  ──┘     poison · promptfoo · garak ·        │
│             (PGD +             compliance gate)                     │
│              sanitiz.)                                              │
│                                                                     │
│  🔒 Adversarial Gate  [FGSM e PGD rodam em paralelo]               │
└─────────────────────────────────────────────────────────────────────┘
```

Os workflows individuais continuam respondendo aos seus próprios triggers (push em dados, PRs, etc.).

---

## Ameaças cobertas (MITRE ATLAS)

| ATLAS ID | Ameaça | Estágio | Controle |
|---|---|---|---|
| AML.T0020 | Data Poisoning | Data | `integrity_check` + `poison_detection` + `label_noise_check` |
| AML.T0024g | Erode Dataset Integrity (split-view) | Data | `integrity_check --row-level` (hash por chunk) |
| AML.T0021ai | Spamming with Chaff Data | Data | `poison_detection --chaff-threshold` (near-duplicate rate) |
| AML.T0018 | Backdoor ML Model | Data + Inference | `model_scan` + `model_behavioral_baseline` |
| AML.T0010 | Supply Chain Compromise | Training | `integrity_check` (SLSA) + `lineage.py` + `pip-audit` + `model_scan` |
| AML.T0043 | Craft Adversarial Examples | Inference | `adversarial_eval` (FGSM/PGD via IBM ART) |
| AML.T0015 | Evade ML Model | Inference | `adversarial_eval` + `input_sanitization` + `binary_input_detector` |
| AML.T0056 | Membership Inference | Inference | `membership_inference` (MI advantage ≤ 10%) |
| AML.T0044 | Extract ML Model | Inference | `model_extraction_test` (knockoff fidelity ≤ 90%) |

---

## Scripts — referência rápida

Todos os scripts aceitam `--data`, `--target` e `--model` como parâmetros. Nenhum tem lógica específica do dataset SAST — funcionam para qualquer projeto ML.

### Estágio 01 · Data

| Script | O que faz | Quando falha |
|---|---|---|
| `integrity_check.py` | SHA256 em dados/modelos, manifesto SLSA. `--row-level` hasha CSV em chunks para detectar substituição parcial de linhas (*split-view poisoning*). | Arquivo adulterado, ausente ou chunk com hash diferente do baseline |
| `ge_validate.py` | Schema, tipos, nulos, cardinalidade | Coluna faltando ou target com nulos |
| `poison_detection.py` | KS · Chi² · JS Divergence · Isolation Forest · IQR · **Chaff Detection** (near-duplicates). `--chaff-threshold` define a taxa máxima tolerada. | JSD > 0.10 no target ou near-duplicate rate > threshold |
| `label_noise_check.py` | Cleanlab OOF label noise. `--fail-on-noise` torna o gate bloqueante. | noise rate > 5% com `--fail-on-noise` |
| `drift_report.py` | Evidently DataDriftPreset + PSI fallback | > 30% features drifted (alerta) |

### Estágio 02 · Model Training

| Script | O que faz | Quando falha |
|---|---|---|
| `model_scan.py` | ModelScan (ProtectAI) + pickle audit + magic bytes + SHA256 | Globals perigosos ou hash mismatch |
| `lineage.py build` | Constrói manifesto de provenance: SHA256 de dados + scripts + configs + modelo + métricas MLflow + resultados de segurança. Compatível com SLSA Level 2. | Sempre gera; verificação acontece no estágio seguinte |
| `generate_mlbom.py` | CycloneDX 1.6 ML BOM: inventário de dependências + Model Card (algoritmo, features, métricas, considerações éticas). | Sempre gera `results/sbom_ml.cdx.json` |
| `fetch_owasp_aibom.py` | Baixa AIBOM do OWASP AIBOM Generator para modelos HuggingFace. Só executa se `HF_MODEL_ID` estiver configurado. | Erro de API |

### Estágio 03 · Inference

| Script | O que faz | Quando falha |
|---|---|---|
| `lineage.py verify` | Re-hasha artefatos e compara ao manifesto — detecta substituição de modelo entre treinamento e deploy. | Hash divergente (AML.T0010 artifact substitution) |
| `model_behavioral_baseline.py` | Fingerprint de predições em set fixo; detecta drift comportamental | JSD > 0.05 ou > 5% predições mudaram |
| `adversarial_eval.py` | FGSM e PGD via IBM ART. Modo `tabular`: BoundaryAttack (único compatível com RF). Modo `images`: FGSM/PGD reais com gradiente (PyTorch). Salva `results/adversarial_{fgsm,pgd}.json`. | acc < 0.75 (FGSM) ou < 0.70 (PGD) |
| `input_sanitization.py` | Lê `adversarial_*.json`. Se o gate falhou, aplica **FeatureSqueezing** + **GaussianAugmentation** + combinação e mede recuperação de acurácia. Defesa automática sem intervenção manual. | Mesmo com defesa, acc ainda abaixo do threshold |
| `binary_input_detector.py` | Treina um classificador binário (Logistic Regression) para distinguir inputs limpos de adversariais **antes** de chegar ao RF. Encapsula com `art.defences.detector.evasion.BinaryInputDetector`. | detection rate < 0.70 ou false positive rate > 0.10 |
| `membership_inference.py` | MembershipInferenceBlackBox — privacidade dos dados de treino | advantage > 10% |
| `model_extraction_test.py` | Knockoff model via queries black-box (AML.T0044) | fidelidade > 90% (WARNING) |

### Governance

| Script | O que faz |
|---|---|
| `compliance_check.py` | Lê `policy.yml` + `results/*.json`; reporta PASS/FAIL/WARN por controle |
| `generate_ci_data.py` | Gera dataset sintético com mesmo schema do real para uso em CI quando `all_findings_flat.csv` está git-ignorado |

---

## Defesas adversariais (IBM ART)

O pipeline implementa defesa em **três camadas** sequenciais para ataques de evasão:

```
Input → [BinaryInputDetector] → adversarial? → BLOQUEADO
                              → limpo?       → [InputSanitization] → RF principal
                                                  FeatureSqueezing
                                                  GaussianAugmentation
```

| Camada | Script | Quando ativa |
|---|---|---|
| **Detecção** | `binary_input_detector.py` | Toda inferência — detecta e bloqueia |
| **Sanitização** | `input_sanitization.py` | Gate falhou — aplica defesa e re-avalia |
| **Gate** | `adversarial_eval.py` | Sempre — mede acc adversarial |

O `adversarial-validation.yml` roda FGSM e PGD **em paralelo** (jobs independentes no GitHub Actions), reduzindo ~50% do tempo do estágio 03.

---

## promptfoo (LLMs)

O `promptfoo.yaml` define testes red-team para quando o pipeline evoluir para incluir componentes LLM. **Para ML clássico (Random Forest), o step é pulado automaticamente** — só executa se `OPENAI_API_KEY` ou `LLM_ENDPOINT` estiver configurado como secret no GitHub.

Gates configurados:

| Nível | Threshold | Testes cobertos |
|---|---|---|
| Global (`defaultTest`) | score ≥ 0.90 | Todos os testes |
| `llm-rubric` individual | threshold: 0.95 | Goal hijacking, jailbreak, data leakage |

---

## Execução local

```bash
# 1. Instalar dependências
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install adversarial-robustness-toolbox

# 2. Gerar checksums baseline (arquivo inteiro + row-level por chunks)
python scripts/integrity_check.py \
    --files all_findings_flat.csv --update \
    --row-level --chunk-size 10000

# 3. Validar dados (inclui chaff detection)
python scripts/ge_validate.py      --data all_findings_flat.csv --target is_risky
python scripts/poison_detection.py --data all_findings_flat.csv --target is_risky \
    --chaff-threshold 0.02

# 4. Treinar (com early stopping + 5-fold CV)
python train_rf.py --seed 42 --experiment-name local-run --kfold 5

# 5. Construir lineage manifest (captura toda a cadeia de custódia)
python scripts/lineage.py build \
    --data  all_findings_flat.csv \
    --model model/rf_model.pkl \
    --config policy.yml requirements.txt \
    --scripts train_rf.py \
    --results-dir results/ \
    --output results/lineage_manifest.json

# 6. Salvar fingerprint comportamental (primeira vez)
python scripts/model_behavioral_baseline.py \
    --data all_findings_flat.csv --target is_risky \
    --model model/rf_model.pkl --update

# 7. Escanear modelo
python scripts/model_scan.py --model model/rf_model.pkl

# 8. Avaliação adversarial + defesas
python scripts/adversarial_eval.py \
    --data all_findings_flat.csv --target is_risky \
    --model model/rf_model.pkl --meta model/feature_names.json \
    --attack fgsm --epsilon 0.1 --n-samples 50 --min-accuracy 0.75

python scripts/input_sanitization.py \
    --data all_findings_flat.csv --target is_risky \
    --model model/rf_model.pkl --meta model/feature_names.json \
    --adv-report results/adversarial_fgsm.json \
    --output results/sanitization_fgsm.json

python scripts/binary_input_detector.py \
    --data all_findings_flat.csv --target is_risky \
    --model model/rf_model.pkl --meta model/feature_names.json \
    --n-samples 50 --output-dir results/

# 9. Compliance gate
python scripts/compliance_check.py --policy policy.yml --results results/

# 10. Ver métricas no MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Acesse http://127.0.0.1:5000
```

---

## Adaptar para outro projeto ML

Nos workflows, ajuste apenas duas variáveis:

```yaml
env:
  DATA_CSV: "meu_dataset.csv"
  TARGET_COLUMN: "minha_label"
```

Substitua o step `Train model` em `secure-experiment.yml` pelo seu script de treino. Todos os scripts de `scripts/` funcionam sem modificação.

---

## Variáveis de ambiente — `train_rf.py`

| Variável | Padrão | Descrição |
|---|---|---|
| `DATA_PATH` | `all_findings_flat.csv` | Caminho do CSV |
| `RF_N_ESTIMATORS` | `200` | Número máximo de árvores |
| `RF_MAX_DEPTH` | `15` | Profundidade máxima |
| `RF_MIN_SAMPLES_LEAF` | `10` | Min samples por folha |
| `RF_OOB_PATIENCE` | `3` | Rounds sem melhora para early stopping |
| `RF_TEST_SIZE` | `0.2` | Fração de teste |
| `GATE_MIN_ROC_AUC` | `0.75` | Threshold mínimo de ROC-AUC |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | URI do MLflow |

---

## Policy (OWASP MLSVS)

Os thresholds dos gates são definidos em [`policy.yml`](policy.yml) e documentados em [`POLICY.md`](POLICY.md). O `compliance_check.py` é o enforcement automatizado — equivalente ao Chef InSpec para ML.

> **Princípio sobre thresholds**: todos os gates usam valores empíricos como ponto de partida. Calibre com baseline do domínio antes de tornar qualquer gate bloqueante em produção (Agent.md, Princípio 1).

---

## Referências

- [OpenSSF MLSecOps Whitepaper 2025](https://openssf.org)
- [Sotiropoulos — Adversarial AI, Packt ch18](https://github.com/PacktPublishing/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies/tree/main/ch18)
- [MITRE ATLAS](https://atlas.mitre.org)
- [OWASP MLSVS](https://owasp.org/www-project-machine-learning-security-verification-standard/)
- [IBM Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [ProtectAI ModelScan](https://github.com/protectai/modelscan)
- [AISP — AI Secure Pipeline](https://github.com/empires-security/aisp)
- [Cleanlab](https://github.com/cleanlab/cleanlab)
- [Evidently AI](https://github.com/evidentlyai/evidently)
- [Sigstore / cosign](https://docs.sigstore.dev)
- [CycloneDX ML SBOM](https://cyclonedx.org/capabilities/mlbom/)
- [OWASP AIBOM Generator](https://genai.owasp.org/resource/owasp-aibom-generator/)
- [MLflow](https://mlflow.org)
- [promptfoo](https://promptfoo.dev)
- [Carlini et al., 2024 — Split-view poisoning](https://arxiv.org/abs/2312.04748)
