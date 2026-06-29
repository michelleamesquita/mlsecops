# MLSecOps Security Policy

**Versão**: 1.0 | **Vigência**: 2025-01 → revisão anual  
**Referências**: OWASP MLSVS 1.0 · MITRE ATLAS 2024 · NIST AI RMF 1.0 · OpenSSF MLSecOps Whitepaper 2025

---

## 1. Objetivo

Esta política define os controles de segurança obrigatórios para o ciclo de vida de modelos de ML neste repositório. Todo modelo promovido para produção **deve** passar em todos os gates bloqueantes definidos neste documento.

Gates marcados como `blocking: false` geram **WARNING** e precisam ser calibrados com baseline do domínio antes de se tornarem bloqueantes (Princípio 1 do `Agent.md`).

---

## 2. Frameworks de referência

| Framework | Aplicação |
|---|---|
| **OWASP MLSVS 1.0** | Verificação de segurança de sistemas ML |
| **MITRE ATLAS 2024** | Taxonomia de ataques adversariais |
| **NIST AI RMF 1.0** | Gestão de risco em sistemas de IA |
| **OpenSSF MLSecOps Whitepaper 2025** | Arquitetura de pipeline seguro |
| **SLSA Framework L1** | Provenance de artefatos |
| **CycloneDX ML SBOM** | Bill of materials de modelos |

---

## 3. Estágio 01 — Data Engineering & Validation

### 3.1 Integridade de dados (OWASP MLSVS V3 · MITRE AML.T0020)

| Controle | Ferramenta | Gate | Blocking |
|---|---|---|---|
| SHA256 de todos os artefatos de dados | `integrity_check.py` | Zero arquivos adulterados | **SIM** |
| Schema + tipos de coluna válidos | `ge_validate.py` | 100% expectations passing | **SIM** |
| Target sem nulos | `ge_validate.py` | null_rate = 0 | **SIM** |
| Prevalência do target entre 0.1% e 99.9% | `ge_validate.py` | Alerta se fora do range | SIM |

### 3.2 Detecção de envenenamento (MITRE AML.T0020 · AML.T0018)

| Controle | Ferramenta | Threshold | Blocking |
|---|---|---|---|
| IQR Outlier Rate | `poison_detection.py` | < 10% por feature | **NÃO** (calibrar) |
| Isolation Forest anomaly rate | `poison_detection.py` | < 2× contamination | **NÃO** (calibrar) |
| KS Test (features numéricas) | `poison_detection.py` | p-value > 0.01 | **NÃO** (calibrar) |
| Chi-squared (features categóricas) | `poison_detection.py` | p-value > 0.01 | **NÃO** (calibrar) |
| **JS Divergence no target** | `poison_detection.py` | **JSD < 0.10** | **SIM** |

> **Nota**: JS Divergence no target é bloqueante pois indica label flipping — a ameaça mais severa de envenenamento. Os outros testes são WARNING até calibração com baseline do domínio.

### 3.3 Label noise (MITRE AML.T0020)

| Controle | Ferramenta | Threshold | Blocking |
|---|---|---|---|
| Taxa de label noise (Cleanlab) | `label_noise_check.py` | < 5% | **SIM** (calibrar para domínio) |

### 3.4 Data drift

| Controle | Ferramenta | Threshold | Blocking |
|---|---|---|---|
| DataDrift (Evidently) | `drift_report.py` | Share drifted cols < 30% | **NÃO** (alerta) |
| PSI por feature | `drift_report.py` | PSI < 0.20 | **NÃO** (alerta) |

---

## 4. Estágio 02 — Experimentação Segura

### 4.1 Supply-chain (MITRE AML.T0010 · OWASP MLSVS V7)

| Controle | Ferramenta | Gate | Blocking |
|---|---|---|---|
| CVEs em dependências Python | `pip-audit` | Zero CVEs HIGH/CRITICAL | **SIM** |
| Reprodutibilidade do treino | `PYTHONHASHSEED=42` + seed fixo | Run reproduzível | **SIM** |
| Notebooks sem secrets/PII | `nbdefense` | Zero issues CRITICAL | **SIM** |

### 4.2 Integridade do modelo (MITRE AML.T0018 · AML.T0010)

| Controle | Ferramenta | Gate | Blocking |
|---|---|---|---|
| SHA256 do artefato treinado | `model_scan.py` + `integrity_check.py` | Hash computado e salvo | **SIM** |
| ModelScan (globals perigosos) | `model_scan.py` (ProtectAI) | Zero globals unsafe | **SIM** |
| Pickle opcode audit | `model_scan.py` | Zero REDUCE/GLOBAL perigosos | **SIM** |
| Magic bytes válidos | `model_scan.py` | Extensão ≡ formato real | **SIM** |
| Assinatura SLSA (Sigstore) | `cosign sign-blob` | Bundle keyless gerado | **SIM** |

### 4.3 Qualidade do modelo

| Controle | Ferramenta | Threshold | Blocking |
|---|---|---|---|
| ROC-AUC (hold-out) | `train_rf.py` | ≥ 0.75 | **SIM** (calibrar) |
| KFold CV ROC-AUC (se configurado) | `train_rf.py --kfold` | ≥ 0.75, std < 0.05 | **NÃO** (alerta) |
| ML SBOM gerado | `cyclonedx-py` | Artefato presente | **SIM** |

---

## 5. Estágio 03 — Validação & Ataques Adversariais

### 5.1 Robustez adversarial (MITRE AML.T0043 · AML.T0015)

| Controle | Ferramenta | Threshold | Blocking |
|---|---|---|---|
| FGSM / ZooAttack (ε=0.1) | `adversarial_eval.py` | acc ≥ 0.75 | **SIM** (calibrar) |
| PGD / HopSkipJump (ε=0.05) | `adversarial_eval.py` | acc ≥ 0.70 | **SIM** (calibrar) |

### 5.2 Privacidade (MITRE AML.T0056 · OWASP MLSVS V6)

| Controle | Ferramenta | Threshold | Blocking |
|---|---|---|---|
| Membership Inference advantage | `membership_inference.py` | ≤ 0.10 (10%) | **SIM** (calibrar) |

### 5.3 Integridade comportamental (MITRE AML.T0018 · AML.T0020)

| Controle | Ferramenta | Threshold | Blocking |
|---|---|---|---|
| Behavioral baseline — JSD nas predições | `model_behavioral_baseline.py` | JSD < 0.05 | **SIM** (a partir do 2º treino) |
| Behavioral baseline — Taxa de desacordo | `model_behavioral_baseline.py` | < 5% predições mudaram | **SIM** (calibrar) |

> **Como funciona**: após o primeiro treino aprovado, executa-se `--update` para salvar o fingerprint comportamental (predições sobre 1000 amostras fixas). Em todo re-treino, o pipeline compara o comportamento atual com o baseline. Se as predições mudaram significativamente sem justificativa — poisoning silencioso, backdoor, ou adulteração do modelo — o gate bloqueia.

### 5.4 Extração de modelo (MITRE AML.T0044)

| Controle | Ferramenta | Threshold | Blocking |
|---|---|---|---|
| Knockoff fidelity | `model_extraction_test.py` | < 0.90 fidelidade | **NÃO** (WARNING) → SIM em serving API |

> **Quando torna-se bloqueante**: em ambientes onde o modelo é exposto via API de inferência (serving), alta fidelidade do knockoff indica que um adversário pode reconstruir o modelo com 5000 queries. Mitigações: prediction throttling, confidence truncation, rate limiting.

### 5.3 LLM Security (quando aplicável — OWASP LLM Top 10)

| Controle | Ferramenta | Gate | Blocking |
|---|---|---|---|
| Prompt injection / Jailbreak | `promptfoo` (plugins: injection, jailbreak, harmful:privacy) | 0 falhas bloqueantes | **SIM** |
| Vulnerabilidades LLM (Garak) | `garak` (probes: dan, encoding, knownbadsignatures) | 0 issues críticos | **SIM** |

> Aplicável apenas quando o pipeline incluir modelos LLM ou endpoints de geração de texto.

---

## 6. Compliance Gate — verificação automatizada

O script `scripts/compliance_check.py` lê os relatórios em `results/` e verifica se todos os gates bloqueantes passaram. **Nenhum modelo é promovido sem o compliance gate passar.**

```bash
python scripts/compliance_check.py --policy policy.yml --results results/
```

---

## 7. Processo de override

Qualquer gate bloqueante pode ser contornado **somente** com aprovação explícita da equipe de segurança:

1. Abrir issue com label `security-override` explicando o motivo
2. Aprovação de **2 membros** do time de segurança via PR review
3. Override válido por **24 horas** (expiração automática)
4. Registro no MLflow run como `param: override_reason`

---

## 8. Princípios sobre thresholds

> *"Threshold sem baseline empírico é teatro de segurança."* — Agent.md, Princípio 1

Todos os thresholds neste documento são **pontos de partida empíricos**. Antes de tornar qualquer gate bloqueante em produção:

1. Execute o pipeline em **30+ runs** de referência para estabelecer distribuição baseline
2. Defina thresholds em percentil 5 (conservador) ou percentil 10 (balanceado) da distribuição
3. Documente a calibração no MLflow e atualize `policy.yml`
4. Domínios sensíveis (saúde, finanças, jurídico) exigem thresholds mais estritos

---

## 9. Mapeamento completo OWASP MLSVS → controles

| MLSVS | Requisito | Controle implementado |
|---|---|---|
| V1 | Threat modeling | `Agent.md` (MITRE ATLAS mapping) |
| V2 | Training data integrity | `integrity_check.py` + `poison_detection.py` |
| V3 | Data validation | `ge_validate.py` + `label_noise_check.py` |
| V4 | Training process security | `train_rf.py` (seed fixo) + `secure-experiment.yml` |
| V5 | Model robustness | `adversarial_eval.py` (ART) |
| V6 | Privacy protection | `membership_inference.py` |
| V7 | Supply chain | `pip-audit` + `model_scan.py` + Sigstore |
| V8 | Deployment security | Sigstore sign + CycloneDX SBOM |
| V9 | Monitoring | `drift_report.py` (pós-deploy) |
| V10 | Compliance | `compliance_check.py` |
