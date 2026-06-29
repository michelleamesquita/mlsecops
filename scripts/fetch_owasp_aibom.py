"""
scripts/fetch_owasp_aibom.py — Baixa AIBOM do OWASP AIBOM Generator
para modelos hospedados no HuggingFace.

OWASP AIBOM Generator: https://genai.owasp.org/resource/owasp-aibom-generator/

Uso:
    python scripts/fetch_owasp_aibom.py \
        --model-id microsoft/codebert-base \
        --output results/sbom_hf_model.cdx.json
"""

import argparse
import json
import logging
import pathlib
import sys
import urllib.error
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AIBOM_API = "https://aibom.owasp.org/api/generate"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch OWASP AIBOM for a HuggingFace model"
    )
    parser.add_argument("--model-id", required=True,
                        help="HuggingFace model ID (e.g. microsoft/codebert-base)")
    parser.add_argument("--output", default="results/sbom_hf_model.cdx.json",
                        help="Output path (default: results/sbom_hf_model.cdx.json)")
    args = parser.parse_args()

    url = f"{AIBOM_API}?model={args.model_id}&format=cyclonedx"
    log.info(f"Fetching OWASP AIBOM for: {args.model_id}")
    log.info(f"  URL: {url}")

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(data, f, indent=2)

        n_components = len(data.get("components", []))
        log.info(f"  OWASP AIBOM saved: {out} ({n_components} components)")

    except urllib.error.HTTPError as e:
        log.warning(f"  OWASP AIBOM Generator returned HTTP {e.code}: {e.reason}")
        log.warning("  Skipping HF model AIBOM — pipeline continues.")
        sys.exit(0)
    except urllib.error.URLError as e:
        log.warning(f"  OWASP AIBOM Generator unreachable: {e.reason}")
        log.warning("  Skipping HF model AIBOM — pipeline continues.")
        sys.exit(0)
    except Exception as e:
        log.warning(f"  Unexpected error fetching AIBOM: {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()
