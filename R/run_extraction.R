# ── MiningMedicinalTaxa — Main Script ────────────────────────────────────
# Run this script to extract plant names and medicinal properties from text.
# Output JSON can be verified at:
#   https://huggingface.co/spaces/alrichardbollans/MedicinalTaxonVerifier
#
# IMPORTANT: the Verifier app crashes on large JSON payloads. Keep individual
# output JSONs ~5000 tokens. For GPT extraction, this is controlled via
# `context_window_k`.

# ── 1. First time only ──────────────────────────────────────────────────
# install.packages("reticulate")
# Sys.setenv(WORKON_HOME = "C:/venvs")     # optional: custom virtualenv location
# source("R/mining_medicinal_taxa.R")
# install_mining_medicinal_taxa()
# # This might take a while. Restart R after install.

# ── 2. Load (every session) ─────────────────────────────────────────────
# Sys.setenv(WORKON_HOME = "C:/venvs")
source("R/mining_medicinal_taxa.R")
mmt_activate()

# Sys.setenv(OPENAI_API_KEY = "sk-...")    # only needed for GPT below

# ── 3. Input ────────────────────────────────────────────────────────────
txt_file <- "5096942.txt"

# ── 4. SciBERT (local, no API key needed) ───────────────────────────────
# Loads ~400MB on first run (cached afterwards).
models <- load_scibert_models()

scibert_results <- run_scibert(
  txt_file,
  models,
  output_json = "scibert_output.json",
  run_re      = TRUE,    # TRUE = also extract relations; FALSE = NER only
  clean_names = TRUE     # filter using WCVP taxonomy
)
print_taxa(scibert_results, "SciBERT")


# ── 5. GPT (requires API key) ───────────────────────────────────────────
# Uncomment the whole block to run GPT extraction.
#
# gpt_results <- run_gpt(
#   txt_file,
#   output_json      = "gpt_output.json",
#   context_window_k = 5,
#   model            = "gpt-4o-2024-08-06",
#   clean_names      = TRUE
# )
# print_taxa(gpt_results, "GPT-4o")

# ── 6. Verify ───────────────────────────────────────────────────────────
# Upload scibert_output.json or gpt_output.json to:
#   https://huggingface.co/spaces/alrichardbollans/MedicinalTaxonVerifier
