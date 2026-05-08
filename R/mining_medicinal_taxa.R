# MiningMedicinalTaxa — R Wrapper
# Requires: reticulate, Python 3.10, MiningMedicinalTaxa installed in a Python virtualenv
#
# Setup (run once):
#   install.packages("reticulate")
#   source("mining_medicinal_taxa.R")
#   install_mining_medicinal_taxa()
#   # Restart R
#
# Every session:
#   source("mining_medicinal_taxa.R")
#   mmt_activate()
#   Sys.setenv(OPENAI_API_KEY = "sk-...")   # only needed for run_gpt()
#
# Output JSON files can be manually verified using the MedicinalTaxonVerifier app:
#   https://huggingface.co/spaces/alrichardbollans/MedicinalTaxonVerifier
#
# NOTE: the Verifier app crashes on large JSON payloads. Keep output JSONs under
# ~4000 tokens. For run_gpt() this is controlled via `context_window_k` — use 3
# or 4 when you intend to feed the output to the Verifier.

if (!requireNamespace("reticulate", quietly = TRUE)) {
  stop("Install reticulate: install.packages('reticulate')")
}
library(reticulate)


# ── Setup ─────────────────────────────────────────────────────────────────

#' Install MiningMedicinalTaxa Python package into a virtualenv
#' @param envname Name of the virtualenv (default "mining-med-taxa")
#' @param python Path to Python 3.10 executable. If NULL, installs Python 3.10
#'   automatically via reticulate.
install_mining_medicinal_taxa <- function(envname = "mining-med-taxa", python = NULL) {
  # Install Python 3.10 if no path given
  if (is.null(python)) {
    installed <- reticulate::install_python("3.10:latest")
    python <- installed
  }

  reticulate::virtualenv_create(envname, python = python)

  # Windows: enable long paths for git (repo has deeply nested files)
  if (.Platform$OS.type == "windows") {
    system("git config --global core.longpaths true")
  }

  # 1. wcvpy
  virtualenv_install(envname, packages = c(
    "wcvpy @ git+https://github.com/alrichardbollans/wcvpy.git@main"
  ))

  # 2. MiningMedicinalTaxa without deps (setup.py pins cause conflicts)
  virtualenv_install(envname, packages = c(
    "MiningMedicinalTaxa @ git+https://github.com/alrichardbollans/MiningMedicinalTaxa.git"
  ), pip_options = "--no-deps")

  # 3. Dependencies matching setup.py + langchain-openai
  virtualenv_install(envname, packages = c(
    "langchain==0.3.22",
    "langchain-core==0.3.83",
    "langchain-openai==0.3.12",
    "pydantic",
    "openpyxl",
    "unidecode",
    "transformers",
    "nltk",
    "scipy",
    "scikit-learn",
    "peft",
    "datasets"
  ))

  message("Done. Restart R, then run: mmt_activate('", envname, "')")
}

#' Activate the MiningMedicinalTaxa virtualenv
#'
#' Renamed from `activate()` to avoid clashes with `tibble::activate`,
#' `tidygraph::activate`, and similar generics.
#'
#' @param envname Name of the virtualenv (default "mining-med-taxa")
mmt_activate <- function(envname = "mining-med-taxa") {
  reticulate::use_virtualenv(envname, required = TRUE)
}


# ── SciBERT ───────────────────────────────────────────────────────────────

#' Load SciBERT models
#'
#' Downloads and loads the fine-tuned SciBERT NER (and optionally RE) models.
#' First run downloads ~400MB from HuggingFace (cached for subsequent runs).
#'
#' @return Models object to pass to run_scibert()
load_scibert_models <- function() {
  message("Loading SciBERT models (first run downloads ~400MB, then cached)...")
  scibert <- import("SciBert.running_scibert")
  models <- scibert$load_scibert()
  message("Models loaded.")
  models
}

#' Run SciBERT extraction on a text file
#'
#' Extracts plant/fungus names and optionally relations (medical conditions,
#' medicinal effects) using a fine-tuned SciBERT model. Runs locally, no API
#' key needed.
#'
#' @param txt_file Path to .txt file
#' @param models SciBERT models object from load_scibert_models()
#' @param output_json Path to save output JSON (compatible with MedicinalTaxonVerifier).
#'   If NULL, no JSON is saved. Note: the Verifier app crashes on JSONs larger
#'   than ~4000 tokens of source text.
#' @param run_re If TRUE, also extract relations — medical conditions and
#'   medicinal effects (slower). If FALSE, NER only (names). Default FALSE.
#' @param clean_names If TRUE, filter extracted names using WCVP taxonomy
#'   knowledge. Removes non-scientific names (e.g. vernacular names, place names).
#'   Default TRUE.
#' @return Data frame with scientific_name, medical_conditions, medicinal_effects
run_scibert <- function(txt_file, models, output_json = NULL, run_re = FALSE, clean_names = TRUE) {
  scibert <- import("SciBert.running_scibert")

  output <- scibert$query_scibert(models, txt_file, json_dump = output_json, run_re = run_re)

  if (clean_names) {
    evaluating <- import("LLM_models.evaluating")
    output <- evaluating$clean_model_annotations_using_taxonomy_knowledge(output)
  }

  .taxa_to_df(output)
}


# ── GPT ───────────────────────────────────────────────────────────────────

#' Run GPT extraction on a text file
#'
#' Extracts plant/fungus names, medical conditions and medicinal effects using
#' OpenAI GPT with structured output. Requires an API key.
#'
#' @param txt_file Path to .txt file
#' @param api_key OpenAI API key. If NULL, reads from the OPENAI_API_KEY env var.
#'   Set with Sys.setenv(OPENAI_API_KEY = "sk-...") or pass directly. When passed
#'   directly, the key is set for the duration of this call only and the previous
#'   value (if any) is restored on exit.
#' @param output_json Path to save output JSON (compatible with MedicinalTaxonVerifier).
#'   If NULL, no JSON is saved. Note: the Verifier app crashes on JSONs larger
#'   than ~4000 tokens of source text — keep `context_window_k` small (3–4) when
#'   producing JSONs you intend to verify.
#' @param context_window_k Context window in thousands of tokens. Controls chunk
#'   size — smaller values = more chunks = better recall but slower and more
#'   API calls. Default 4 (~4000 tokens per chunk; Verifier-compatible). Use 10
#'   for faster runs when you don't need to verify the output.
#' @param model OpenAI model name (default "gpt-4o-2024-08-06")
#' @param clean_names If TRUE, filter extracted names using WCVP taxonomy
#'   knowledge. Removes non-scientific names. Default TRUE.
#' @return Data frame with scientific_name, medical_conditions, medicinal_effects
run_gpt <- function(txt_file, api_key = NULL, output_json = NULL, context_window_k = 4,
                    model = "gpt-4o-2024-08-06", clean_names = TRUE) {

  # Resolve API key. If passed directly, set it only for this call.
  if (!is.null(api_key)) {
    old <- Sys.getenv("OPENAI_API_KEY", unset = NA)
    Sys.setenv(OPENAI_API_KEY = api_key)
    on.exit({
      if (is.na(old)) Sys.unsetenv("OPENAI_API_KEY") else Sys.setenv(OPENAI_API_KEY = old)
    }, add = TRUE)
  } else if (Sys.getenv("OPENAI_API_KEY") == "") {
    stop("No API key found. Either pass api_key or set Sys.setenv(OPENAI_API_KEY = 'sk-...')")
  }

  langchain <- import("langchain_openai")
  running   <- import("LLM_models.running_models")

  gpt_model <- langchain$ChatOpenAI(model = model, temperature = 0)
  context_window <- running$get_input_size_limit(as.integer(context_window_k))

  output <- running$query_a_model(gpt_model, txt_file, context_window,
                                  json_dump = output_json, single_chunk = FALSE)

  if (clean_names) {
    evaluating <- import("LLM_models.evaluating")
    output <- evaluating$clean_model_annotations_using_taxonomy_knowledge(output)
  }

  .taxa_to_df(output)
}


# ── Pretty print ──────────────────────────────────────────────────────────

#' Print extraction results
#' @param taxa_df Data frame from run_scibert() or run_gpt()
#' @param title Label for the output
print_taxa <- function(taxa_df, title = "Results") {
  n <- nrow(taxa_df)
  cat(sprintf("\n%s\n  %s - %d taxa found\n%s\n", strrep("=", 60), title, n, strrep("=", 60)))
  for (i in seq_len(n)) {
    row <- taxa_df[i, ]
    cat(sprintf("\n  [%d] %s\n", i, row$scientific_name))
    cond <- if (is.na(row$medical_conditions) || row$medical_conditions == "") "\u2014" else row$medical_conditions
    eff <- if (is.na(row$medicinal_effects) || row$medicinal_effects == "") "\u2014" else row$medicinal_effects
    cat(sprintf("      Conditions: %s\n", cond))
    cat(sprintf("      Effects:    %s\n", eff))
  }
  cat(sprintf("\n%s\n\n", strrep("=", 60)))
}


# ── Internal ──────────────────────────────────────────────────────────────

#' Convert Python TaxaData to R data frame
#' @keywords internal
.taxa_to_df <- function(taxa_data) {
  taxa <- taxa_data$taxa
  if (is.null(taxa) || length(taxa) == 0) {
    return(data.frame(scientific_name = character(),
                      medical_conditions = character(),
                      medicinal_effects = character(),
                      stringsAsFactors = FALSE))
  }

  data.frame(
    scientific_name = vapply(taxa, function(t) {
      if (is.null(t$scientific_name)) NA_character_ else t$scientific_name
    }, character(1)),
    medical_conditions = vapply(taxa, function(t) {
      mc <- t$medical_conditions
      if (is.null(mc)) NA_character_ else paste(mc, collapse = "; ")
    }, character(1)),
    medicinal_effects = vapply(taxa, function(t) {
      me <- t$medicinal_effects
      if (is.null(me)) NA_character_ else paste(me, collapse = "; ")
    }, character(1)),
    stringsAsFactors = FALSE
  )
}
