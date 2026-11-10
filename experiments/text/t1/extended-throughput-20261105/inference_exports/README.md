# Separate GPT-2-scale language exports

Validation-selected124/125M language weights after the registered151M-token continuation. These weights are separate from the24M audiovisual fields. Full optimizer/RNG, final states, sources, all fixed samples and the fresh-test/compute report remain one directory above. Check manifest.json for SHA256 and actual selected token exposure.

From the repository root, run `python -m exploration.scale_text_infer --checkpoint experiments/text/t1/extended-throughput-20261105/inference_exports/EXP-T1-222.pt --prompt 'The experiment shows' --device cpu`. This is raw512-context continuation, not an instruction-following factual assistant. Both architectures/seeds, including poor outputs, are retained.
