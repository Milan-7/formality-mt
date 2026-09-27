# Du or Sie? Formality in English→German Machine Translation

Code for the research poster *"Du or Sie? Controlling Formality in English→German Machine Translation: Neural MT vs. LLMs"* (Trends in Natural Language Processing, Trier University, SoSe 2026).

English has a single *you*; German forces a choice between formal *Sie* and informal *du*. This project measures which form MT systems choose by default, whether the choice depends on the conversational setting, and whether LLMs can be steered to the requested form by a prompt instruction.

## Research questions

1. **RQ1: Default.** Which form of address do MT systems produce when the source gives no explicit cue?
2. **RQ2: Context.** Does the choice differ between customer-service calls, phone conversations and casual chat?
3. **RQ3: Control.** Does a prompt instruction ("use *Sie*" / "use *du*") steer LLM translations, and at what cost in translation quality?

## Setup

| Component | Choice |
|---|---|
| Data | CoCoA-MT EN→DE test set (Nadejde et al., 2022): 600 segments = 3 domains × 200, each with a formal and an informal reference |
| Neural MT | `Helsinki-NLP/opus-mt-en-de` (bilingual), `facebook/nllb-200-distilled-600M` (multilingual); beam search (4 beams) |
| LLMs | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`; greedy decoding; prompts: neutral / formal / informal |
| Formality metric | Official CoCoA-MT / IWSLT 2022 phrase-matching scorer (punctuation-robust variant; strict variant also reported) + reference-free lexical detector as robustness check |
| Quality metric | chrF and BLEU (sacreBLEU); multi-reference for uncontrolled output |
| Statistics | Binomial test (default vs. 50 %), χ² test of independence (domain × formality), Cramér's V |

## Reproduce

1. Open `formality_mt.ipynb` in Google Colab (`File → Upload notebook`, or open it from GitHub).
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all` (about 30–45 minutes). All results are written to `outputs/` and downloaded as `outputs.zip`.

To recreate the poster figures and extra tables (single-reference chrF, cue-word analysis, detector agreement), run in the same environment afterwards:

```
!python poster_figures.py outputs contrastive-controlled-mt/CoCoA-MT/test/en-de
```

They are written to `outputs/poster/` (figure sizes equal the print sizes on the A1 poster).

The dataset is pinned to commit `441e23a` of [amazon-science/contrastive-controlled-mt](https://github.com/amazon-science/contrastive-controlled-mt). Library versions, GPU, prompts and decoding settings of each run are logged in `outputs/env.json`.

## Outputs

| File | Content |
|---|---|
| `outputs/hyp/*.de` | System translations, one per line (`<system>__<condition>.de`) |
| `outputs/per_sentence.csv` | Every translation with its formality labels |
| `outputs/formality_official.csv` | Label counts and percentages per system × condition × domain |
| `outputs/quality.csv` | chrF / BLEU per system × condition × domain |
| `outputs/stat_tests.csv` | Binomial and χ² tests |
| `outputs/poster/` | Poster figures (SVG/PDF/PNG) and tables from `poster_figures.py` |

## References

- Nadejde, M., Currey, A., Hsu, B., Niu, X., Federico, M., & Dinu, G. (2022). CoCoA-MT: A Dataset and Benchmark for Contrastive Controlled MT with Application to Formality. *Findings of NAACL 2022*, 616–632.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Controlling Politeness in Neural Machine Translation via Side Constraints. *NAACL-HLT 2016*, 35–40.

## Licences

CoCoA-MT data: CDLA-Sharing-1.0. The formality scorer is imported from the dataset repository (MIT-0). Models are used under their respective licences.
