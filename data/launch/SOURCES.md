# Sources for Gemini 3.7 Flash and 3.8 Flash launch benchmark tables

Collected 2026-10-02. All data comes from primary Google pages. No secondary source was used for any number.
Raw copies of every page, PDF and image used, and the two scripts that turned them into these CSVs, are kept outside this repository. In the raw HTML and text copies, the long dash character was replaced with a plain hyphen to follow the house rule. Nothing else in them was edited.

## Gemini 3.7 Flash

| URL | Date on page | Table format | Use |
|---|---|---|---|
| https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-3-7-flash/ | 13 August 2026 (datePublished 2026-08-13T17:00Z, dateModified 2026-08-13T21:09Z) | Image: `gemini-3-7-flash__evals__benchmarks-table__2x.original.png` (raw/g37_table.png), alt text "a chart displaying AI model benchmarks" | Launch announcement; full table as image, plus 5 single-benchmark bar charts and a performance to cost chart |
| https://deepmind.google/models/model-cards/gemini-3-7-flash/ | Published 13 August 2026, "Results as of August 2026" | HTML text table | Transcription source for the CSV (same rows and values as the blog image) |
| https://deepmind.google/models/evals-methodology/gemini-3-7-flash (redirects to https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_model_evaluation.pdf) | "as of August, 2026" | Methodology as PDF text; results table embedded as an image (raw/pdf37-000.png) | Per-row methodology notes in the footnote column |
| https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-7-flash | "Last updated 2026-10-01 UTC"; same table in Wayback capture of 2026-08-30 | HTML text table, 4 rows | Second, smaller Google table ("standard developer evaluation benchmarks"), appended as the last 4 CSV rows |

## Gemini 3.8 Flash

| URL | Date on page | Table format | Use |
|---|---|---|---|
| https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/ | 2 September 2026 (datePublished 2026-09-02T15:00Z, dateModified 2026-09-03T19:59Z) | Image: `gemini-3.8-flash__evals__table_light_1.original.png` (raw/g38_table.png), alt text "comparison chart showing Gemini 3.8 Flash and other models" | Launch announcement; full table as image, plus charts for DeepSWE, Vals Finance Agent v2, Harvey's Legal Agent Benchmark, HLE-Verified and several Gemini 3.8 Flash Cyber charts |
| https://deepmind.google/models/model-cards/gemini-3-8-flash/ | Published 2 September 2026, "Results as of September, 2026" | HTML text table now; at launch (Wayback 2026-09-02 15:14 UTC) the table was an image | Transcription source for the CSV (same rows and values as the blog image) |
| https://deepmind.google/models/evals-methodology/gemini-3-8-flash (redirects to https://storage.googleapis.com/deepmind-media/gemini/gemini_3-8_flash_model_evaluation.pdf) | "as of September, 2026"; revised after launch | Methodology as PDF text; results table embedded as an image (raw/pdf38-000.png) | Per-row methodology notes in the footnote column |
| https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash | "Last updated 2026-10-01 UTC"; same table in Wayback capture of 2026-09-03 | HTML text table, 7 rows | Second, smaller Google table, appended as the last 7 CSV rows |

## Checks done

- For both models, the blog table image, the model card HTML table and the evaluation PDF table image show identical rows and values (checked by eye, cell by cell).
- Launch-day versions via the Wayback Machine (read only):
  - 3.7 model card capture 2026-08-13 17:34 UTC: table identical to the current page.
  - 3.7 evaluation PDF: current file has the same digest as the 2026-08-14 capture.
  - 3.7 and 3.8 blog posts: launch-day captures (2026-08-13 18:48 UTC, 2026-09-02 15:46 UTC) reference the same table image files as today.
  - 3.8 evaluation PDF: the launch capture (2026-09-02 15:45 UTC) differs from today's file in text only. Two notes were added later: the Opus 5 DeepSWE rounding note and the LABBench2 sub-task list. The embedded table image is byte-identical.
- Missing cells (printed as a dash) are written as `not reported`.
- Scores are written without the % sign. The unit column carries the unit.

## Column conventions

- `variant`: the setting printed in the table's notes column or Gemini cell (for example "No tools", "Partial score; batch tool enabled", "agentic"). For rows matched to the ECI list, important settings taken from the methodology PDF or from Epoch's benchmark pages are added in square brackets.
- `footnote`: the table footnote (prices) and, prefixed "Methodology PDF:", the row-specific methodology note. Rows from the Cloud guides are prefixed "Second Google table".
- `comparison_scores`: the other columns of the same table, as "Model: value".
- `source_url`: launch-table rows cite the blog post (image) and the model card (text). Cloud guide rows cite the guide.
- General methodology for all launch rows: Gemini scores are pass@1 unless noted. "Single attempt" allows no majority voting or parallel test-time compute. Runs use the Gemini API (model ids gemini-3.7-flash and gemini-3.8-flash) with default sampling. Smaller benchmarks are averaged over several trials. Non-Gemini results are self-reported by their providers unless noted, at the maximum available reasoning setting.

## ECI matching notes

Epoch's own pages were checked for the version Epoch uses (epoch.ai/data/eci-documentation/data, epoch.ai/benchmarks/frontiercode, /terminal-bench, /gdpval, /osworld-2, /hle, /deepswe).

Matched:
- FrontierCode 1.1 Main to FrontierCode. Epoch tracks the 1.1 revision, Main subset, so this is an exact match.
- DeepSWE v1.1 to DeepSWE. Epoch tracks v1.1 from the Datacurve leaderboard. The Gemini scores are Google's own runs (mini-swe-agent, high thinking).
- OSWorld-2.0 to OSWorld 2.0. Google reports the partial score, maxed over 3 runs. For 3.8, batched tool calls were enabled. Epoch's default metric is binary full-task accuracy, so the numbers are not directly comparable.
- Cloud guide "DeepSWE" (version not printed) to DeepSWE.
- Cloud guide "Humanity's Last Exam (HLE)" to HLE. The setting (tools or not, subset, harness) is not stated.

Near misses, left unmatched on purpose (change these if you want a looser rule):
- Terminal-bench 2.1. ECI's "Terminal Bench" is Terminal-Bench 2.0. Version 2.1 keeps the same 89 tasks but fixes 28 of them, and scores moved up by as much as 12 points (per the Terminal-Bench 2.1 release notes).
- HLE-Verified. This is a different item set: 1,811 items, of which 1,143 are revised, and 689 uncertain items are excluded. It is not the original HLE.
- GDPVal-AA v2 (Elo). ECI's GDPval uses OpenAI's public GDPval leaderboard win rate, while this row is Artificial Analysis's Elo.
- Terminal-bench 3.0 and 4.0 (different benchmarks from Terminal-Bench 2.x), Agent's Last Exam (not HLE), SWE-Bench Pro (excluded by your rule), and Code Arena / LMArena WebDev (WebDev Arena is not on your list).

## Not extracted

- The Gemini 3.8 Flash Cyber charts in the 3.8 blog post: CyberGym pass@1; CWE-Bench / StaticBench pass@1 vs cost, with the text giving 47.2% vs 47.8% for "a leading frontier model"; real-world vulnerability discovery, with the text saying over 70%; and the Gray Swan IPI prompt injection chart. They describe the Cyber variant, or are separate charts, and are not part of the launch table. None of them is on the ECI list.
- The 3.7 blog single-benchmark bar charts and the performance to cost chart. They repeat table rows. Only the DeepSWE chart was read, and it agrees with the table (48.6% for 3.6 Flash).
