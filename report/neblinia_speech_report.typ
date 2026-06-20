#set document(title: "NeblinIA-Speech: A Technical Report", author: "NeblinIA Lab")
#set page(
  paper: "a4",
  margin: (x: 2.2cm, y: 2.4cm),
  numbering: "1",
  footer: context [
    #set text(8pt, fill: luma(40%))
    NeblinIA-Speech preview-0.1
    #h(1fr)
    #counter(page).display("1 / 1", both: true)
  ],
)
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => [
  #v(0.4em) #block(text(13pt, weight: "bold", it.body)) #v(0.2em)
]
#show heading.where(level: 2): it => [
  #v(0.2em) #block(text(11pt, weight: "bold", it.body)) #v(0.1em)
]
#show raw: set text(font: "DejaVu Sans Mono", size: 9pt)
#set table(stroke: 0.4pt + luma(60%), inset: 6pt)

#align(center)[
  #v(0.5cm)
  #text(20pt, weight: "bold")[NeblinIA-Speech]
  #v(0.15cm)
  #text(13pt)[A foundational ASR system for Mexican Indigenous languages]
  #v(0.2cm)
  #text(11pt, style: "italic")[Technical Report, preview-0.1]
  #v(0.15cm)
  #text(10pt, fill: luma(35%))[NeblinIA Lab. 2026-06-20.]
  #v(0.4cm)
  #line(length: 60%, stroke: 0.5pt + luma(60%))
]

#v(0.3cm)

#block(inset: (x: 0.6cm), [
  #text(10pt, weight: "bold")[Abstract.]
  #text(10pt)[
  We build a foundational automatic speech recognition (ASR) model for 23 Mexican
  Indigenous languages plus Mexican Spanish, starting from Whisper-large-v3-turbo and
  roughly ten hours of open data per language. We report a final fair-evaluation word
  error rate (WER) of 58.99 and character error rate (CER) of 26.45 on a private,
  contamination-resistant benchmark, a reduction of 7.0 WER points over our prior best.
  The path that worked combined broad multilingual data scaling with reinforcement
  learning (GSPO). Equally important, we document the methods that did not work:
  teacher-forced supervised fine-tuning that overfits and amplifies repetition, full
  fine-tuning that is incompatible with our toolchain, label smoothing that crashes the
  model, best-of-K self-distillation that overfits, maximum-entropy frontier weighting
  that fails on bimodal data, and anti-repetition decode guards that hurt because these
  languages use grammatical reduplication. The central methodological finding is that
  teacher-forced validation metrics anti-correlate with real autoregressive quality in
  late training, so checkpoint selection must use autoregressive evaluation.
  ]
])

= Introduction

NeblinIA is a foundational research lab focused on speech technology for the Indigenous
languages of Mexico. This report covers the first preview model, an end-to-end effort to
take a general multilingual ASR backbone and adapt it to 23 low-resource Mexican languages
spanning the Oto-Manguean family (Mixtec, Zapotec, Chinantec, Amuzgo, Mazatec, Cuicatec),
Uto-Aztecan (Nahuatl), Totonacan (Totonac), and Mixe-Zoque (Zoque), among others. These
languages are polysynthetic or agglutinative, frequently tonal, and lack standardized
orthographies, which makes word error rate a harsh metric and makes the data scarcity
acutely limiting.

The stated target was an average WER of 20 with no repetition. We did not reach 20, and a
core contribution of this report is an honest account of why: at roughly ten hours per
language, a handful of hard or data-starved languages dominate the average and cannot be
fixed by modeling alone. We did reach a usable, top-ranked model, and we mapped the
solution space thoroughly enough that the remaining levers are clear.

== Goals and constraints

The lab operates under three standing constraints that shaped every decision. First, the
benchmark must be contamination-resistant and evaluated fairly, meaning every model
(ours and the baselines) is scored with the identical decoding protocol. Second, all
training data must be openly licensed. Third, the work must be reproducible, version
controlled, and documented, including the failures.

= Data

== In-domain corpus

The in-domain training and evaluation data come from the Omnilingual ASR corpus
(`facebook/omnilingual-asr-corpus`, CC BY 4.0). For the 23 target Mexican languages this
yields 22,549 training segments and 5,784 development segments after forced alignment of
the roughly 60-second source clips into segments under 30 seconds. The per-language counts
are balanced near 1,000 segments, with one exception (`nhq` at 258 segments) that proves
important later. This is the hard ceiling of the in-domain source: about ten hours per
language.

== Open transfer data

To scale beyond the in-domain ceiling we added Common Voice version 26 (CC0) for ten
related Mexican Indigenous languages already present on disk: Nahuatl variants `ncx` and
`nlv`, Zoque `zoc` (same family as our test language `zoh`), Mazatec `mau`, Cuicatec `cut`
and `cux`, Purépecha `pua`, Yaqui `yaq`, Seri `sei`, and Tarahumara `tar`. These contribute
about 75,000 additional validated clips. Common Voice clips are short single utterances, so
they need no forced alignment, and `soundfile` reads the mp3 audio directly. We kept the
checkpoint commercial-clean by excluding all CC-BY-NC sources (for example the large
OpenSLR Mixtec and Nahuatl deposits).

The combined "broad" manifest holds 97,646 clips. All audio is decoded under a single
forced `es` language token, the standard low-resource bucket trick: the token becomes a
generic "Mexican languages" marker and the model learns to map acoustics to the target
orthography directly.

#table(
  columns: (auto, auto, auto, 1fr),
  table.header[Source][License][Clips][Role],
  [Omnilingual ASR (23 MX langs)], [CC BY 4.0], [22,549], [in-domain train, plus 5,784 dev],
  [Common Voice v26 (10 related)], [CC0], [\~75,000], [transfer fuel, train only],
  [Broad combined manifest], [open], [97,646], [multistage pre-training],
)

= Benchmark and evaluation

== The fair protocol

The held-out test set (MEXA, 5,925 clips across the 23 languages plus Spanish) is private:
its reference transcripts are never published, so it functions as a contamination-resistant
answer key. Every model is scored through the same faster-whisper (CTranslate2) engine with
beam size 1, faster-whisper temperature fallback, and automatic language detection for the
non-Spanish languages. This parity is the integrity of the leaderboard. An earlier internal
result that gave our model a friendlier decode than the baselines was discarded once the
unfairness was identified.

== The teacher-forced trap

The single most important evaluation lesson of the project: the in-training validation
metric is teacher-forced (the decoder always sees the gold prefix), and it
anti-correlates with real autoregressive quality in late training. A model can show a
beautiful, monotonically improving teacher-forced WER while its free-running greedy
decoding collapses into repetition. Because of this, we built a fast autoregressive triage
that runs raw greedy decoding (single temperature, no fallback) on the development set and
reports per-language WER, CER, and a loop rate (fraction of clips where a 3-gram repeats
three or more times in a row). All checkpoint selection uses this autoregressive triage,
not the teacher-forced number.

#figure(
  image("figures/fig4_tf_trap.png", width: 92%),
  caption: [The teacher-forced trap. For two supervised runs, the optimistic teacher-forced
  WER (gray) looks good while the real autoregressive WER (orange) is far worse.],
)

= Methods

== Supervised fine-tuning

The backbone is Whisper-large-v3-turbo (809M parameters, a distilled model with a
four-layer decoder), adapted with LoRA through the Unsloth library. We extended the
adapter from the minimal attention projections to all linear modules (`q`, `k`, `v`, `out`,
`fc1`, `fc2`), at rank 64, with gradient checkpointing. The decisive hyperparameter turned
out to be not a learning rate but a stopping rule: training must stop early, near 0.9 of an
epoch, because full training overfits and amplifies repetition.

== Reinforcement learning (GSPO)

The post-training method is Group Sequence Policy Optimization (GSPO, arXiv 2507.18071),
a critic-free, group-relative policy gradient. For each clip we sample a group of eight
hypotheses, score each with a verifiable reward, compute a group-relative advantage
(reward minus group mean), and update with a sequence-level, length-normalized importance
ratio and a small Kullback-Leibler penalty to a frozen supervised anchor. The reward is a
composite of negative CER and negative WER plus explicit anti-repetition penalties for
over-generation length and n-gram repetition. Whisper consumes audio, so the loop is hand
rolled rather than borrowed from a text-only library; a key efficiency fix encodes the
audio once per clip and reuses the encoder output across the eight samples to avoid an
out-of-memory failure.

= Results

The final model, internally `preview-0.9-broadgspo` and released as NeblinIA-Speech
preview-0.1, scores as follows on the fair benchmark.

#table(
  columns: (1fr, auto, auto),
  table.header[Model][WER][CER],
  [NeblinIA-Speech preview-0.1 (broad data plus GSPO)], [*58.99*], [*26.45*],
  [Prior best (preview-0.3, attention-only SFT plus GSPO)], [66.02], [28.26],
)

The improvement is 7.0 WER points and the model evaluates quickly, with no catastrophic
looping. Per-language results are bimodal, which is the heart of the remaining problem: a
group of languages is approaching usable quality while a group of hard or data-starved
languages still fails.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  table.header[Lang][WER][Lang][WER][Lang][WER],
  [spa], [18.8], [vmj], [66.4], [nhn], [77.2],
  [zor], [39.2], [vmc], [66.5], [chq], [77.3],
  [zoh], [49.6], [ztn], [67.3], [nhg], [79.7],
  [tlp], [53.1], [ztu], [69.2], [mig], [86.2],
  [amu], [56.4], [tcf], [74.8], [xti], [88.8],
  [trq], [60.0], [vmz], [75.9], [nhq], [100.2],
  [ztp], [61.8], [zpv], [76.1], [zts], [106.4],
  [vmp], [62.3], [pmq], [76.6], [ncf], [65.6],
)

Spanish at 18.8 and Zapotec `zor` at 39.2 show that the per-language floor is genuinely low
when data exists. The languages above 85 (`mig`, `xti`, `nhq`, `zts`) are the ones that
loop and that lack the data to be fixed by the current recipe. `nhq`, with only 258 training
segments, is the clearest case of data starvation.

#figure(
  image("figures/fig1_per_language.png", width: 100%),
  caption: [Per-language WER on the fair benchmark. The distribution is bimodal: a few
  languages approach usable quality while a tail of hard or data-starved languages caps the
  average at 58.99.],
)

== What worked, measured autoregressively

The autoregressive development triage (raw greedy, 460 clips) is the metric that tracks
reality. It tells a clean story: data scale reduces looping, but reinforcement learning is
what closes the gap.

#table(
  columns: (1fr, auto, auto, auto),
  table.header[Configuration][WER][CER][Loop rate],
  [preview-0.3: attention-only SFT, then GSPO], [89], [50], [11.3%],
  [broad base: 97k data, all-module SFT, 0.9 epoch], [108], [60], [15.9%],
  [p05: 22.5k data, all-module SFT, \~4 epochs], [139], [87], [26.3%],
  [balanced-broad: 47.5k data, all-module SFT, 2 epochs], [123], [75], [23.9%],
)

Reading across: tripling the data (p05 to broad) cut the loop rate from 26.3% to 15.9% and
WER from 139 to 108, so data scale helps free-running generation. But the supervised base
still loses to the earlier GSPO model, because reinforcement learning is decisive. Applying
GSPO to the broad base drove held-out greedy dev CER from 0.544 to 0.422 and produced the
final 58.99 fair-eval result.

#figure(
  image("figures/fig3_loop_vs_wer.png", width: 96%),
  caption: [Looping and WER are tightly coupled. More data moves a configuration down and
  left (less looping, lower WER); reinforcement learning moves it further still.],
)

#figure(
  image("figures/fig2_rl_trajectory.png", width: 96%),
  caption: [GSPO on the broad base. The held-out greedy (autoregressive) dev CER falls from
  0.544 to 0.422. This is the metric that tracks real quality.],
)

= What failed

This section is the point of the report. Each item below is a real experiment that did not
work, with the reason.

== Best-of-K self-distillation (RFT)

We sampled eight hypotheses per clip and kept the lowest-CER sample as a new supervised
target (the rejection-sampling fine-tuning recipe). The model overfit its own low-diversity
samples and amplified looping rather than reducing it. The recovered checkpoint looped so
badly that a fair evaluation did not finish in 33 minutes (a clean evaluation takes about
five). Negative result, abandoned.

== All-module SFT alone

Extending the adapter to all linear modules improved the teacher-forced WER from 78 to 64,
which looked excellent. The autoregressive triage told the truth: WER 139 and a 26.3% loop
rate, worse than the baseline. More adapter capacity bought a better teacher-forced fit and
a worse free-running model, the signature of exposure bias.

== Full fine-tuning and label smoothing

Unsloth full fine-tuning of Whisper crashes at the first step with a decoder input conflict
(`cannot specify both decoder_input_ids and decoder_inputs_embeds`). Separately, and more
insidiously, label smoothing triggers the identical crash because the Hugging Face label
smoother feeds the model differently from the patched Whisper forward. We misdiagnosed this
twice (first blaming full fine-tuning, then a corrupted compile cache) before isolating
label smoothing as the true cause. The fix is to use neither with this toolchain.

== Maximum-entropy frontier weighting (MGPO)

The VibeThinker signal phase weights each clip by how close it sits to the learnable
frontier (solved about half the time). This assumes a smooth difficulty gradient. Our
difficulty distribution is bimodal: clips are either solved or total looping garbage, with
almost nothing in between. The frontier was nearly empty, so the weighting down-weighted
essentially everything and the gradient went dead. The VibeThinker recipe is built for
reasoning tasks and does not transfer to perception-bound low-resource ASR.

== Anti-repetition decode guards

We tested `no_repeat_ngram_size` and `repetition_penalty` to suppress loops at decode time.
Overall WER got worse, from 58.99 to 61.61. The reason is linguistic: these languages use
reduplication grammatically, so penalizing repeated tokens suppresses correct output. The
guards helped the single worst looper (`nhq`, 100 to 82) but hurt Spanish (18.8 to 20.7)
and reduplicating languages (`vmj`, 66 to 78). The conclusion is that "no repetition" must
be learned through reinforcement learning, which it was, and not bolted on at decode time.

== Continuation RL

Running more GSPO from the already-tuned model, with a lower learning rate and a stronger
anti-repetition reward, never beat its own starting point: dev CER stayed between 0.43 and
0.44 against a 0.414 start. The first GSPO pass extracts the available gain; more of the
same kind plateaus.

== Balanced data and full-epoch training

We rebuilt the broad manifest with Common Voice capped per language so the 23 test
languages carried half the weight rather than a quarter, then trained two full epochs. Both
choices were wrong. Full training overfit and pushed the loop rate to 23.9% and WER to 123,
and reducing the transfer data made the base weaker. This confirmed, a second time, that
the transfer data helps and that early stopping is essential.

= Engineering and infrastructure

Two engineering lessons cost real time and are worth recording.

The first is a five-hour GPU idle caused by a process-detection bug. A chain of background
watchers used `pgrep -f PATTERN` to decide whether a training run was still alive.
`pgrep -f` matches the full command line of every process, including the watcher's own
inline command and any concurrent status-check command that contained the same pattern, so
the watchers kept seeing a finished run as alive and never launched the next stage. The fix
is to detect completion through an explicit `EXIT` marker written to the log file
(`grep -q "^EXIT"`), never through process-name matching, and to trust
`nvidia-smi --query-compute-apps` for "is anything on the GPU."

The second is operational: the cloud machine is an ephemeral container whose `/tmp` is
wiped on restart, which once silently destroyed a staged launcher. All launchers, state,
and persisted models now live under a durable path, and the conversion and evaluation
tooling was hardened against three Transformers 5.5 quirks (a stub tokenizer, a
`processor_config.json` versus `preprocessor_config.json` naming change, and a `dtype`
versus `torch_dtype` config key).

= Key lessons

+ Teacher-forced validation metrics lie for this task. Select checkpoints by autoregressive
  evaluation with an explicit loop metric.
+ Data scale reduces repetition (a measured 26.3% to 15.9% loop rate at three times the
  data), but reinforcement learning is what actually closes the WER gap.
+ Supervised fine-tuning must stop early. Every full-epoch run overfit and amplified
  looping.
+ "No repetition" is a training property, not a decode-time patch, because these languages
  use grammatical reduplication.
+ Recipes designed for reasoning models (frontier weighting, self-distillation, chain of
  thought) assume structure that low-resource perceptual ASR does not have.

= Limitations and future work

The honest ceiling is data. At about ten hours per language, the six to eight hardest
languages dominate the average and cannot be modeled into shape. Average WER of 20 is not
reachable at this data scale. The model also produces no timestamps, operates on segments
under 30 seconds, and decodes everything through a single `es` bucket rather than a true
language identifier.

The documented next levers, both larger investments, are a higher-capacity backbone
(whisper-large-v3 with its 32-layer decoder, traded against slower reinforcement learning),
and targeted per-language data collection for the failing languages, including the large
untranscribed Nahuatl deposits that could be pseudo-labeled. A faster reinforcement-learning
loop using accelerated sampling would help exploration but would not raise the data ceiling.

= Conclusion

NeblinIA-Speech preview-0.1 is a real, top-ranked foundational ASR model for 23 Mexican
Indigenous languages, improving the fair-evaluation WER from 66 to 59 through a combination
of open multilingual data scaling and verifiable-reward reinforcement learning. The result
is bankable and usable, and the surrounding catalogue of what failed, and precisely why,
is intended to save the next effort from repeating the same dead ends. The languages that
remain hard are limited by data, not by the method, which sets a clear agenda for the next
round.
