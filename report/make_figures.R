# NeblinIA-Speech report figures. tidyverse / ggplot2. No em dashes anywhere.
suppressMessages({library(ggplot2); library(dplyr); library(scales); library(forcats); library(stringr)})

outdir <- "figures"; dir.create(outdir, showWarnings = FALSE)

# niebla (fog) palette: cool slate to warm alarm
col_good <- "#2a9d8f"; col_mod <- "#e9c46a"; col_hard <- "#e76f51"
ink <- "#1d2733"; mist <- "#5b6b7a"; paper <- "#fbfcfd"
base <- theme_minimal(base_size = 13) +
  theme(
    plot.background = element_rect(fill = paper, color = NA),
    panel.background = element_rect(fill = paper, color = NA),
    plot.title = element_text(face = "bold", size = 16, color = ink),
    plot.subtitle = element_text(size = 11, color = mist, margin = margin(b = 10)),
    plot.caption = element_text(size = 9, color = mist, hjust = 0),
    axis.title = element_text(color = mist, size = 11),
    axis.text = element_text(color = ink),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#e7ecf0"),
    legend.position = "top", legend.title = element_text(color = mist, size = 10),
    plot.margin = margin(16, 20, 12, 16)
  )

# ---- Figure 1: per-language fair WER (the bimodal story) ----
lang <- tribble(
  ~lang, ~wer,
  "spa",18.81,"zor",39.23,"zoh",49.62,"tlp",53.12,"amu",56.38,"trq",60.02,
  "ztp",61.75,"vmp",62.34,"ncf",65.63,"vmj",66.44,"vmc",66.50,"ztn",67.26,
  "ztu",69.19,"tcf",74.82,"vmz",75.89,"zpv",76.06,"pmq",76.59,"nhn",77.15,
  "chq",77.31,"nhg",79.67,"mig",86.17,"xti",88.78,"nhq",100.16,"zts",106.43
) |>
  mutate(tier = case_when(wer < 45 ~ "usable (< 45)", wer < 75 ~ "moderate (45 to 75)", TRUE ~ "failing (> 75)"),
         tier = factor(tier, levels = c("usable (< 45)","moderate (45 to 75)","failing (> 75)")),
         lang = fct_reorder(lang, wer))

f1 <- ggplot(lang, aes(wer, lang, fill = tier)) +
  geom_col(width = 0.72) +
  geom_vline(xintercept = 58.99, linetype = "dashed", color = ink, linewidth = 0.5) +
  annotate("text", x = 58.99, y = 1.2, label = "average 58.99", hjust = -0.06, vjust = 0,
           size = 3.4, color = ink, fontface = "bold") +
  geom_text(aes(label = sprintf("%.0f", wer)), hjust = -0.18, size = 3.1, color = mist) +
  scale_fill_manual(values = c("usable (< 45)" = col_good, "moderate (45 to 75)" = col_mod, "failing (> 75)" = col_hard)) +
  scale_x_continuous(limits = c(0, 118), expand = expansion(mult = c(0, 0.02))) +
  labs(title = "Per-language word error rate is bimodal",
       subtitle = str_wrap("Fair faster-whisper evaluation, NeblinIA-Speech preview-0.1. Spanish and zor are near usable; a tail of hard, data-starved languages caps the average.", 95),
       x = "Word error rate", y = NULL, fill = NULL,
       caption = "nhq has only 258 training segments. Lower is better.") + base
ggsave(file.path(outdir, "fig1_per_language.png"), f1, width = 9, height = 7.2, dpi = 200)

# ---- Figure 2: GSPO reinforcement-learning trajectory ----
rl <- tribble(
  ~step, ~cer,
  0,0.544, 25,0.501, 50,0.456, 75,0.464, 100,0.475, 125,0.4375, 150,0.422, 175,0.4435, 200,0.443
)
bestpt <- rl |> slice_min(cer, n = 1)
f2 <- ggplot(rl, aes(step, cer)) +
  geom_line(color = mist, linewidth = 0.9) +
  geom_point(color = ink, size = 2.2) +
  geom_hline(yintercept = 0.544, linetype = "dotted", color = mist) +
  annotate("text", x = 4, y = 0.544, label = "SFT base 0.544", vjust = -0.7, hjust = 0, size = 3.3, color = mist) +
  geom_point(data = bestpt, color = col_good, size = 4) +
  annotate("text", x = bestpt$step, y = bestpt$cer, label = "best 0.422", vjust = 1.9, size = 3.5, color = col_good, fontface = "bold") +
  scale_y_continuous(labels = label_number(accuracy = 0.01), expand = expansion(mult = c(0.10, 0.05))) +
  labs(title = "Reinforcement learning closes the gap supervised learning cannot",
       subtitle = "Held-out greedy dev CER during GSPO on the broad base. This is autoregressive, the honest metric.",
       x = "GSPO step", y = "Dev CER (greedy, autoregressive)",
       caption = "Lower is better. Group of 8, verifiable reward, KL to a frozen anchor.") + base
ggsave(file.path(outdir, "fig2_rl_trajectory.png"), f2, width = 9, height = 5.4, dpi = 200)

# ---- Figure 3: looping vs WER across configurations (data and RL move toward origin) ----
cfg <- tribble(
  ~name, ~data_k, ~wer, ~loop, ~method,
  "p05: 22.5k SFT", 22.5, 139, 26.3, "SFT only",
  "balanced 47.5k SFT", 47.5, 123, 23.9, "SFT only",
  "broad: 97k SFT", 97, 108, 15.9, "SFT only",
  "preview-0.3: SFT + RL", 22.5, 89, 11.3, "SFT + RL"
)
f3 <- ggplot(cfg, aes(loop, wer, color = method)) +
  geom_point(aes(size = data_k), alpha = 0.9) +
  geom_text(aes(label = name), vjust = -1.1, size = 3.3, color = ink, show.legend = FALSE) +
  scale_color_manual(values = c("SFT only" = col_hard, "SFT + RL" = col_good)) +
  scale_size_continuous(range = c(4, 11), name = "train clips (k)") +
  scale_x_continuous(limits = c(8, 30)) + scale_y_continuous(limits = c(80, 150)) +
  labs(title = "More data lowers looping, reinforcement learning lowers it further",
       subtitle = "Autoregressive dev triage. Each point is a configuration; toward the lower left is better.",
       x = "Loop rate (percent of clips that repeat)", y = "Word error rate", color = NULL,
       caption = "Loop rate and WER are tightly coupled. Data scale and RL both push toward the origin.") + base
ggsave(file.path(outdir, "fig3_loop_vs_wer.png"), f3, width = 9, height = 5.8, dpi = 200)

# ---- Figure 4: the teacher-forced trap ----
trap <- tribble(
  ~model, ~metric, ~wer,
  "all-module SFT (p05)", "teacher-forced", 64,
  "all-module SFT (p05)", "real autoregressive", 139,
  "balanced-broad SFT", "teacher-forced", 58.8,
  "balanced-broad SFT", "real autoregressive", 123
) |> mutate(metric = factor(metric, levels = c("teacher-forced","real autoregressive")))
f4 <- ggplot(trap, aes(model, wer, fill = metric)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.62) +
  geom_text(aes(label = sprintf("%.0f", wer)), position = position_dodge(width = 0.7), vjust = -0.4, size = 3.6, color = ink) +
  scale_fill_manual(values = c("teacher-forced" = mist, "real autoregressive" = col_hard)) +
  scale_y_continuous(limits = c(0, 155), expand = expansion(mult = c(0, 0.04))) +
  labs(title = "Why teacher-forced metrics cannot be trusted here",
       subtitle = str_wrap("The optimistic teacher-forced WER improves while the real free-running WER is far worse. Checkpoint selection must use the autoregressive number.", 95),
       x = NULL, y = "Word error rate", fill = NULL,
       caption = "Both models looked good on teacher-forced eval and failed on autoregressive decoding.") + base
ggsave(file.path(outdir, "fig4_tf_trap.png"), f4, width = 9, height = 5.4, dpi = 200)

cat("figures written to", outdir, "\n")
