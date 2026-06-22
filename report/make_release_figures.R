# Release figures for the NeblinIA-Speech model card + GitHub.
# Fig A: per-language WER vs CER dumbbell (the gap = orthographic inflation, the honest story).
# Fig B: architecture scoreboard (what worked on the held-out test).
suppressMessages({library(ggplot2); library(dplyr); library(stringr); library(scales)})

# niebla (fog) palette
col_good <- "#2a9d8f"; col_mod <- "#e9c46a"; col_hard <- "#e76f51"
ink <- "#1d2733"; mist <- "#8a99a8"; paper <- "#fbfcfd"

base <- theme_minimal(base_size = 13) +
  theme(plot.background = element_rect(fill = paper, color = NA),
        panel.background = element_rect(fill = paper, color = NA),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(color = "#e7ecf0"),
        plot.title = element_text(face = "bold", size = 16, color = ink),
        plot.subtitle = element_text(size = 11, color = mist, margin = margin(b = 10)),
        plot.caption = element_text(size = 9, color = mist, hjust = 0),
        axis.text = element_text(color = ink), axis.title = element_text(color = mist),
        legend.position = "top", legend.title = element_blank(),
        legend.text = element_text(color = ink))

# ---------- Fig A: per-language dumbbell ----------
d <- read.csv("report/data/per_language.csv")
d$label <- paste0(d$lang, "  (", d$family, ")")
d$label <- reorder(d$label, d$wer)

figA <- ggplot(d) +
  geom_segment(aes(y = label, yend = label, x = cer, xend = wer),
               color = mist, linewidth = 0.9, alpha = 0.6) +
  geom_point(aes(y = label, x = wer, color = "WER"), size = 3.1) +
  geom_point(aes(y = label, x = cer, color = "CER (content)"), size = 3.1) +
  geom_vline(xintercept = 54.4, linetype = "22", color = col_hard, alpha = 0.5) +
  geom_vline(xintercept = 24.2, linetype = "22", color = col_good, alpha = 0.6) +
  annotate("text", x = 54.4, y = 0.6, label = "overall WER 54.4", hjust = -0.03,
           vjust = 0, size = 3, color = col_hard, fontface = "bold") +
  annotate("text", x = 24.2, y = 0.6, label = "overall CER 24.2", hjust = 1.03,
           vjust = 0, size = 3, color = col_good, fontface = "bold") +
  scale_color_manual(values = c("WER" = col_hard, "CER (content)" = col_good)) +
  scale_x_continuous(limits = c(0, 110), breaks = seq(0, 100, 20)) +
  labs(title = "NeblinIA-Speech preview-1.0 (whisper-large-v3)",
       subtitle = str_wrap("Per-language error on the held-out MEXA test. The WER-CER gap is orthographic noise, not mishearing: these languages have no standard spelling, so CER is the fairer measure of content accuracy.", 95),
       x = "error rate (%, lower is better)", y = NULL,
       caption = "23 Mexican Indigenous languages + Spanish  |  private contamination-resistant benchmark, 5,925 clips") +
  base
ggsave("report/figures/release_per_language.png", figA, width = 9, height = 7.2, dpi = 200)

# ---------- Fig B: architecture scoreboard ----------
sb <- read.csv("report/data/scoreboard.csv")
sb$model <- reorder(sb$model, -sb$wer)
figB <- ggplot(sb, aes(x = model, y = wer, fill = tag)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = paste0("WER ", wer)), hjust = -0.12, size = 4, color = ink, fontface = "bold") +
  geom_text(aes(label = ifelse(is.na(cer), "", paste0("CER ", cer))), y = 2, hjust = 0,
            size = 3.3, color = paper, fontface = "bold") +
  coord_flip() +
  scale_fill_manual(values = c("best" = col_good, "prev" = col_mod, "other" = col_hard), guide = "none") +
  scale_y_continuous(limits = c(0, 105), expand = expansion(mult = c(0, 0.02))) +
  labs(title = "What worked: pretrained autoregressive + decoder capacity",
       subtitle = str_wrap("Held-out MEXA test WER. The full 32-layer decoder (large-v3) beat the 4-layer turbo base WITH reinforcement learning, before any RL of its own. CTC trails autoregressive on these polysynthetic languages.", 95),
       x = NULL, y = "WER (%, lower is better)",
       caption = "Byte-level, ByT5 speech-LLM, and from-scratch baselines were tried and dropped (see findings.md)") +
  base + theme(panel.grid.major.x = element_line(color = "#e7ecf0"),
               panel.grid.major.y = element_blank())
ggsave("report/figures/release_scoreboard.png", figB, width = 9, height = 4.6, dpi = 200)

cat("wrote release_per_language.png + release_scoreboard.png\n")
