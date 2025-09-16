# title: "Exploring the impact of cognitive conflict on subsequent cognitive processes (Manuscript)"
# author: "Marta La Pietra"
# date: "2025-04-25"
# Project: SweetC - Behavioral Experiment
# Description: This script analyzes the data from the Experiment 3 (Stroop task intermixed with a semantic categorisation task + an impromptu implicit memory task). The effect of interest pertains the accuracy at the implicit memory task.


# Libraries
library(ggeffects)
library(ggplot2)
library(rstanarm)
library(ggsignif)
library(devtools)
library(readxl)
library(emmeans)
library(here)         # relative paths
library(tidyverse)    # tidy functions
library(knitr)        # knit functions
library(kableExtra)   # extra markdown functions
library(purrr)        # map functions
library(lme4)         # mixed-effects regressions
library(lmerTest)     # mixed-effects regressions
library(AICcmodavg)   # predictSE()
library(broom.mixed)  # tidy()
library(ggrepel)      # geom_text_repel
library(sjPlot)       # tab_model
library(rstatix)      # cohen's d
library(ggridges)     # density plot
library(tidytext)
library(readxl)
library(devtools)
library(dotwhisker)
library(dplyr)
library(tidyr)
library(brms)
library(car)

## Data
# Specify relative paths
dir_analysis <- here("GitHub/") # indicate your directory
dir_parent <- str_remove(dir_analysis, "/analysis")
dir_data <- str_c(dir_parent, "/data")
dir_graphs <- str_c(dir_parent, "/graphs")

# Load the proper dataset
stroop <- read_excel(str_c(dir_data, "/exp3_stroop_clean.xlsx")) 

# Exploratory analyses: Linear Mixed-Effects Model
# Order the Stroop trial type for the LMM
stroop$Congruence <- factor(stroop$Congruence, levels = c("Incongruent","Congruent","Neutral"))
# Log-transform reaction time values to account for their positively skewed distribution
stroop$log_RT <- log(stroop$RT)
# Fit the model
model_stroop = lmer(log_RT ~ Congruence + (1|P), data=stroop, REML = FALSE)
summary(model_stroop)

# Plot Stroop mean Reaction Times
stroop_RTs <- read_excel(str_c(dir_data, "/exp3_stroop_meanRTs.xlsx"))

conflict_theme <- theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"), # Add axis lines
        panel.border = element_blank(),
        aspect.ratio = 1,
        axis.text.x = element_text(size = 20,color = "black"),
        axis.text.y = element_text(size = 20,color = "black"),
        axis.title = element_text(size = 20),
        axis.title.x = element_text(margin = margin(t = 5)),  # t = top margin
        axis.title.y = element_text(margin = margin(r = 5)),   # r = right margin
        legend.position = "none")

plot_width = 6
plot_height = 6

# Define the color palette
congruence <- c("Congruent", "Incongruent", "Neutral")
# Define the color palette
palette <- c("#ee4266", "#ffd23f", "#3bceac")

# Define the order of the Stroop trial types
stroop$Congruence <- factor(stroop$Congruence, levels = c("Congruent", "Incongruent", "Neutral"))

fig2_data <- stroop_RTs %>%
  mutate(Congruence = fct_relevel(Congruence, congruence))

# Calculate means and SEMs
means <- fig2_data %>%
  group_by(Congruence) %>%
  summarise(mean = mean(RT), .groups = 'drop')

sems <- fig2_data %>%
  group_by(Congruence) %>%
  summarise(sem = sd(RT) / sqrt(n()), .groups = 'drop')

# Combine means and SEMs
summary_data <- means %>%
  left_join(sems, by = "Congruence")

# Create the plot
fig2_plot <- ggplot(fig2_data, aes(x = Congruence, y = RT, color = Congruence)) +
  # Violin plot
  geom_violin(aes(fill = Congruence), alpha = 0.7, trim = TRUE, scale = "width", width = 0.5) +
  # Swarm plot
  geom_jitter(aes(x = Congruence, y = RT), color = "grey", alpha = 0.3, width = 0.2) +
  # SEM error bars
  geom_errorbar(data = summary_data, aes(x = Congruence, y = mean, ymin = mean - sem, ymax = mean + sem),
                width = 0.1, color = "black", size = 0.8) +
  # Mean points
  geom_point(data = summary_data, aes(x = Congruence, y = mean), color = "black", size = 3) +
  # Chance level line
  # geom_hline(yintercept = 0.33, color = "#d90429", linetype = "dashed", size = 1) +
  # Customize colors
  scale_fill_manual(values = palette) +
  scale_color_manual(values = c("#ee4266", "#ffd23f", "#3bceac")) +  # Custom colors for Conflict Levels
  scale_y_continuous(name = "Mean Reaction Times (s)", breaks = seq(0.2, 1.5, by = 0.2), limits =c(0.4, 1.25))+#,labels = scales::number_format(scale = 1, suffix = "%", accuracy = 0.1)) +  # Customize y-axis
  labs(y = "RT", x = "Trial Type") +
  conflict_theme
fig2_plot

# Save the plots as figures in the "stroop" folder
# Experiment 3: Stroop task + Implicit Memory test
ggsave(filename=str_c(dir_graphs, "/stroop/fig2_exp3.pdf"), fig2_plot, width = 6, height = 6, useDingbats=F)
ggsave(filename=str_c(dir_graphs, "/stroop/fig2_exp3.png"), fig2_plot, width = 6, height = 6)

# Calculate significance
# Perform pairwise t-tests
pairwise_tests <- pairwise.t.test(
  stroop$RT,
  stroop$Congruence,
  p.adjust.method = "bonferroni"  # Adjust p-values for multiple comparisons
)

# Display the results
print(pairwise_tests)

# Extract p-values
p_values <- pairwise_tests$p.value

# Define significance levels
significance_level <- 0.05

# Identify significant comparisons
significant_comparisons <- which(p_values < significance_level, arr.ind = TRUE)
significant_comparisons

# Define significance annotations
fig2_plot_sign <- fig2_plot +
  geom_signif(comparisons = list(c("Congruent", "Incongruent"), c("Incongruent", "Neutral"),c("Congruent", "Neutral")),
              annotations = c("***",   "***", "***"),  # Adjust based on p-values
              color = "black",
              y_position = c(1.1, 1.15, 1.20),  # Adjust y positions to avoid overlap
              tip_length = 0.01,
              vjust = 0.3,
              size = 0.5,  textsize = 6)

fig2_plot_sign

# Save the plots as figures in the "stroop" folder with statistical significance signs
ggsave(filename=str_c(dir_graphs, "/stroop/fig2_exp3_sign.pdf"), fig2_plot_sign, width = 6, height = 6, useDingbats=F)
ggsave(filename=str_c(dir_graphs, "/stroop/fig2_exp3_sign.png"), fig2_plot_sign, width = 6, height = 6)

#------------------------------------------------------- Secondary task
# Load the proper dataset
memory <- read_excel(str_c(dir_data, "/exp3_secondary.xlsx")) 

# Exploratory analyses: Linear Mixed-Effects Models for secondary Accuracy
# Order the Stroop trial type
memory$Congruence <- factor(memory$Congruence, levels = c("Congruent", "Incongruent", "Neutral"))
# Accuray at the memory task
model = lmer(Accuracy ~ Congruence + (1|P), data=memory, REML = FALSE)
summary(model)

# --------------------- Figures for the secondary task
# Load the proper dataset
memory_results <- read_excel(str_c(dir_data, "/exp3_secondary_summary.xlsx")) 

# Plot mean Accuracy at the secondary task
# Define the color palette
congruence <- c("Congruent", "Incongruent", "Neutral")
# Define the color palette
palette <- c("#ee4266", "#ffd23f", "#3bceac")

fig3_data <- memory_results %>%
  mutate(Congruence = fct_relevel(Congruence, congruence))

# Calculate means and SEMs
means <- fig3_data %>%
  group_by(Congruence) %>%
  summarise(mean = mean(Accuracy), .groups = 'drop')

sems <- fig3_data %>%
  group_by(Congruence) %>%
  summarise(sem = sd(Accuracy) / sqrt(n()), .groups = 'drop')

# Combine means and SEMs
summary_data <- means %>%
  left_join(sems, by = "Congruence")

# Create the plot
fig3_plot <- ggplot(fig3_data, aes(x = Congruence, y = Accuracy, color = Congruence)) +
  # Violin plot
  geom_violin(aes(fill = Congruence), alpha = 0.7, trim = TRUE, scale = "width", width = 0.5) +
  # Swarm plot
  geom_jitter(aes(x = Congruence, y = Accuracy), color = "grey", alpha = 0.3, width = 0.2) +
  # SEM error bars
  geom_errorbar(data = summary_data, aes(x = Congruence, y = mean, ymin = mean - sem, ymax = mean + sem),
                width = 0.1, color = "black", size = 0.8) +
  # Mean points
  geom_point(data = summary_data, aes(x = Congruence, y = mean), color = "black", size = 3) +
  # Chance level line
  # geom_hline(yintercept = 0.33, color = "#d90429", linetype = "dashed", size = 1) +
  # Customize colors
  scale_fill_manual(values = palette) +
  scale_color_manual(values = c("#ee4266", "#ffd23f", "#3bceac")) +  # Custom colors for Conflict Levels
  scale_y_continuous(name = "Accuracy", breaks = seq(70, 100, by = 10))+#,labels = scales::number_format(scale = 1, suffix = "%", accuracy = 0.1)) +  # Customize y-axis
  labs(y = "Accuracy", x = "Previous Trial Type") +
  conflict_theme
fig3_plot

# Save the plots as figures in the "secondary" folder
# Experiment 3: Stroop task + Implicit Memory Test
ggsave(filename=str_c(dir_graphs, "/secondary/fig3_exp3.pdf"), fig3_plot, width = 6, height = 6, useDingbats=F)
ggsave(filename=str_c(dir_graphs, "/secondary/fig3_exp3.png"), fig3_plot, width = 6, height = 6)

# Perform pairwise t-tests
pairwise_tests <- pairwise.t.test(
  secondary$Accuracy,
  secondary$Congruence,
  p.adjust.method = "bonferroni"  # Adjust p-values for multiple comparisons
)

# Display the results
print(pairwise_tests)

# Extract p-values
p_values <- pairwise_tests$p.value

# Define significance levels
significance_level <- 0.05

# Identify significant comparisons
significant_comparisons <- which(p_values < significance_level, arr.ind = TRUE)
significant_comparisons

# Define significance annotations
fig3_plot_sign <- fig3_plot +
  geom_signif(comparisons = list(c("Congruent", "Incongruent"), c("Incongruent", "Neutral"),c("Congruent", "Neutral")),
              annotations = c("n.s.", "n.s.", "n.s."),  # Adjust based on p-values
              color = "black",
              y_position = c(100.5, 101.5, 103.5),  # Adjust y positions to avoid overlap
              tip_length = 0.01,
              vjust = 0.1,
              size = 0.5,  textsize = 6)

fig3_plot_sign

# Save the plots as figures in the "secondary" folder with statistical significance signs
ggsave(filename=str_c(dir_graphs, "/secondary/fig3_exp3_sign.pdf"), fig3_plot_sign, width = 6, height = 5, useDingbats=F)
ggsave(filename=str_c(dir_graphs, "/secondary/fig3_exp3_sign.png"), fig3_plot_sign, width = 6, height = 6)


# ----------------------------------------- Reaction Times at the Implicit Memory task
# Clean the main dataset of the secondary task
memory <- memory[memory$RT >= 0.05, ]
excluded_toofast <- memory[memory$RT <= 0.05, ]
excluded_wrong <- memory[memory$Accuracy ==0, ]
memory_clean <- memory[memory$Accuracy == 1, ]

# Exploratory analyses: Linear Mixed-Effects Models for secondary RTs
# Order the Stroop trial type
memory_clean$Congruence <- factor(memory_clean$Congruence, levels = c("Congruent", "Incongruent", "Neutral"))

# Log-transform reaction time values to account for their positively skewed distribution
memory_clean$log_RT <- log(memory_clean$RT)
modelRT = lmer(log_RT ~ Congruence + (1|P), data=memory_clean, REML = FALSE)
summary(modelRT)

# Filter the data and calculate the mean and standard deviation, rounded to 2 decimals
stats_RT <- memory_clean %>%
  filter(Congruence == "Neutral") %>%
  summarize(
    mean = mean(RT, na.rm = TRUE), 2,
    sd = sd(RT, na.rm = TRUE), 2
  )

# Extract and print the values individually
mean <- stats_RT$mean*1000
sd <- stats_RT$sd*1000

# Print with two decimals
cat("Mean RTs:", mean, "\n")
cat("Standard Deviation RTs:", sd, "\n")

#----------------- Figures for the secondary task: RTs at the implicit memory test
# Calculate means and SEMs
means <- fig5_data %>%
  group_by(Congruence) %>%
  summarise(mean = mean(MeanRT), .groups = 'drop')

sems <- fig5_data %>%
  group_by(Congruence) %>%
  summarise(sem = sd(MeanRT) / sqrt(n()), .groups = 'drop')

# Combine means and SEMs
summary_data <- means %>%
  left_join(sems, by = "Congruence")

# Create the plot
fig5_plot <- ggplot(fig5_data, aes(x = Congruence, y = MeanRT, color = Congruence)) +
  # Violin plot
  geom_violin(aes(fill = Congruence), alpha = 0.7, trim = TRUE, scale = "width", width = 0.5) +
  # Swarm plot
  geom_jitter(aes(x = Congruence, y = MeanRT), color = "grey", alpha = 0.3, width = 0.2) +
  # SEM error bars
  geom_errorbar(data = summary_data, aes(x = Congruence, y = mean, ymin = mean - sem, ymax = mean + sem),
                width = 0.1, color = "black", size = 0.8) +
  # Mean points
  geom_point(data = summary_data, aes(x = Congruence, y = mean), color = "black", size = 3) +
  # Chance level line
  # geom_hline(yintercept = 0.33, color = "#d90429", linetype = "dashed", size = 1) +
  # Customize colors
  scale_fill_manual(values = palette) +
  scale_color_manual(values = c("#ee4266", "#ffd23f", "#3bceac")) +  # Custom colors for Conflict Levels
  scale_y_continuous(name = "Mean Reaction Times (s)", breaks = seq(0, 3, by = 0.5))+#,labels = scales::number_format(scale = 1, suffix = "%", accuracy = 0.1)) +  # Customize y-axis
  labs(y = "MeanRT", x = "Previous Trial Type") +
  conflict_theme
fig5_plot

# Save the plots as figures in the "secondary" folder
ggsave(filename=str_c(dir_graphs, "/secondary/fig5_exp3.pdf"), fig5_plot, width = 6, height = 5, useDingbats=F)
ggsave(filename=str_c(dir_graphs, "/secondary/fig5_exp3.png"), fig5_plot, width = 6, height = 6)


# Perform pairwise t-tests
pairwise_tests <- pairwise.t.test(
  memory_results$MeanRT,
  memory_results$Congruence,
  p.adjust.method = "bonferroni"  # Adjust p-values for multiple comparisons
)

# Display the results
print(pairwise_tests)

# Extract p-values
p_values <- pairwise_tests$p.value

# Define significance levels
significance_level <- 0.05

# Identify significant comparisons
significant_comparisons <- which(p_values < significance_level, arr.ind = TRUE)
significant_comparisons

# Define significance annotations
fig5_plot_sign <- fig5_plot +
  geom_signif(comparisons = list(c("Congruent", "Incongruent"), c("Incongruent", "Neutral"),c("Congruent", "Neutral")),
              annotations = c("n.s.",   "n.s.", "n.s."),  # Adjust based on p-values
              color = "black",
              y_position = c(3, 3.1, 3.2),  # Adjust y positions to avoid overlap
              tip_length = 0.01,
              vjust = 0.1,
              size = 0.5,  textsize = 6)

fig5_plot_sign

# Save the plots as figures in the "secondary" folder
ggsave(filename=str_c(dir_graphs, "/secondary/fig5_exp3_sign.pdf"), fig5_plot_sign, width = 6, height = 5, useDingbats=F)
ggsave(filename=str_c(dir_graphs, "/secondary/fig5_exp3_sign.png"), fig5_plot_sign, width = 6, height = 6)


