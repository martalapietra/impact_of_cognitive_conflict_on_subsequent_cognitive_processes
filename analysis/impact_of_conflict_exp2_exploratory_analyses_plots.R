# title: "Exploring the impact of cognitive conflict on subsequent cognitive processes (Manuscript)"
# author: "Marta La Pietra"
# date: "2025-04-25"
# Project: SweetC - Behavioral Experiment
# Description: This script analyzes the data from the Experiment 2 (Stroop task intermixed with a Go/No-Go task) in the Registered Report titled "Exploring the impact of cognitive conflict on subsequent cognitive processes". 


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
dir_data <- str_c(dir_parent, "/data/")
dir_graphs <- str_c(dir_parent, "/graphs")

# Load the proper dataset
stroop <- read_excel(str_c(dir_data, "/exp2_stroop_clean.xlsx")) 

# Exploratory analyses: Linear Mixed-Effects Model
# Order the Stroop trial type for the LMM
stroop$Congruence <- factor(stroop$Congruence, levels = c("Incongruent","Congruent","Neutral"))
# Log-transform reaction time values to account for their positively skewed distribution
stroop$log_RT <- log(stroop$RT)
# Fit the model
model_stroop = lmer(log_RT ~ Congruence + (1|P), data=stroop, REML = FALSE)# Fit the model
summary(model_stroop)

# Load the proper dataset
stroop_RTs <- read_excel(str_c(dir_data, "/exp2_stroop_meanRTs.xlsx"))

# Plot Stroop mean Reaction Times
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
# Experiment 2: Stroop task + Go/No-Go task
ggsave(filename=str_c(dir_graphs, "/stroop/fig2_exp2.pdf"), fig2_plot, width = 6, height = 6, useDingbats=F)
ggsave(filename=str_c(dir_graphs, "/stroop/fig2_exp2.png"), fig2_plot, width = 6, height = 6)

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
              annotations = c("***",   "***", "n.s."),  # Adjust based on p-values
              color = "black",
              y_position = c(1.1, 1.15, 1.20),  # Adjust y positions to avoid overlap
              tip_length = 0.01,
              vjust = 0.3,
              size = 0.5,  textsize = 6)

fig2_plot_sign

# Save the plots as figures in the "stroop" folder with statistical significance signs
ggsave(filename=str_c(dir_graphs, "/stroop/fig2_exp2_sign.pdf"), fig2_plot_sign, width = 6, height = 6, useDingbats=F)
ggsave(filename=str_c(dir_graphs, "/stroop/fig2_exp2_sign.png"), fig2_plot_sign, width = 6, height = 6)

#------------------------------------------------------- Secondary task
# Load the proper dataset
secondary <- read_excel(str_c(dir_data, "/exp2_secondary_clean.xlsx")) 

# Select only Hits
GO <- filter(secondary, Type == "GO")
# Filter rows where RT <= 0.05
excluded_toofast <- GO[GO$RT2 <= 0.05, ]
# Exclude rows where RT <= 0.05
GO <- GO[GO$RT2 >= 0.05, ]

# Exploratory analyses: Linear Mixed-Effects Models for secondary RTs
# Order the Stroop trial type
GO$Congruence <- factor(GO$Congruence, levels = c("Incongruent", "Congruent", "Neutral"))
# Log-transform reaction time values to account for their positively skewed distribution
GO$log_RT2 <- log(GO$RT2)
# Fit the model to see how RTs are affected by the Stroop trial type
model_secondary = lmer(log_RT2 ~ Congruence + (1|P), data=GO, REML = FALSE)
summary(model_secondary)

# Accuracy to Go trials - Both groups
TotalGO <- secondary %>%
  filter((Type == 'GO' | Type == 'Miss'))  

# Hit percentage
stats_ACC <- TotalGO %>%
  # filter(Group_Map == "200/160") %>%  #'240/120', '200/160'
  # filter(Congruence == "Neutral") %>% #"Congruent", "Incongruent", "Neutral"
  summarize(
    mean_ACC = mean(Accuracy2, na.rm = TRUE), 2,
    sd_ACC = sd(Accuracy2, na.rm = TRUE), 2
  )
mean_ACC <- stats_ACC$mean_ACC*100
sd_ACC <- stats_ACC$sd_ACC*100
# Print with two decimals
cat("Mean:", mean_ACC, "\n")
cat("Standard Deviation:", sd_ACC, "\n")

# Miss percentage
# Calculate total number of GO trials
total_GO_trials <- nrow(TotalGO)
# Calculate number of Miss trials
miss_trials <- nrow(filter(TotalGO, Type == 'Miss'))
# Calculate the percentage of misses
miss_percentage <- (miss_trials / total_GO_trials) * 100
# Calculate the standard deviation of the percentage of misses
p <- miss_trials / total_GO_trials  # Proportion of misses
sd_miss_percentage <- sqrt(p * (1 - p) / total_GO_trials) * 100
# Print results with two decimals
cat("Percentage of Misses:", round(miss_percentage, 2), "%\n")
cat("Standard Deviation of Miss Percentage:", round(sd_miss_percentage, 2), "%\n")

# Correct rejections
TotalNOGO <- secondary %>%
  filter((Type=='NO-GO'| Type=='False Alarm'))

# Filter the data and calculate the mean and standard deviation, rounded to 2 decimals
stats_ACC <- TotalNOGO %>%
  # filter(Group_Map == "200/160") %>% #'240/120', '200/160'
  # filter(Congruence == "Congruent") %>% #"Congruent", "Incongruent", "Neutral"
  summarize(
    mean_ACC = mean(Accuracy2, na.rm = TRUE), 2,
    sd_ACC = sd(Accuracy2, na.rm = TRUE), 2
  )
mean_ACC <- stats_ACC$mean_ACC*100
sd_ACC <- stats_ACC$sd_ACC*100
# Print with two decimals
cat("Mean:", mean_ACC, "\n")
cat("Standard Deviation:", sd_ACC, "\n")

# False Alarm percentage
# Calculate total number of GO trials
total_NOGO_trials <- nrow(TotalNOGO)
# Calculate number of Miss trials
FA_trials <- nrow(filter(TotalNOGO, Type == 'False Alarm'))
# Calculate the percentage of misses
FA_percentage <- (FA_trials / total_NOGO_trials) * 100
# Calculate the standard deviation of the percentage of misses
p <- FA_trials / total_NOGO_trials  # Proportion of misses
sd_FA_percentage <- sqrt(p * (1 - p) / total_NOGO_trials) * 100
# Print results with two decimals
cat("Percentage of Misses:", round(FA_percentage, 2), "%\n")
cat("Standard Deviation of Miss Percentage:", round(sd_FA_percentage, 2), "%\n")

# Fit the model to see how accuracy was affected by Stroop trial type
secondary$Congruence <- factor(secondary$Congruence, levels = c("Congruent", "Incongruent", "Neutral"))
model_accuracy = lmer(Accuracy2 ~ Congruence + (1|P), data=secondary, REML = FALSE)
summary(model_accuracy)

# ------- Analysis for groups
# Group 1 = 240 GO, 120 NO-GO
# Group 2 = 200 GO, 160 NO-GO
# Comparison between the two groups
model_groups = lmer(RT2 ~ Group + (1|P), data=GO, REML = FALSE)
summary(model_groups)

# Group 1
group1 <- filter(GO, Group == "1")
group1$Congruence <- factor(group1$Congruence, levels = c("Incongruent","Congruent","Neutral"))
group1$log_RT2 <- log(group1$RT2)
model_group1 = lmer(log_RT2 ~ Congruence + (1|P), data=group1, REML = FALSE)
summary(model_group1)

# Group 2
group2 <- filter(GO, Group == "2")
group2$Congruence <- factor(group2$Congruence, levels = c("Incongruent","Congruent","Neutral"))
group2$log_RT2 <- log(group2$RT2)
model_group2 = lmer(log_RT2 ~ Congruence + (1|P), data=group2, REML = FALSE)
summary(model_group2)

# Filter the data and calculate the mean and standard deviation, rounded to 2 decimals
stats_RTs <- GO %>%
  filter(Group == "2") %>%
  filter(Congruence == "Incongruent") %>%
  summarize(
    mean_RTs = mean(RT2, na.rm = TRUE), 2,
    sd_RTs = sd(RT2, na.rm = TRUE), 2
  )

# Extract and print the values individually
mean_RTs <- stats_RTs$mean_RTs*1000
sd_RTs <- stats_RTs$sd_RTs*1000

# Print with two decimals
cat("Mean RTs:", mean_RTs, "\n")
cat("Standard Deviation RTs:", sd_RTs, "\n")

# --------------------- Figures for the secondary task
# Load the proper dataset
secondary_RTs <- read_excel(str_c(dir_data, "/exp2_secondary_meanRTs.xlsx")) 

## Select the different groups, if you are interested
# secondary <- secondary[secondary$Group == "1", ] #1 #2 
# !!!!Please remember to save the plots as Fig.4 if you select only one of the groups.!!!!

# Order the Stroop trial type
secondary_RTs$Congruence <- factor(secondary_RTs$Congruence, levels = c("Congruent", "Incongruent", "Neutral"))

# Plot mean Reaction Times of the secondary task
# Define the color palette
congruence <- c("Congruent", "Incongruent", "Neutral")
# Define the color palette
palette <- c("#ee4266", "#ffd23f", "#3bceac")

fig3_data <- secondary_RTs %>%
  mutate(Congruence = fct_relevel(Congruence, congruence))

means <- fig3_data %>%
  group_by(Congruence) %>%
  summarise(mean = mean(RT2), .groups = 'drop') #ACCURACY AND MeanRT FOR MEMORY

sems <- fig3_data %>%
  group_by(Congruence) %>%
  summarise(sem = sd(RT2) / sqrt(n()), .groups = 'drop')

# Combine means and SEMs
summary_data <- means %>%
  left_join(sems, by = "Congruence")

# Create the plot
fig3_plot <- ggplot(fig3_data, aes(x = Congruence, y = RT2, color = Congruence)) +
  # Violin plot
  geom_violin(aes(fill = Congruence), alpha = 0.7, trim = TRUE, scale = "width", width = 0.5) +
  # Swarm plot
  geom_jitter(aes(x = Congruence, y = RT2), color = "grey", alpha = 0.3, width = 0.2) +
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
  scale_y_continuous(name = "Mean Reaction Times (s)", breaks = seq(0, 1.5, by = 0.2),limits =c(0.2, 1.1))+#,labels = scales::number_format(scale = 1, suffix = "%", accuracy = 0.1)) +  # Customize y-axis
  labs(y = "RT2", x = "Previous Trial Type") +
  conflict_theme
fig3_plot

# Save the plots as figures in the "secondary" folder
# Experiment 2: Stroop task + Go/No-Go task
ggsave(filename=str_c(dir_graphs, "/secondary/fig3_exp2.pdf"), fig3_plot, width = 6, height = 6, useDingbats=F) # If you select only one group change the figure name in 1) group1: fig4A_exp2.pdf, 2) group2: fig4B_exp2.pdf
ggsave(filename=str_c(dir_graphs, "/secondary/fig3_exp2.png"), fig3_plot, width = 6, height = 6) # If you select only one group change the figure name in 1) group1: fig4A_exp2.png, 2) group2: fig4B_exp2.png

# Perform pairwise t-tests
pairwise_tests <- pairwise.t.test(
  secondary$RT2,
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
              annotations = c("***",  "n.s.", "**"),  # Adjust based on p-values
              color = "black",
              y_position = c(0.9, 0.95, 1.0),  # Adjust y positions to avoid overlap
              tip_length = 0.01,
              vjust = 0.1,
              size = 0.5,  textsize = 6)
fig3_plot_sign

# Save the plots as figures in the "secondary" folder with statistical significance signs
ggsave(filename=str_c(dir_graphs, "/secondary/fig3_exp2_sign.pdf"), fig3_plot_sign, width = 6, height = 6, useDingbats=F) # If you select only one group change the figure name in 1) group1: fig4A_exp2.pdf, 2) group2: fig4B_exp2.pdf
ggsave(filename=str_c(dir_graphs, "/secondary/fig3_exp2_sign.png"), fig3_plot_sign, width = 6, height = 6) # If you select only one group change the figure name in 1) group1: fig4A_exp2.png, 2) group2: fig4B_exp2.png

