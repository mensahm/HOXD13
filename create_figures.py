# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:51:34 2023

@author: Martin
"""

from utils_hox import *

# Supplementary Table 1 and Supplementary Table 2 need the follwing format: SuppT1.csv and SuppT2.tsv

# load data
get_table("SuppT2.tsv","numtable_without.csv",["label", "sex"],features,2,numval, rows=(None,25), separate_sides=False)
get_table("SuppT2.tsv","numtable_withgli.csv",["label", "sex"],features,2,numval, separate_sides=False)

# create density plots
kde_plots("numtable_without.csv", False, "without", True)
kde_plots("numtable_withgli.csv", True, "withgli", True)

# create decision tree plots
mut_labels=['ins7Ala', 'ins8Ala', 'ins9Ala', "missense", "truncating", "other"]
gli_labels=["GLI3", 'ins7Ala', 'ins8Ala', 'ins9Ala']
ins_labels=['ins7Ala', 'ins8Ala', 'ins9Ala']
tree_tables=[["numtable_withgli.csv",gli_labels,"withgli",2,True],["numtable_without.csv",ins_labels, "without",2,False]]
calculate_decision_trees([3], tree_tables)

# create severity-score-box-plots
boxplot_severity(severity_scores_perfam("SuppT1.csv"), "Boxplot")

# create heatmap (note needs to be sorted in Excel)
plot_heatmap ("numtable_without.csv",2)