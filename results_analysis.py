#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas
import numpy
import matplotlib.pyplot as pyplot
from Storage import DataStorage
from GeneticAlgorithm import GeneticParameters


# In[113]:


storage = DataStorage("simulation_data.db")
df = storage.get_pandas_dataframe()


# In[114]:


# In[115]:


sine_wave_experiments = [53, 54, 55]
kuramoto_experiments = [94, 95, 96]

sine_wave_fastest_genes = dict()
for exp_no in sine_wave_experiments: sine_wave_fastest_genes[exp_no] = []

kuramoto_fastest_genes = dict()
for exp_no in kuramoto_experiments: kuramoto_fastest_genes[exp_no] = []


# In[116]:


e = df[df['experiment'] == 53]['generation'].unique()


# In[117]:


for exp_no in sine_wave_experiments:
    experiment = df[df['experiment'] == exp_no]
    generations = experiment['generation'].unique()
    generations.sort()
    for gen in generations:
        gen_data = experiment[experiment['generation'] == gen].sort_values(by=["blps"], ascending=False)
        fastest = gen_data.iloc[0]
        gene = GeneticParameters.get_genotype_from_data(fastest.to_dict())
        sine_wave_fastest_genes[exp_no].append(gene)


# In[118]:


for exp_no in kuramoto_experiments:
    experiment = df[df['experiment'] == exp_no]
    generations = experiment['generation'].unique()
    generations.sort()
    for gen in generations:
        gen_data = experiment[experiment['generation'] == gen].sort_values(by=["blps"], ascending=False)
        fastest = gen_data.iloc[0]
        gene = GeneticParameters.get_genotype_from_data(fastest.to_dict())
        kuramoto_fastest_genes[exp_no].append(gene)


# In[119]:


sine_best_blps = [[val.body_length_per_second for val in experiment] for experiment in sine_wave_fastest_genes.values()]
kuramoto_best_blps = [[val.body_length_per_second for val in experiment] for experiment in kuramoto_fastest_genes.values()]


# In[120]:


figure, axes = pyplot.subplots(2, figsize=(10, 10))


# In[121]:


for ax in axes.flat:
    ax.cla()

for i,exp in enumerate(sine_best_blps):
    axes[0].plot(exp, label="Experiment " + str(i+1))
axes[0].legend()
for i,exp in enumerate(kuramoto_best_blps):
    axes[1].plot(exp, label="Experiment " + str(i+1))
axes[1].legend()

for ax in axes.flat:
    ax.set(xlabel="Generations", ylabel="Body lengths per second ($V_{b/s}$) ($cm\cdot s^{-1}$)")

axes[0].set_title("Sine wave actuation")
axes[1].set_title("CPG (Hopf) actuation")


# In[122]:

pyplot.show()


# In[123]:


fig_1, ax_1 = pyplot.subplots(5, 2, figsize=(10, 18))
fig_2, ax_2 = pyplot.subplots(5, 2, figsize=(10, 18))


# In[124]:


sine_wave_best_n = [val.number_of_nodes for val in sine_wave_fastest_genes[53]]
sine_wave_best_m = [val.mass_per_node for val in sine_wave_fastest_genes[53]]
sine_wave_best_mass_radius = [val.mass_radius for val in sine_wave_fastest_genes[53]]
sine_wave_best_spring_constant = [val.spring_constant for val in sine_wave_fastest_genes[53]]
sine_wave_best_spring_damping = [val.spring_damping for val in sine_wave_fastest_genes[53]]
sine_wave_best_amplitude = [val.actuator_params[0] for val in sine_wave_fastest_genes[53]]
sine_wave_best_actuator_count = [numpy.array(val.actuator_direction_selector).sum() for val in sine_wave_fastest_genes[53]]
sine_wave_best_data = [sine_wave_best_n, sine_wave_best_m, sine_wave_best_mass_radius, sine_wave_best_spring_constant, sine_wave_best_spring_damping, sine_wave_best_amplitude, sine_wave_best_actuator_count]
sine_wave_best_labels = ["Number of nodes", "Mass per node", "Mass radius", "Spring Constant", "Spring Damping", "Amplitude", "Actuator Count"]


# In[125]:


hopf_best_n = [val.number_of_nodes for val in kuramoto_fastest_genes[94]]
hopf_best_m = [val.mass_per_node for val in kuramoto_fastest_genes[94]]
hopf_best_mass_radius = [val.mass_radius for val in kuramoto_fastest_genes[94]]
hopf_best_spring_constant = [val.spring_constant for val in kuramoto_fastest_genes[94]]
hopf_best_spring_damping = [val.spring_damping for val in kuramoto_fastest_genes[94]]
hopf_best_amplitude = [val.actuator_params[1] for val in kuramoto_fastest_genes[94]]
hopf_best_actuator_count = [numpy.array(val.actuator_direction_selector).sum() for val in kuramoto_fastest_genes[94]]
hopf_best_alpha = [val.actuator_params[2] for val in kuramoto_fastest_genes[94]]
hopf_best_beta = [val.actuator_params[3] for val in kuramoto_fastest_genes[94]]
hopf_best_data = [hopf_best_n, hopf_best_m, hopf_best_mass_radius, hopf_best_spring_constant, hopf_best_spring_damping, hopf_best_amplitude, hopf_best_actuator_count, hopf_best_alpha, hopf_best_beta]
hopf_best_labels = ["Number of nodes", "Mass per node", "Mass radius", "Spring Constant", "Spring Damping", "Amplitude", "Actuator Count", "Alpha", "Beta"]


# In[126]:


for x in range(5):
    for y in range(2):
        i = x * 2 + y
        if i < len(sine_wave_best_data):
            ax_1[x][y].plot(sine_wave_best_data[i])
            ax_1[x][y].set(ylabel=sine_wave_best_labels[i], xlabel="Generations")

for x in range(5):
    for y in range(2):
        i = x * 2 + y
        if i < len(hopf_best_data):
            ax_2[x][y].plot(hopf_best_data[i])
            ax_2[x][y].set(ylabel=hopf_best_labels[i], xlabel="Generations")


# In[127]:

pyplot.show()


# In[162]:


fig_2, ax_2 = pyplot.subplots(3, figsize=(5, 15))


# In[163]:


no_feedback_experiments = [940, 950, 960]
for i, exp_no in enumerate(no_feedback_experiments):
    no_feedback = df[df['experiment'] == exp_no].sort_values(by=["blps"], ascending=True)
    to_plot = []
    for row in no_feedback.iloc:
        gene = GeneticParameters.get_genotype_from_data(row.to_dict())
        to_plot.append(gene.body_length_per_second)
    ax_2[i].cla()
    ax_2[i].plot(kuramoto_best_blps[i], label="with feedback")
    ax_2[i].plot(to_plot, label="without feedback")
    ax_2[i].set(title="Experiment " + str(i+1), xlabel="Generation", ylabel="Performance (BLPS)")
    ax_2[i].legend()


# In[ ]:





# In[164]:


pyplot.show()


# In[131]:


len(to_plot)


# In[132]:


exp_fb_nums = [300, 100]
exp_nofb_nums = [301, 101]


# In[157]:


fig_3, ax_3 = pyplot.subplots(2, figsize=(10, 15))
exp_n = ["Experiment 1", "Experiment 3"]

# In[160]:


for i in range(len(exp_fb_nums)):
    exp_with_feedback = df[df['experiment'] == exp_fb_nums[i]]
    exp_without_feedback = df[df['experiment'] == exp_nofb_nums[i]]
    fb_plt = []
    for row in exp_with_feedback.iloc:
        gene = GeneticParameters.get_genotype_from_data(row.to_dict())
        fb_plt.append(gene.body_length_per_second)
    nofb_plt = []
    for row in exp_without_feedback.iloc:
        gene = GeneticParameters.get_genotype_from_data(row.to_dict())
        nofb_plt.append(gene.body_length_per_second)
    print("Len fb:", len(fb_plt), " ; nofb: ", len(nofb_plt))
    ax_3[i].cla()
    ax_3[i].plot(fb_plt, label="With feedback")
    ax_3[i].plot(nofb_plt, label="Without feedback")
    ax_3[i].set(title=exp_n[i], xlabel="Randomized terrain", ylabel="Performance (BLPS)")
    ax_3[i].legend()


# In[ ]:





# In[161]:


pyplot.show()


# In[ ]:




