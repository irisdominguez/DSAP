import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import itertools

import profile_generation
import auxiliary_model

def ds(base_profile, ref_profile):
    sim = 1.0 - np.sum(np.abs(base_profile - ref_profile)) * 0.5
    return sim

def normalize_profile(profile):
    return profile / profile.sum()

def ds_representational(base_profile, target_profile=None):
    if target_profile is None:
        representational_profile = base_profile.copy()
        representational_profile[:] = 1
    else:
        representational_profile = target_profile
    representational_profile = normalize_profile(representational_profile)
    return ds(base_profile, representational_profile)

def ds_evenness(base_profile, target_profile=None):
    if target_profile is None:
        evenness_profile = base_profile.copy()
        evenness_profile[:] = 1
    else:
        evenness_profile = target_profile
    evenness_profile[base_profile == 0] = 0
    evenness_profile = normalize_profile(evenness_profile)
    return ds(base_profile, evenness_profile)

def ds_stereotypical(demographic_info, axis, target, target_labels):
    partials = []
    axis_groups=auxiliary_model.get_groups(axis)

    for label in target_labels:
        di_label = demographic_info[demographic_info[target] == label]
        di_rest = demographic_info[demographic_info[target] != label]
        p1 = profile_generation.generate_profile(di_label, 
                                               axis, 
                                               base_groups=axis_groups)
        p2 = profile_generation.generate_profile(di_rest, 
                                               axis, 
                                               base_groups=axis_groups)
        partials.append(ds(p1, p2))
    return np.mean(partials)



def generate_distance_matrix(ax_profiles):
    simil = {}
    for dataset1, dataset2 in itertools.combinations(ax_profiles.keys(), 2):
        simil[(dataset1, dataset2)] = ds(ax_profiles[dataset1], 
                                         ax_profiles[dataset2])

    ct = pd.crosstab(ax_profiles.keys(), ax_profiles.keys(), 
                     rownames=[None], colnames=[None])
    
    ct[:] = 1
    for (n1, n2), b in simil.items():
        ct.loc[n2, n1] = b
        ct.loc[n1, n2] = b
    return ct

def cluster_number_to_id(num):
    if num == 0:
        return ""
    q, r = divmod(num-1, 26)
    return cluster_number_to_id(q) + chr(ord("A") + r)

def dataset_clustering(similarity_matrix):
    Z = scipy.cluster.hierarchy.complete(similarity_matrix)
    clusters = scipy.cluster.hierarchy.fcluster(Z, criterion='distance', t=0.6)
    order = scipy.cluster.hierarchy.leaves_list(Z)

    clustermap = dict([(x,y) for (y,x) in enumerate(list(dict.fromkeys([clusters[i] for i in order])))])

    clusters = [clustermap[x] for x in clusters]

    clusterdic = list(zip(similarity_matrix.index,
                         [cluster_number_to_id(c+1) for c in clusters]))

    clusters = [clusters[x] for x in order]
    clusterdic = pd.Series(dict([clusterdic[x] for x in order]))

    colors = sns.color_palette("hls", len(set(clusters)))
    
    colors = pd.Series(dict((x, colors[clusters[i]]) for i, (x, y) in enumerate(clusterdic.to_dict().items())))

    return type('obj', (object,), {
        'Z' : Z,
        'assignation': clusterdic,
        'colors': colors
    })

def plot_similarity_matrix(similarity_matrix, clustering=None):
    if clustering is None:
        clustering = dataset_clustering(similarity_matrix)

    cmap = sns.color_palette("ch:s=2.5,r=.6,d=0.2,l=1", as_cmap=True)
    
    def formatnumber(x):
        x = round(x, 2)
        if x == 0:
            return '0'
        elif x == 1:
            return '1'
        else:
            return f'{x:.2f}'[1:]
    
    annot = similarity_matrix.applymap(formatnumber)
    
    cm = sns.clustermap(similarity_matrix.astype('float'), method="complete", cmap=cmap, 
                    col_linkage=clustering.Z,
                    row_linkage=None,
                    annot=annot, fmt='s',
                    col_colors=clustering.colors,
                    dendrogram_ratio=(0, 0.16), square=False,
                    vmin=0, vmax=1, figsize=(8,9), 
                    cbar_pos=(0.3, 0, 0.4, 0.01),
                    cbar_kws=dict(orientation='horizontal'))
    
    for i, cl in enumerate(clustering.assignation):
            y = 0.5
            x = i + 0.5
            cm.ax_col_colors.text(x, y, cl, ha="center", va="center")


def generate_bias_table(ax_profiles, axis, vocabulary):
    bias_table = {}
    
    for dataset, profile in ax_profiles.items():
        demographic_info = profile_generation.load_demographic_info(dataset)
        bias_table[dataset] = {
            r"Representational ($DS_R$)": ds_representational(profile),
            r"Evenness ($DS_E$)": ds_evenness(profile),
            r"Stereotypical ($DS_S$)": ds_stereotypical(demographic_info, 
                                                         axis, 
                                                         'label', vocabulary),
        }
    bias_table = pd.DataFrame(bias_table).T
    return bias_table