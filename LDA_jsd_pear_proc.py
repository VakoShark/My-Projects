'''
# This code is for LDA research with Dr. Stine, the utility code belongs to him
# This code takes in multiple document-by-topic matrices and converts them to 
  JSD matrices, and then takes those for each model (with stops and without stops)
  and compares each of them by dimensionality with Pearson correlation and Procrustes distance
  and writes those all to a singular CSV file
'''

import os
import corpus_processing
import compare_dist_matrices
import jsd_matrices
import csv
import numpy as np

if __name__ == '__main__':
    experiment_name = 'exp1'
    subreddit = 'hinduism'
    k_vals = ['50']
    model_ids = ['001', '002']
    model_types = ['wstops', 'nostops']
    
    # Directories
    cwd = os.getcwd()  # Directory of this program
    project_dir = os.path.dirname(cwd)  # Main project directory
    exp_dir = os.path.join(cwd, experiment_name)

    lda_dir = os.path.join(exp_dir, 'Applied Data Science\\3_lda')
    jsd_dir = corpus_processing.make_dir(os.path.join(exp_dir, 'Applied Data Science\\4_jsd'))
    print('Directories created')
    
    topic_dists_name = 'topic_dists'
    
    # Creating the JSD matrices and saving them to file
    for k_index, k in enumerate(k_vals):
        k_path = corpus_processing.make_dir(os.path.join(jsd_dir, 'k-' + k[k_index]))
        k_path_nolist = corpus_processing.make_dir(os.path.join(k_path, 'wstops'))
        k_path_wlist = corpus_processing.make_dir(os.path.join(k_path, 'nostops'))
        for id_index, id in enumerate(model_ids):
            for type_index, type in enumerate(model_types):
                
                # Getting the paths for the models with the current naming scheme
                model_path = os.path.join(lda_dir, subreddit, model_ids[id_index] + '_k-' + k_vals[k_index] + '_' + model_types[type_index])
                doc_topic_path = os.path.join(model_path, topic_dists_name, 'hinduism_tdists.txt')
                doc_topic_fname = os.path.basename(doc_topic_path)
                
                # Creating the JSD matrix and saving it to file
                jsd_matrix_fname = 'jsd_' + model_ids[id_index] + '_k-' + k_vals[k_index] + '_' + model_types[type_index]
                if model_types[type_index] == 'wstops':
                    jsd_matrix_path = os.path.join(k_path_nolist, jsd_matrix_fname)
                elif model_types[type_index] == 'nostops':
                    jsd_matrix_path = os.path.join(k_path_wlist, jsd_matrix_fname)
                    
                jsd_matrix = jsd_matrices.calculate_jsd_matrix(doc_topic_path, jsd_matrix_path)
                print('JSD matrix ' + jsd_matrix_fname + ' created')
        
    # Using the JSD matrices to calculate each of the pearson and procrustes comparisons
    pear_proc_dir = corpus_processing.make_dir(os.path.join(exp_dir, 'Applied Data Science\\5_pear_proc'))
    print('pear_proc_dir created')
    
    # Creating the CSV entry data consisting of: 'model_name_1', 'model_name_2', 'k', 'pearson', 'procrustes'
    csv_header = ['model_name_1', 'model_name_2', 'k', 'pearson', 'procrustes']
    csv_data = []
    
    for k_index, k in enumerate(k_vals):
        k_paths = os.path.join(jsd_dir, 'k-' + k[k_index])
        k_path_nolist = corpus_processing.make_dir(os.path.join(k_paths, 'wstops'))
        k_path_wlist = corpus_processing.make_dir(os.path.join(k_paths, 'nostops'))
        jsd_files_wstops = os.listdir(k_path_nolist)
        jsd_files_nostops = os.listdir(k_path_wlist)
        for f1 in jsd_files_wstops:
            for f2 in jsd_files_nostops:
                f1_path = os.path.join(k_path_nolist, f1)
                f2_path = os.path.join(k_path_wlist, f2)
                f1_data = np.load(f1_path)
                f2_data = np.load(f2_path)
                pear, proc = compare_dist_matrices.get_pear_and_proc(f1_data, f2_data)
                csv_data.append([str(f1), str(f2), k, pear, proc])
                print(str(f1) + ' vs ' + str(f2) + ' comparison complete')
                
    # Creating the CSV file and writing the data to it
    file_name = 'model_distances_data.csv'
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        writer.writerows(csv_data)
    
    print('Done!')
    