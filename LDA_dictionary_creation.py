'''
# This is code for LDA research working with Dr. Stine, most of this belongs to him
# This code takes in a subreddit as input and creates 2 different corpora for later use:
    1) corpus WITHOUT any stopwords removed
    2) corpus WITH stopwords removed from https://github.com/mimno/Mallet/blob/master/stoplists/en.txt
'''

import os
import corpus_processing

if __name__ == '__main__':
    experiment = 'exp1'
    subreddit = 'hinduism'
    
    # Directories
    cwd = os.getcwd()  # Directory of this program
    project_dir = os.path.dirname(cwd)  # Main project directory
    data_dir = os.path.join(project_dir, 'Applied Data Science\\0_data')
    
    print(str(data_dir))
    
    # Create directory for storing experiment-specific results
    exp_dir = corpus_processing.make_dir(os.path.join(cwd, experiment))
    print("exp_dir created")
    
    # Create directory for storing each bag-of-words object
    corpora_dir = corpus_processing.make_dir(os.path.join(exp_dir, 'Applied Data Science\\2_corpora'))
    print("corpora_dir created: " + str(corpora_dir))
    
    # Read in pre-made set of stopwords
    stoplist = corpus_processing.load_stoplist('stoplist.txt')
    print("stoplist created")
    
    # Open list of bot users to ignore
    ignore_users = list(set(open('ignore_users.txt', 'r', encoding='utf-8').read().lower().split(',\n')))
    print("ignore_users created")
    
    # Keep each dictionary in the following list:
    dict_list = []
    
    # Make a list of each document path for the subreddit. Function takes a list of subreddits, hence the [].
    print("Making doc_paths")
    doc_paths = corpus_processing.get_document_paths([subreddit], data_dir)[:1000]
    print("doc_paths created")

    # Create gensim dictionary object for subreddit (first one with stopwords left in)
    subr_dict_wstops = corpus_processing.make_dictionary_memory_friendly(doc_paths, [], ignore_users,
                                                                  min_tokens=35, n_below=5, n_keep=30000)
    print("subr_dict_wstops created")
    
    # Create gensim dictionary object for subreddit (second one with stopwords taken out)
    subr_dict_nostops = corpus_processing.make_dictionary_memory_friendly(doc_paths, stoplist, ignore_users,
                                                                  min_tokens=35, n_below=5, n_keep=30000)
    print("subr_dict_nostops created")
    
    # Append the subreddit-specific dictionary to the dict_list for later use (Doing both lists at one time).
    dict_list.append(subr_dict_wstops)
    dict_list.append(subr_dict_nostops)

    print('subreddit: ' + subreddit + ' has ' + str(len(subr_dict_wstops)) + ' unique postprocessing word types.')
    print('subreddit: ' + subreddit + ' has ' + str(len(subr_dict_nostops)) + ' unique postprocessing word types.')
    
    # Merge the subreddit-specific dictionaries into a single dictionary to allow easier combinations of corpora.
    #merged_dictionary = corpus_processing.make_universal_dictionary(dict_list)
    #print("merged_dictionary created")
    
    # Save the universal dictionary within the experiment directory.
    dictionary_path = os.path.join(exp_dir, 'dictionary_wstops.dict')
    corpus_processing.write_dictionary(dictionary_path, subr_dict_wstops)
    dictionary_path = os.path.join(exp_dir, 'dictionary_nostops.dict')
    corpus_processing.write_dictionary(dictionary_path, subr_dict_nostops)
    print("Dictionaries saved")

    # Use the memory-friendly class to create the bag-of-words corpus
    subr_corpus_wstops = corpus_processing.MyCorpus(subr_dict_wstops, doc_paths, [], ignore_users)
    print("subr_corpus_wstops created")
    subr_corpus_nostops = corpus_processing.MyCorpus(subr_dict_nostops, doc_paths, stoplist, ignore_users)
    print("subr_corpus_wlist created")

    # Make a subreddit-specific directory in corpora_dir.
    subr_corpus_dir = os.path.join(corpora_dir, subreddit)
    corpus_processing.create_dir(subr_corpus_dir)
    print("subreddit directory created")

    # Save the corpus object in the new directory.
    subr_corpus_path_wstops = os.path.join(subr_corpus_dir, 'corpus_wstops.mm')
    corpus_processing.write_corpus(subr_corpus_path_wstops, subr_corpus_wstops)
    subr_corpus_path_nostops = os.path.join(subr_corpus_dir, 'corpus_nostops.mm')
    corpus_processing.write_corpus(subr_corpus_path_nostops, subr_corpus_nostops)
    print("Corpus files saved")

    # Save document-level information that corresponds with the corpus object.
    subr_corpus_data_path_wstops = os.path.join(subr_corpus_dir, 'corpus_data_wstops.csv')
    subr_corpus_wstops.write_corpus_data(subr_corpus_data_path_wstops)
    subr_corpus_data_path_nostops = os.path.join(subr_corpus_dir, 'corpus_data_nostops.csv')
    subr_corpus_nostops.write_corpus_data(subr_corpus_data_path_nostops)
    print("Document information saved")
    