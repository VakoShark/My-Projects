'''
# This code is for LDA research with Dr. Stine, and most of this code belongs to him
# This code takes in 2 different corpora (with and without stopwords) and
  trains LDA models on them with different values of k (10, 50, 200, 500)
'''

import os
import numpy as np
import corpus_processing
import topic_modeling

if __name__ == '__main__':
    experiment_name = 'exp1'
    
    # Designate model ids, k values, and random ints for each round of models to be trained.
    model_ids = ['001', '002']
    k = 50;
    random_states = np.random.choice(1000, 4, replace=False)
    num_ids = len(model_ids)
    
    #assert len(model_ids) == len(random_states)
    
    # Fixed modeling parameters
    passes = 15
    eval_every = None
    iterations = 400
    subreddit = 'hinduism'
    
    # Directories
    cwd = os.getcwd()  # Directory of this program
    project_dir = os.path.dirname(cwd)  # Main project directory
    exp_dir = os.path.join(cwd, experiment_name)
    corpora_dir = os.path.join(exp_dir, 'Applied Data Science\\2_corpora')
    
    lda_dir = corpus_processing.make_dir(os.path.join(exp_dir, 'Applied Data Science\\3_lda'))
    
    # Dictionary is universal for each subreddit in this experiment, so only need to read it in once.
    dictionary_path = os.path.join(exp_dir, 'dictionary_wstops.dict')
    dictionary_wstops = corpus_processing.load_dictionary(dictionary_path)
    dictionary_path = os.path.join(exp_dir, 'dictionary_nostops.dict')
    dictionary_nostops = corpus_processing.load_dictionary(dictionary_path)
    print("dictionary_path created")
    
    # Load corpus.
    corpus_path_wstops = os.path.join(corpora_dir, subreddit, 'corpus_wstops.mm')
    corpus_wstops = corpus_processing.load_corpus(corpus_path_wstops)
    corpus_path_nostops = os.path.join(corpora_dir, subreddit, 'corpus_nostops.mm')
    corpus_nostops = corpus_processing.load_corpus(corpus_path_nostops)
    print("corpora loaded")

    # Load corpus data.
    corpus_data_path_wstops = os.path.join(corpora_dir, subreddit, 'corpus_data_wstops.csv')
    corpus_data_wstops = corpus_processing.read_corpus_data(corpus_data_path_wstops)
    corpus_data_path_nostops = os.path.join(corpora_dir, subreddit, 'corpus_data_nostops.csv')
    corpus_data_nostops = corpus_processing.read_corpus_data(corpus_data_path_nostops)
    print("corpora data loaded")
    
    # For each set of model specifications, train a model with the current subreddit.
    # This loop is WITHOUT STOPWORDS REMOVED
    for m_index, model_id in enumerate(model_ids):
        print('  model id: ' + model_id + ' (stopwords not removed)')
        #random_state = random_states[m_index]

        # Create model-specific directory
        model_dir = corpus_processing.make_dir(os.path.join(lda_dir, subreddit, model_id + '_k-' + str(k) + '_wstops'))

        # Specify file path for logging model progress
        log_path = os.path.join(model_dir, 'model.log')

        # For model training, shuffle the documents in the corpus.
        shuffled_indices_wstops = topic_modeling.get_shuffled_corpus_indices(corpus_wstops)
        shuffled_corpus_wstops = topic_modeling.ShuffledCorpus(corpus_wstops, shuffled_indices_wstops)
        print("indices shuffled")

        # Specify a model path.
        model_files_dir = corpus_processing.make_dir(os.path.join(model_dir, 'model_files'))
        model_path_wstops = os.path.join(model_files_dir, 'lda_model_wstops')
        print("model path created")

        # Train the model.
        model_wstops = topic_modeling.train_lda_model(shuffled_corpus_wstops, dictionary_wstops, k, random_states[m_index],
                                               log_path=log_path, passes=passes, iterations=iterations,
                                               eval_every=eval_every)
        print("model INCLUDING stopwords trained")

        # Save model to file.
        model_wstops.save(model_path_wstops)

        # Write some basic info about the model to file.
        model_description_path = os.path.join(model_dir, 'model_description.txt')

        model_description = {'k': k,
                             'random_state': random_states[m_index],
                             'passes': passes,
                             'iterations': iterations,
                             'eval_every': eval_every}
        print("model description written")

        topic_modeling.write_model_description(model_description_path, model_description)

        # Write highest probability words to file for reference.
        topic_modeling.write_top_words(model_wstops, model_dir, k, num_words=100)
        print("highest probability words written")

        # Create document-topic matrix using the regular corpus (not the shuffled corpus).
        topic_dist_dir = os.path.join(model_dir, 'topic_dists')
        corpus_processing.create_dir(topic_dist_dir)
        doc_topics_path = os.path.join(topic_dist_dir, subreddit + '_tdists.txt')
        doc_topics_wstops = topic_modeling.write_and_return_topic_dists_memory_friendly(model_wstops, corpus_wstops, k, doc_topics_path)
        print("document-topic matrix created")

        # Write topics summary file
        topic_summary_path_wstops = os.path.join(model_dir, 'topics_summary_wstops.csv')
        topic_modeling.write_topic_summary_file(topic_summary_path_wstops, model_wstops, k, doc_topics_wstops, num_words=20)
        print("topics summary file written")

        # Write lists of exemplary documents for each topic.
        topic_modeling.write_exemplary_docs(doc_topics_wstops, model_dir, k, corpus_data_wstops, num_docs=50)

        print('  ' + model_id + '_wstops done.')
        
    # For each set of model specifications, train a model with the current subreddit.
    # This loop is WITH STOPWORDS REMOVED
    for m_index, model_id in enumerate(model_ids):
        print('  model id: ' + model_id + ' (stopwords removed)')
        #random_state = random_states[m_index]

        # Create model-specific directory
        model_dir = corpus_processing.make_dir(os.path.join(lda_dir, subreddit, model_id + '_k-' + str(k) +'_nostops'))

        # Specify file path for logging model progress
        log_path = os.path.join(model_dir, 'model.log')

        # For model training, shuffle the documents in the corpus.
        shuffled_indices_nostops = topic_modeling.get_shuffled_corpus_indices(corpus_nostops)
        shuffled_corpus_nostops = topic_modeling.ShuffledCorpus(corpus_nostops, shuffled_indices_nostops)
        print("indices shuffled")

        # Specify a model path.
        model_files_dir = corpus_processing.make_dir(os.path.join(model_dir, 'model_files'))
        model_path_nostops = os.path.join(model_files_dir, 'lda_model_nostops')
        print("model path created")

        # Train the model.
        model_nostops = topic_modeling.train_lda_model(shuffled_corpus_nostops, dictionary_nostops, k, random_states[m_index+num_ids],
                                               log_path=log_path, passes=passes, iterations=iterations,
                                               eval_every=eval_every)
        print("model NOT INCLUDING stopwords trained")

        # Save model to file.
        model_nostops.save(model_path_nostops)

        # Write some basic info about the model to file.
        model_description_path = os.path.join(model_dir, 'model_description.txt')

        model_description = {'k': k,
                             'random_state': random_states[m_index+num_ids],
                             'passes': passes,
                             'iterations': iterations,
                             'eval_every': eval_every}
        print("model description written")

        topic_modeling.write_model_description(model_description_path, model_description)

        # Write highest probability words to file for reference.
        topic_modeling.write_top_words(model_nostops, model_dir, k, num_words=100)
        print("highest probability words written")

        # Create document-topic matrix using the regular corpus (not the shuffled corpus).
        topic_dist_dir = os.path.join(model_dir, 'topic_dists')
        corpus_processing.create_dir(topic_dist_dir)
        doc_topics_path = os.path.join(topic_dist_dir, subreddit + '_tdists.txt')
        doc_topics_nostops = topic_modeling.write_and_return_topic_dists_memory_friendly(model_nostops, corpus_nostops, k, doc_topics_path)
        print("document-topic matrix created")

        # Write topics summary file
        topic_summary_path_nostops = os.path.join(model_dir, 'topics_summary_nostops.csv')
        topic_modeling.write_topic_summary_file(topic_summary_path_nostops, model_nostops, k, doc_topics_nostops, num_words=20)
        print("topics summary file written")

        # Write lists of exemplary documents for each topic.
        topic_modeling.write_exemplary_docs(doc_topics_nostops, model_dir, k, corpus_data_nostops, num_docs=50)

        print('  ' + model_id + '_nostops done.')