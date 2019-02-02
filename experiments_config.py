"""Module containing experiments configuration."""
results_path = '/home/arccha/fungus_results/'
seed = 9001
batch_size = 32
number_of_bg_slices_per_image = 1
number_of_fg_slices_per_image = 16

fv_param_grid = [
    {
        'fisher_vector__gmm_samples_number': [10000],
        'fisher_vector__gmm_clusters_number': [5, 10, 20, 50],
        'svc__C': [1, 10, 100, 1000],
        'svc__kernel': ['linear']
    },
    {
        'fisher_vector__gmm_samples_number': [10000],
        'fisher_vector__gmm_clusters_number': [5, 10, 20, 50],
        'svc__C': [1, 10, 100, 1000],
        'svc__gamma': [0.001, 0.0001],
        'svc__kernel': ['rbf']
    }
]

bow_param_grid = [
    {
        'bag_of_words__samples_number': [10000],
        'bag_of_words__clusters_number': [5, 10, 20, 50, 100, 200, 500],
        'svc__C': [1, 10, 100, 1000],
        'svc__kernel': ['linear'],
    },
    {
        'bag_of_words__samples_number': [10000],
        'bag_of_words__clusters_number': [5, 10, 20, 50, 100, 200, 500],
        'svc__C': [1, 10, 100, 1000],
        'svc__gamma': [0.001, 0.0001],
        'svc__kernel': ['rbf']
    }
]
