import train as tr

def polish_pickle(feature_params, test_accuracy):
    filename = tr.make_pickle_filename('trained_models', feature_params, test_accuracy)
    data = tr.load_data(filename)
    data['feature_params'] = feature_params
    tr.save_data(data, filename)
    
    # Check feature_params were saved correctly.
    data = tr.load_data(filename)
    print('feature_params:', data['feature_params'].str())

if __name__ == '__main__':
    fp = tr.get_defaults()
    fp.pix_per_cell = 10
    polish_pickle(fp, 99.32)
    
    fp = tr.get_defaults()
    fp.color_space = 'HSV'
    polish_pickle(fp, 99.32)