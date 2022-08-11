import find_tfl_attention
from DataBase import DataBase
from Kernel import Kernel
import data_utils as du
import train_demo as td

from crop_validation import crops_validation

if __name__ == '__main__':
    find_tfl_attention.run_attention()
    get_zoom_rect()
    crops_validation()
    # DataBase().export_tfls_coordinates_to_h5()
    # DataBase().export_tfls_decisions_to_h5()
    train_dataset = du.TrafficLightDataSet('Resources', 'Resources/leftImg8bit/train')
    test_dataset = du.TrafficLightDataSet('Resources', 'Resources/leftImg8bit/test', is_train=False)
    NN = du.ModelManager.make_empty_model()
    x = td.train_a_model(NN, train_dataset, test_dataset, log_dir='Resources/log_dir', num_epochs=50)
    td.examine_my_results("Resources", 'Resources/leftImg8bit/train', 'Resources/log_dir/model.pkl', test_dataset)
