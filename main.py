import sys
import yaml
import pandas as pd

from processData import processData
def run_pipeline(run_config):
    # create a dataframe using texts and lables
    trainDF = pd.DataFrame()
    # trainDF['text'] = texts
    # trainDF['label'] = labels
    data = processData(run_config).feature()


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print(__doc__)
        exit()
    else:
        run_yamlfile = sys.argv[1]
        run_config = yaml.load(open(run_yamlfile))
        print(run_config)
    run_pipeline(run_config)