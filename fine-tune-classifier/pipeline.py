import preprocessing
import feature_extraction
import train

import warnings
warnings.filterwarnings("ignore")

preprocessing.main()
feature_extraction.main()
train.main()

print("Run complete!")
