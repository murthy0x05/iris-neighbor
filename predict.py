import sys
import joblib
import numpy as np


def main(argv):
    if len(argv) != 4:
        print('Usage: predict.py sepal_length sepal_width petal_length petal_width')
        return 2
    sample = [float(x) for x in argv]
    model = joblib.load('model.joblib')
    pred = model.predict(np.array(sample).reshape(1, -1))[0]
    print(pred)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
