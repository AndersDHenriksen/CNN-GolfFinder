do_plot = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    do_plot = False

import numpy as np
from tensorflow.python.keras import backend as K

test_randomly = True


def GolfBallPrediction(sess, model, data, config):
    # Load data
    idx = np.random.choice(data.y_test.shape[0], 1) if test_randomly else int(config.do_predict)
    x = data.input_test[[idx]]
    y = data.y_test[[idx]]

    # Eval data
    feed_dict = {model.x: x, model.y: y, model.is_training: False, K.learning_phase(): 0}
    y_predict, loss, acc = sess.run([model.y_out, model.squared_error, model.accuracy], feed_dict=feed_dict)

    if do_plot:
        # Plot result
        plt.figure()
        plt.imshow(x[0, :, :, :3])
        plt.plot(*y_predict[0], 'xr')
        circ = plt.Circle(y_predict[0], 11, color='r', fill=False, linestyle='--')
        plt.gca().add_artist(circ)
        plt.title("Accuracy: {:.2f}".format(acc))
        plt.show()
        # _ = "breakpoint"
        # plt.close('all')