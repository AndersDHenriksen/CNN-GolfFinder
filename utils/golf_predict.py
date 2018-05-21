import matplotlib.pyplot as plt
import numpy as np

def GolfBallPrediction(sess, model, data, config):
    # load data
    idx = int(config.predict_instead_of_training)
    x = data.input[[idx]]
    y = data.y[[idx]]

    # Eval data
    feed_dict = {model.x: x, model.y: y, model.is_training: False}
    y_predict = sess.run(model.y_out, feed_dict=feed_dict)

    plt.figure()
    plt.imshow(x[0, :, :, :3])
    plt.plot(*y_predict[0], 'xr')
    circ = plt.Circle(y_predict[0], 11, color='r', fill=False, linestyle='--')
    plt.gca().add_artist(circ)
    plt.show()
    _ = "breakpoint"
    plt.close('all')