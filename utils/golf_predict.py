import matplotlib.pyplot as plt
import numpy as np

test_randomly = True


def GolfBallPrediction(sess, model, data, config):
    # Load data
    idx = np.random.choice(data.y_test.shape[0], 1) if test_randomly else int(config.do_predict)
    x = data.input_test[[idx]]
    y = data.y_test[[idx]]

    # Eval data
    feed_dict = {model.x: x, model.y: y, model.is_training: False}
    y_predict, loss, acc = sess.run([model.y_out, model.squared_error, model.accuracy], feed_dict=feed_dict)

    # Plot result
    plt.figure()
    plt.imshow(x[0, :, :, :3])
    plt.plot(*y_predict[0], 'xr')
    circ = plt.Circle(y_predict[0], 11, color='r', fill=False, linestyle='--')
    plt.gca().add_artist(circ)
    plt.title(f"Accuracy: {acc:.2f}")
    plt.show()
    # _ = "breakpoint"
    # plt.close('all')