import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.abspath(""))))
from utils import *
from cem.concept_discovery.utils import *
import cem.data.dsprites as dsprites
import cem.data.mnist_add as mnist

dsprites_permutation = np.load("../dsprites_permutation.npy")

dsprites_train_dl, dsprites_val_dl, dsprites_test_dl = dsprites.load_dsprites(
    "quadrant_shape_shape_hidden",
    "../../../../datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    concepts_and_label=dsprites.quadrant_shape_shape_hidden,
    permutation=dsprites_permutation
)

mnist.set_root_dir("../../../../datasets/mnist")

dsprites_model, dsprites_model_test_results = train_dsprites_model("quadrant_shape_shape_hidden", 4, 12, save_path="dsprites_model")
_, dsprites_embs, _ = calculate_embeddings(dsprites_model, [dsprites_train_dl], 16)
np.save("dsprites_embs.npy", dsprites_embs)

mnist_two_digits_model, mnist_two_digits_model_test_results = train_mnist_model(2, 1, "mnist_two_digits_model")
_, mnist_two_digits_embs, _ = calculate_embeddings(mnist_two_digits_model, [mnist.train_dl(2, 1)], 16)
np.save("mnist_two_digits_embs.npy", mnist_two_digits_embs)

mnist_five_digits_model, mnist_five_digits_model_test_results = train_mnist_model(5, 1, "mnist_five_digits_model")
_, mnist_five_digits_embs, _ = calculate_embeddings(mnist_five_digits_model, [mnist.train_dl(5, 1)], 16)
np.save("mnist_five_digits_embs.npy", mnist_five_digits_embs)

mnist_twenty_five_digits_model, mnist_twenty_five_digits_model_test_results = train_mnist_model(25, 1, "mnist_twenty_five_digits_model")
_, mnist_twenty_five_digits_embs, _ = calculate_embeddings(mnist_twenty_five_digits_model, [mnist.train_dl(25, 1)], 16)
np.save("mnist_twenty_five_digits_embs.npy", mnist_twenty_five_digits_embs)
