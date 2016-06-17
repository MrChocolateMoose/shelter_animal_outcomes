import numpy as np
import theano
import os
import sys
import timeit

sys.path.append(os.path.abspath('../DeepLearningTutorials/code'))
from DBN import DBN

def run_dbn_model(data_set,
                  finetune_lr=0.1, pretraining_epochs=10,
                  pretrain_lr=0.01, k=1, training_epochs=1000,
                  batch_size=10):


    # get partitioned data sets
    train_data, test_data, validation_data = data_set.get_data()

    # get train x,y, and id
    train_data_x, train_data_y, train_data_id = train_data

    train_ten_x, train_ten_y = theano.shared(np.asarray(train_data_x, dtype=theano.config.floatX)), \
                               theano.shared(np.asarray(train_data_y, dtype='int32'))
                               #theano.shared(np.asarray(pd.get_dummies(train_data_y).as_matrix(), dtype='int32'))

    # get test x,y, and id
    test_data_x,  test_data_y, test_data_id  = test_data

    test_ten_x, test_ten_y = theano.shared(np.asarray(test_data_x, dtype=theano.config.floatX)), \
                             theano.shared(np.asarray(test_data_y, dtype='int32'))
                             #theano.shared(np.asarray(pd.get_dummies(test_data_y).as_matrix(), dtype='int32'))

    # get validation x,y and id
    validation_data_x, validation_data_y, validation_data_id = validation_data

    validation_ten_x, validation_ten_y = theano.shared(np.asarray(validation_data_x, dtype=theano.config.floatX)), \
                                         theano.shared(np.asarray(validation_data_y, dtype='int32'))
                                         #theano.shared(np.asarray(pd.get_dummies(validation_data_y).as_matrix(), dtype='int32'))

    ten_data_set = [(train_ten_x,train_ten_y), (validation_ten_x, validation_ten_y), (test_ten_x, test_ten_y)]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_ten_x.get_value(borrow=True).shape[0] / batch_size
    print ("n_train_batches: " + str(n_train_batches))

    cols = train_data_x.shape[1]
    assert(train_data_x.shape[1] == test_data_x.shape[1]  and test_data_x.shape[1] == validation_data_x.shape[1])
    print("cols: " + str(cols))

    print("train_x" + str(train_ten_x.get_value(borrow=True).shape))
    print("train_y" + str(train_ten_y.get_value(borrow=True).shape))
    print("valid_x" + str(validation_ten_x.get_value(borrow=True).shape))
    print("valid_y" + str(validation_ten_y.get_value(borrow=True).shape))
    print("test_x" + str(test_ten_x.get_value(borrow=True).shape))
    print("test_y" + str(test_ten_y.get_value(borrow=True).shape))

    # numpy random generator
    numpy_rng = np.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins= cols,
              hidden_layers_sizes=[cols*10, cols*10, cols*10],
              n_outs=5)

    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_ten_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    for i in range(dbn.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print np.mean(c)

    end_time = timeit.default_timer()
    # end-snippet-2
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
        datasets=ten_data_set,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

           # if patience <= iter:
           #     done_looping = True
           #     break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'obtained at iteration %i, '
            'with test performance %f %%'
        ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))
