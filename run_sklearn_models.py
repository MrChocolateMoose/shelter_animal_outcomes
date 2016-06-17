import numpy as np
import pandas as pd
import sklearn.preprocessing

from sklearn.metrics import log_loss

def run_sklearn_models(data_sets, output_kaggle_predictions):

#{'learning_rate': 0.01, 'max_depth': 5, 'min_samples_leaf': 10}
# {'learning_rate': 0.02, 'max_depth': 3, 'min_samples_leaf': 5}

# enhanced
# {'n_estimators': 250, 'learning_rate': 0.05, 'max_depth': 4, 'min_samples_leaf': 15}
# {'n_estimators': 250, 'learning_rate': 0.02, 'max_depth': 4, 'min_samples_leaf': 10}

    train_mean_acc_vec = []
    test_mean_acc_vec = []
    train_log_loss_vec = []
    test_log_loss_vec = []
    submission_data_frame_vec = []

    for data_set in data_sets:

        # get partitioned data sets
        train_data, test_data, validation_data = data_set.get_data()

        # get train x,y, and id
        train_data_x, train_data_y, train_data_id = train_data

        # get test x,y, and id
        test_data_x,  test_data_y, test_data_id  = test_data

        # get validation x,y and id
        #validation_data_x, validation_data_y, validation_data_id = validation_data

        models = data_set.get_models()


        test_prob_ys = []
        for model in models:

            #train this data set on x and y
            model.fit(train_data_x, train_data_y)

            #record log loss
            train_log_loss_vec.append(
                log_loss(train_data_y, model.predict_proba(train_data_x)))
            #record accuracy
            train_mean_acc_vec.append(
                model.score(train_data_x, train_data_y))

            #agglomerate
            test_prob_ys.append(
                model.predict_proba(test_data_x))



        test_prob_y = np.mean(np.array(test_prob_ys), axis=0)
        test_prob_y = sklearn.preprocessing.normalize(test_prob_y, norm='l1', axis=1)


        partial_submission_data_frame = pd.DataFrame(test_prob_y)
        partial_submission_data_frame.columns = data_set.y_categories

        partial_submission_data_frame.insert(0, "ID", test_data_id.values)

        submission_data_frame_vec.append(partial_submission_data_frame)

        if not output_kaggle_predictions:

            test_mean_acc_vec.append(
                model.score(test_data_x, test_data_y))

            test_log_loss_vec.append(
                log_loss(test_data_y, test_prob_y))


    submission_data_frame = pd.concat(submission_data_frame_vec)
    submission_data_frame.sort(columns="ID", inplace=True)

    #print(submission_data_frame)


    #print(model.__class__.__name__,
    #              "Mean Train Accuracy:", train_mean_acc_vec, "Train Log Loss", train_log_loss_vec,
    #              "Mean Test Accuracy:", test_mean_acc_vec, "Test Log Loss", test_log_loss_vec)

    print("Mean Train Accuracy:", np.mean(train_mean_acc_vec), "Train Log Loss", np.mean(train_log_loss_vec),
          "Mean Test Accuracy:", np.mean(test_mean_acc_vec), "Test Log Loss", np.mean(test_log_loss_vec))


    if output_kaggle_predictions:
        submission_data_frame.to_csv("data/" + "submission" + ".csv", index=False)

