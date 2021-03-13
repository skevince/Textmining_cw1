from run import model_run, build_model_and_dataset, model_ensemble_run
import question_classifier
import torch


# train model
def train(glove=False, biLSTM=False, freeze=False):
    # ngpu = 1
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # print(device)
    # print(torch.cuda.get_device_name(0))

    if_GPU = False
    # print('if_GPU:', if_GPU)
    # if if_GPU:
    #     device = device
    # else:
    #     device = None
    device = None

    batch_size = int(question_classifier.get_config('Model', 'batch_size'))
    train_epoch = int(question_classifier.get_config('Model', 'epoch'))
    dev_epoch = int(question_classifier.get_config('Model', 'dev_epoch'))

    model_save_path = question_classifier.get_config('Model', 'model_save_path')
    # build model and datasetloader
    model, traindataloader, devdataloader, testdataloader, label_list = build_model_and_dataset(if_glove=glove,
                                                                                                if_biLSTM=biLSTM,
                                                                                                if_freeze=freeze,
                                                                                                batch_size=batch_size,
                                                                                                device=device)
    if if_GPU:
        model = model.to(device)
    print('train start')
    model, train_loss, train_acc = model_run(model, run_type='Train', epoch_range=train_epoch, batch_size=batch_size,
                                             device=device, dataloader=traindataloader)
    print('train finish')

    print('develop start')
    model, dev_loss, dev_acc = model_run(model, run_type='Develop', epoch_range=dev_epoch, batch_size=batch_size,
                                         device=device, dataloader=devdataloader)
    print('develop finish')
    torch.save(model, model_save_path)
    return


# test model
def test(if_glove=False, confu_matrix=False):
    # ngpu = 1
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # print(device)
    # print(torch.cuda.get_device_name(0))
    model_save_path = question_classifier.get_config('Model', 'model_save_path')
    path_eval_result = question_classifier.get_config('PATH', 'path_eval_result')
    if_GPU = False
    # print('if_GPU:', if_GPU)
    # if if_GPU:
    #     device = device
    # else:
    #    device = None
    device = None

    test_batch_size = int(question_classifier.get_config('Model', 'test_batch_size'))
    # build model and datasetloader
    model, traindataloader, devdataloader, testdataloader, label_list = build_model_and_dataset(if_glove=if_glove,
                                                                                                if_biLSTM=True,
                                                                                                if_freeze=True,
                                                                                                batch_size=test_batch_size,
                                                                                                device=device)
    model = torch.load(model_save_path, map_location='cpu')
    if if_GPU:
        model = model.to(device)
    print('test start')
    prediction, true_label, test_acc = model_run(model, run_type='Test', batch_size=test_batch_size,
                                                 device=device, dataloader=testdataloader)

    output_file = open(path_eval_result, 'w+')
    for i in range(len(prediction)):
        print(label_list[int(prediction[i])], file=output_file)
    print('test_accuracy ', test_acc, file=output_file)
    output_file.close()
    print('test_accuracy：', test_acc)
    print('test finish')
    if confu_matrix:
        from sklearn.metrics import confusion_matrix
        classes = label_list
        confusion_matrix = confusion_matrix(true_label, prediction, labels=range(len(classes)))
        plot_confu_matrix(confusion_matrix=confusion_matrix, classes=classes, label_list=label_list)

    return


# test ensemble model
def ensemble(confu_matrix=False):
    # ngpu = 1
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # print(device)
    # print(torch.cuda.get_device_name(0))
    ensemble_size = int(question_classifier.get_config('Ensemble', 'ensemble_size'))
    emsemble_path_model = question_classifier.get_config('Ensemble', 'emsemble_path_model')
    path_eval_result = question_classifier.get_config('PATH', 'path_eval_result')
    test_batch_size = int(question_classifier.get_config('Model', 'test_batch_size'))
    if_GPU = False
    # print('if_GPU:', if_GPU)
    # if if_GPU:
    #     device = device
    # else:
    #     device = None
    device = None
    #build model and datasetloader
    model, traindataloader, devdataloader, testdataloader, label_list = build_model_and_dataset(True,
                                                                                                True,
                                                                                                True,
                                                                                                batch_size=test_batch_size,
                                                                                                device=device)

    print('ensemble test start')
    prediction, true_label, test_acc = model_ensemble_run(ensemble_size, emsemble_path_model,
                                                          device=device, dataloader=testdataloader)
    output_file = open(path_eval_result, 'w+')
    for i in range(len(prediction)):
        print(label_list[int(prediction[i])], file=output_file)
    print('ensemble test_accuracy ', test_acc, file=output_file)
    output_file.close()
    print('ensemble test_accuracy：', test_acc)
    print('ensemble test finish')

    if confu_matrix:
        from sklearn.metrics import confusion_matrix
        classes = label_list
        confusion_matrix = confusion_matrix(true_label, prediction, labels=range(len(classes)))
        plot_confu_matrix(confusion_matrix=confusion_matrix, classes=classes, label_list=label_list)
    return


# train bowandlstm model
def bowandlstm_train(glove=False, biLSTM=True, freeze=False):
    # ngpu = 1
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # print(device)
    # print(torch.cuda.get_device_name(0))

    if_GPU = False
    # print('if_GPU:', if_GPU)
    # if if_GPU:
    #     device = device
    # else:
    #     device = None
    device = None
    batch_size = int(question_classifier.get_config('Model', 'batch_size'))
    train_epoch = int(question_classifier.get_config('Model', 'epoch'))
    dev_epoch = int(question_classifier.get_config('Model', 'dev_epoch'))

    model_save_path = question_classifier.get_config('Model', 'model_save_path')
    # build model and datasetloader
    model, traindataloader, devdataloader, testdataloader, label_list = build_model_and_dataset(if_glove=glove,
                                                                                                if_biLSTM=biLSTM,
                                                                                                if_freeze=freeze,
                                                                                                batch_size=batch_size,
                                                                                                device=device,
                                                                                                bowandlstm=True)
    if if_GPU:
        model = model.to(device)
    print('train start')
    model, train_loss, train_acc = model_run(model, run_type='Train', epoch_range=train_epoch, batch_size=batch_size,
                                             device=device, dataloader=traindataloader)
    print('train finish')

    print('develop start')
    model, dev_loss, dev_acc = model_run(model, run_type='Develop', epoch_range=dev_epoch, batch_size=batch_size,
                                         device=device, dataloader=devdataloader)
    print('develop finish')
    torch.save(model, model_save_path)


# plot confusion matrix
def plot_confu_matrix(confusion_matrix=None, classes=None, label_list=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=270)
    plt.yticks(tick_marks, classes)
    iters = np.reshape([[[i, j] for j in range(len(label_list))] for i in range(len(label_list))],
                       (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confusion_matrix[i, j]), fontsize=6, verticalalignment='center',
                 horizontalalignment='center')

    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()
