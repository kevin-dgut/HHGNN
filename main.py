import sys



import os
from sklearn.utils import compute_class_weight
from sklearn.model_selection import StratifiedKFold

from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from HHGNN.dataload import Logger,DataLoader
from HHGNN.opt import *
from metrics import accuracy, auc, prf, metrics
from HHGNN.model import HeteroGNN





if __name__ == '__main__':

    opt = parase_opt()
    args = opt.initialize()

    task_name = opt.args.task
    log_filename = os.path.join(opt.args.log_path, f"log.txt")
    log_dir = os.path.dirname(log_filename)

    log = Logger(log_filename)
    sys.stdout = log
    dl = DataLoader(opt)

    batch_size = opt.args.batch_size

    graphs,labels = dl.get_all_data()



    n_folds = opt.args.n_folds

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kf.split(graphs, labels))

    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32)
    sens = np.zeros(n_folds, dtype=np.float32)
    spes = np.zeros(n_folds, dtype=np.float32)
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds, 3], dtype=np.float32)



    for fold in range(n_folds):


        train_idx = splits[fold][0]
        test_idx = splits[fold][1]

        print("\r\n========================== Fold {} ==========================".format(fold))

        train_loader = dl.get_fold_batches(train_idx, batch_size, shuffle=True)
        test_loader = dl.get_fold_batches(test_idx, batch_size, shuffle=False)
        fold_model_path =os.path.join( opt.args.best_model_path , f"{opt.args.task}{fold}.pth")
        model = HeteroGNN(in_channels=1500,hidden_channels=128, out_channels=32, num_classes=opt.args.num_classes)
        model = model.to(opt.args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=opt.args.lr,weight_decay=opt.args.wd)
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(opt.args.device)
        criterion = CrossEntropyLoss(weight=class_weights_tensor)



        def train():
            model.train()
            best_acc=0
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            for epoch in range(opt.args.epochs):
                epoch_loss = 0
                all_preds = []
                all_labels = []


                for batch_idx, (batch_graph, batch_label) in enumerate(train_loader):
                    batch_graph = batch_graph.to(opt.args.device)
                    batch_label = batch_label.to(opt.args.device)
                    optimizer.zero_grad()
                    output = model(batch_graph)
                    loss = criterion(output, batch_label)
                    loss.backward()

                    optimizer.step()
                    batch_size=batch_label.shape[0]
                    epoch_loss += loss.item()*batch_size
                    _, predicted = output.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_label.cpu().numpy())

                epoch_loss /= len(train_loader.dataset)

                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                _, epoch_accuracy = accuracy(all_preds, all_labels)



                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_accuracy)


                print()
                print(f"Epoch {epoch + 1}/{opt.args.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")


                model.eval()

                val_loss = 0
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for batch_graph, batch_label in test_loader:

                        batch_graph = batch_graph.to(opt.args.device)
                        batch_label = batch_label.to(opt.args.device)
                        output = model(batch_graph)
                        _, preds = torch.max(output, 1)



                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch_label.cpu().numpy())



                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                correct_test,acc_test = accuracy(all_preds, all_labels)
                sen_test, spe_test = metrics(all_preds, all_labels)
                auc_score_test = auc(all_preds, all_labels)
                precision_test, recall_test, f1_test = prf(all_preds, all_labels)

                print(f"Validation Accuracy : {acc_test:.4f}")


                val_losses.append(val_loss)
                val_accuracies.append(acc_test)

                if acc_test > best_acc:
                    best_acc = acc_test

                    sens[fold] = sen_test
                    spes[fold] = spe_test
                    aucs[fold] = auc_score_test
                    prfs[fold] = (precision_test, recall_test, f1_test)

                    best_model = model.state_dict()
                    os.makedirs(os.path.dirname(fold_model_path), exist_ok=True)

                    if best_model is not None:
                        torch.save(best_model, fold_model_path)


            accs[fold] = best_acc




        def evaluate():
            print('  Start testing...')
            model.load_state_dict(torch.load(fold_model_path))
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch_graph, batch_label in test_loader:
                    batch_graph = batch_graph.to(opt.args.device)
                    batch_label = batch_label.to(opt.args.device)
                    output = model(batch_graph)
                    _, preds = torch.max(output, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_label.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            correct, acc = accuracy(all_preds, all_labels)
            sen, spe = metrics(all_preds, all_labels)
            auc_score = auc(all_preds, all_labels)
            precision, recall, f1 = prf(all_preds, all_labels)

            accs[fold] = acc
            sens[fold] = sen
            spes[fold] = spe
            aucs[fold] = auc_score
            prfs[fold] = (precision, recall, f1)
            print(
                "Fold {} test accuracy: {:.5f}, test AUC: {:.5f}".format(
                    fold, accs[fold], aucs[fold]
                ))

        if opt.args.train == 1:
            train()
        elif opt.args.train == 0:
            evaluate()



    print("\r\n========================== Finish ==========================")
    print("=> Average test accuracy in {}-fold CV: {:.4f}({:.4f})".format(n_folds, np.mean(accs), np.std(accs)))
    print("=> Average test sensitivity in {}-fold CV: {:.4f}({:.4f})".format(n_folds, np.mean(sens), np.std(sens)))
    print("=> Average test specificity in {}-fold CV: {:.4f}({:.4f})".format(n_folds, np.mean(spes), np.std(spes)))
    print("=> Average test AUC in {}-fold CV: {:.4f}({:.3})".format(n_folds, np.mean(aucs), np.std(aucs)))
    _, _, f1 = np.mean(prfs, axis=0)
    _, _, f1_std = np.std(prfs, axis=0)
    print("=>F1-score {:.4f}({:.4f})".format(
         f1, f1_std
    ))

