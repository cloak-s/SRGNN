from push_forward import topk_ppr_matrix, feature_fusion
import time
from seeds import test_seeds
import torch
from utils import get_data, bootstrapping, accuracy, get_mask, f1
import numpy as np
from early_stop import EarlyStopping, Stop_args
from models import MLP
from arguments import parse_args

args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature, adj, labels, _ = get_data(args.dataset)
model = MLP(in_channel=feature.shape[1],
            hidden=args.hidden,
            num_class=labels.max().item() + 1,
            drop_prob=args.drop_prob)
print(model)
print("Parameters numbers:", sum(p.numel() for p in model.parameters()))
criterion = torch.nn.CrossEntropyLoss()


def train_model(model, optimizer, train_features, val_features, labels, idx_train, idx_val, weight_decay):
    # train_epoch = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    out_put = model(train_features)
    loss_train = criterion(out_put, labels[idx_train]) + weight_decay * torch.sum(model.lin1.weight ** 2) / 2

    loss_train.backward()
    optimizer.step()
    acc_train = accuracy(out_put, labels[idx_train])
    # train_end_epoch = time.perf_counter() - train_epoch
    # print("the time of each epoch:{:.4f}".format(train_end_epoch))

    # validation
    model.eval()
    output = model(val_features)
    loss_val = criterion(output, labels[idx_val])
    acc_val = accuracy(output, labels[idx_val])
    return loss_val.item(), acc_val.item()


loss = []
acc = []
macro_score = []
count = 0
avg_time = []
best_epochs = []

for seed in test_seeds:
    count += 1
    model = model.to(device)

    idx_train, idx_val, idx_test = get_mask(graph_name=args.dataset, seed=seed, labels=labels,
                                            train_per_class=args.num_train_class,
                                            val_per_class=args.num_val_class, Fixed_total=True)

    if count == 1:
        print("训练集数量:", idx_train.shape)
        print("验证集数量:", idx_val.shape)
        print("测试集数量:", idx_test.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)
    model.reset_parameters()

    # Pre-processing
    pre_process_time = time.time()
    train_ppr = topk_ppr_matrix(adj, args.alpha, args.epsilon, idx_train, args.k, "row")
    train_data = feature_fusion(train_ppr.indices, train_ppr.indptr, idx_train, args.gama, feature)
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)

    val_ppr = topk_ppr_matrix(adj, args.alpha, args.epsilon, idx_val, args.k, "row")
    val_data = feature_fusion(val_ppr.indices, val_ppr.indptr, idx_val, args.gama, feature)
    val_data = torch.tensor(val_data, dtype=torch.float32).to(device)
    # val_data = torch.tensor(feature_fusion(val_ppr.indices, val_ppr.indptr, idx_val, args.gama, feature),
    #                         dtype=torch.float32).to(device)

    pre_process_time = time.time() - pre_process_time

    idx_train = torch.tensor(idx_train)
    idx_val = torch.tensor(idx_val)
    labels = labels.to(device)

    train_infer_time = time.time()
    for epoch in range(args.epochs):
        # train_epoch = time.perf_counter()
        loss_val, acc_val = train_model(model, optimizer, train_data, val_data,
                                        labels, idx_train, idx_val, args.weight_decay)
        # train_end_epoch = time.perf_counter() - train_epoch
        # print("the time of each epoch:{:.2f}".format(train_end_epoch * 1000))
        if early_stopping.check([acc_val, loss_val], epoch):
            break
    train_and_infer_time = time.time() - train_infer_time
    model.load_state_dict(early_stopping.best_state)

    # inference
    model.eval()
    test_ppr = topk_ppr_matrix(adj, args.alpha, args.epsilon, idx_test, args.k, "row")
    test_data = torch.tensor(feature_fusion(test_ppr.indices, test_ppr.indptr, idx_test, args.gama, feature)
                             , dtype=torch.float32)
    idx_test = torch.tensor(idx_test)
    model = model.to('cpu')
    labels = labels.to('cpu')
    output = model(test_data)
    # loss_test = criterion(output, labels[idx_test])
    acc_test = accuracy(output, labels[idx_test])
    micro, macro = f1(output, labels[idx_test])

    # CPU_memory = get_max_memory_bytes()

    print("count:", count,
          " seed:", seed,
          " train_time={:.2f}".format(train_and_infer_time),
          " preprocess_data = {:.4f}".format(pre_process_time),
          # " best_epoch:", early_stopping.best_epoch,
          "  -------->Test set results:",
          # "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "macro={:.4f}".format(macro),
          "cuda memory = {:.3f} GBs".format(torch.cuda.max_memory_allocated() / 1024 ** 3),
          # "cpu memory = {:.3f} GBs".format(CPU_memory / 1024 ** 3)
          )

    # loss.append(loss_test.item())
    acc.append(micro.item())
    avg_time.append(train_and_infer_time)
    macro_score.append(macro)
    best_epochs.append(early_stopping.best_epoch)
print("avg time: {:.2f}".format(np.mean(avg_time)))
print("best epoch:", int(np.mean(best_epochs))+100)
bootstrapping(acc)
bootstrapping(macro_score)
