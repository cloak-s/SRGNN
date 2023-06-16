import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=50)             # 16
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=5)  #
    parser.add_argument('--dataset', type=str, default="amazon_photo")
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--hidden', type=int, default=64)

    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.15)

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=5e-5)

    parser.add_argument('--num_train_class', type=int, default=20)
    parser.add_argument('--num_val_class', type=int, default=30)
    parser.add_argument('--gama', type=float, default=0.15)  # 0.6
    parser.add_argument('--drop_prob', type=float, default=0.2)  #
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--early_stopping', type=bool, default=True)
    return parser.parse_args()

