import datetime
import logging
import pathlib
import zipfile
import torch.optim as optim
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataloader import *
from model import *
from param_parser import parameter_parser
from utils import *
from utils import _init_fn


# %% ====================== Definition of the main train module ======================
def train(args):
    # %% ====================== Experiment settings ======================
    # Path of saving results of experiments
    start = datetime.datetime.now()
    best = datetime.datetime.now()
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Path of saving codes of experiments
    args.code_dir = "codes/"
    if not os.path.exists(args.code_dir):
        os.makedirs(args.code_dir)

    # Settings of the logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"), filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Saving the parameters of experiments
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Flag of whether to save the model's code
    if args.store_code:
        zipf = zipfile.ZipFile(os.path.join(args.code_dir, args.name + '.zip'), 'w', zipfile.ZIP_DEFLATED)
        zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
        zipf.close()

    # %% ====================== Load data ======================
    # Loading train/val/test data, respectively
    print('1. Loading train/val/test data...')
    train_data = pd.read_csv(args.data_train)
    val_data = pd.read_csv(args.data_val)
    test_data = pd.read_csv(args.data_test)

    # Loading dicts needed below
    user_id2idx_dict = get_map_dict(args.user_map)
    user_list = list(user_id2idx_dict.keys())
    poi_id2idx_dict = get_map_dict(args.poi_map)
    hotness2idx_dict = get_map_dict(args.hotness_map)
    region2idx_dict = get_map_dict(args.region_map)
    poi_features = get_poi_features(args.poi_features)

    # Creating train/val/test dataset
    print('2. Creating train/val/test dataset...')
    train_dataset = TrajectoryDatasetTrain(args, train_data, poi_id2idx_dict, poi_features)
    val_dataset = TrajectoryDatasetVal(args, val_data, user_list, poi_id2idx_dict, poi_features)
    test_dataset = TrajectoryDatasetTest(args, test_data, user_list, poi_id2idx_dict, poi_features)

    # Creating train/val/test dataloader
    print('3. Creating train/val/test dataloader...')
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=False,
                              pin_memory=True, collate_fn=lambda x: x, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, drop_last=False,
                            pin_memory=True, collate_fn=lambda x: x, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, drop_last=False,
                             pin_memory=True, collate_fn=lambda x: x, num_workers=0)

    # %% ====================== Build models ======================
    print('4. Building models...')
    user_embedding_model = CommonEmbedding(len(user_id2idx_dict.keys()), args.user_embed_dim)
    region_embedding_model = CommonEmbedding(len(region2idx_dict.keys()), args.region_embed_dim)
    hotness_embedding_model = CommonEmbedding(len(hotness2idx_dict.keys()), args.hotness_embed_dim)
    checkin_embedding_model = CheckInEmbedding()
    predict_model = TransformerModel(len(poi_id2idx_dict.keys()), args.checkin_embed_dim,
                                     args.head, args.hid, args.layers, dropout=args.dropout)

    optimizer = optim.Adam(params=
                           list(predict_model.parameters()) + list(checkin_embedding_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    user_embedding_model = user_embedding_model.to(device=args.device)
    region_embedding_model = region_embedding_model.to(device=args.device)
    hotness_embedding_model = hotness_embedding_model.to(device=args.device)
    checkin_embedding_model = checkin_embedding_model.to(device=args.device)
    predict_model = predict_model.to(device=args.device)

    all_user_embeddings = user_embedding_model(args, range(len(user_id2idx_dict.keys())))
    all_user_embeddings = all_user_embeddings.reshape(len(user_id2idx_dict.keys()), args.user_embed_dim)

    all_region_embeddings = region_embedding_model(args, range(len(region2idx_dict.keys())))
    all_region_embeddings = all_region_embeddings.reshape(len(region2idx_dict.keys()), args.region_embed_dim)

    all_hotness_embeddings = hotness_embedding_model(args, range(len(hotness2idx_dict.keys())))
    all_hotness_embeddings = all_hotness_embeddings.reshape(len(hotness2idx_dict.keys()), args.hotness_embed_dim)

    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []

    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []

    max_val_score = -np.inf

    print("5. Training start...")
    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50} Epoch:{epoch:03d} {'*' * 50}\n")

        checkin_embedding_model.train()
        predict_model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []

        checkin_sequence_mask = predict_model.generate_square_subsequent_mask(args.batch).to(args.device)

        for b_idx, batch in enumerate(train_loader):
            if len(batch) != args.batch:
                checkin_sequence_mask = predict_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            batch_input_checkin_embedding_seqs = []
            batch_seq_lens = []
            batch_seq_labels_p_idx_seqs = []

            for trajectory_data in batch:
                label_poi_idx, checkin_embedding_seq = get_trajectory_embeddings(
                    args, trajectory_data, user_id2idx_dict, all_user_embeddings,
                    all_region_embeddings, all_hotness_embeddings, checkin_embedding_model
                )

                batch_input_checkin_embedding_seqs.append(checkin_embedding_seq)
                batch_seq_lens.append(checkin_embedding_seq.shape[0])

                batch_seq_labels_p_idx_seqs.append(torch.LongTensor(label_poi_idx))

            batch_checkin_embedding_padded = pad_sequence(batch_input_checkin_embedding_seqs, batch_first=True,
                                                          padding_value=-1)
            label_p_idx_padded = pad_sequence(batch_seq_labels_p_idx_seqs, batch_first=True, padding_value=-1)
            x_checkin_embedding = batch_checkin_embedding_padded.to(device=args.device, dtype=torch.float)
            y_p_idx = label_p_idx_padded.to(device=args.device, dtype=torch.long)
            y_pred_p = predict_model(x_checkin_embedding, checkin_sequence_mask)
            loss = criterion_poi(y_pred_p.transpose(1, 2), y_p_idx)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mrr = 0

            batch_label_p_idx = y_p_idx.detach().cpu().numpy()
            batch_pred_p_idx = y_pred_p.detach().cpu().numpy()

            for label_pois, pred_pois, seq_len in zip(batch_label_p_idx, batch_pred_p_idx, batch_seq_lens):
                label_pois = label_pois[:seq_len]
                pred_pois = pred_pois[:seq_len, :]

                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)

            train_batches_top1_acc_list.append(top1_acc / len(batch_label_p_idx))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_p_idx))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_p_idx))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_p_idx))
            train_batches_mrr_list.append(mrr / len(batch_label_p_idx))
            train_batches_loss_list.append(loss.detach().cpu().numpy())

            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                logging.info(f'Epoch：{epoch}，\tbatch：{b_idx}\n'
                             f'Loss: \n'
                             f'\t\tbatch training loss: {loss.item():.2f}，\tepoch average loss: {np.mean(train_batches_loss_list):.2f}\n'
                             f'Current batch performance: \n'
                             f'\t\tACC@1：{np.mean(train_batches_top1_acc_list):.4f}，\t'
                             f'ACC@5：{np.mean(train_batches_top5_acc_list):.4f}，\t'
                             f'ACC@10：{np.mean(train_batches_top10_acc_list):.4f}，\t'
                             f'ACC@20：{np.mean(train_batches_top20_acc_list):.4f}，\t'
                             f'MRR：{np.mean(train_batches_mrr_list):.4f}\n'
                             + '=' * 100)

        checkin_embedding_model.eval()
        predict_model.eval()

        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []

        val_checkin_sequence_mask = predict_model.generate_square_subsequent_mask(args.batch).to(args.device)

        for vb_idx, batch in enumerate(val_loader):
            if len(batch) != args.batch:
                val_checkin_sequence_mask = predict_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            val_batch_input_checkin_embedding_seqs = []
            val_batch_seq_lens = []
            val_batch_seq_labels_p_idx_seqs = []

            for trajectory_data in batch:
                val_label_poi_idx, val_checkin_embedding_seq = get_trajectory_embeddings(
                    args, trajectory_data, user_id2idx_dict, all_user_embeddings, all_region_embeddings,
                    all_hotness_embeddings, checkin_embedding_model
                )

                val_batch_input_checkin_embedding_seqs.append(val_checkin_embedding_seq)
                val_batch_seq_lens.append(val_checkin_embedding_seq.shape[0])

                val_batch_seq_labels_p_idx_seqs.append(torch.LongTensor(val_label_poi_idx))

            val_batch_checkin_embedding_padded = pad_sequence(val_batch_input_checkin_embedding_seqs, batch_first=True, padding_value=-1)
            val_label_p_idx_padded = pad_sequence(val_batch_seq_labels_p_idx_seqs, batch_first=True, padding_value=-1)
            val_x_checkin_embedding = val_batch_checkin_embedding_padded.to(device=args.device, dtype=torch.float)
            val_y_p_idx = val_label_p_idx_padded.to(device=args.device, dtype=torch.long)
            val_y_pred_p = predict_model(val_x_checkin_embedding, val_checkin_sequence_mask)
            val_loss = criterion_poi(val_y_pred_p.transpose(1, 2), val_y_p_idx)

            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mrr = 0

            val_batch_label_p_idx = val_y_p_idx.detach().cpu().numpy()
            val_batch_pred_p_idx = val_y_pred_p.detach().cpu().numpy()

            for label_pois, pred_pois, seq_len in zip(val_batch_label_p_idx, val_batch_pred_p_idx, val_batch_seq_lens):
                label_pois = label_pois[:seq_len]
                pred_pois = pred_pois[:seq_len, :]

                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)

            val_batches_top1_acc_list.append(top1_acc / len(val_batch_label_p_idx))
            val_batches_top5_acc_list.append(top5_acc / len(val_batch_label_p_idx))
            val_batches_top10_acc_list.append(top10_acc / len(val_batch_label_p_idx))
            val_batches_top20_acc_list.append(top20_acc / len(val_batch_label_p_idx))
            val_batches_mrr_list.append(mrr / len(val_batch_label_p_idx))
            val_batches_loss_list.append(val_loss.detach().cpu().numpy())

            if (vb_idx % (args.batch * 2)) == 0:
                sample_idx = 0
                logging.info(f'Current epoch：{epoch}\tcurrent batch：{vb_idx}\t'
                             f'val_batch_loss:{val_loss.item():.2f}\t'
                             f'val_batch_top1_acc:{top1_acc / len(val_batch_label_p_idx):.2f}\t'
                             f'val_move_loss:{np.mean(val_batches_loss_list):.2f}\n'
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f}\t'
                             f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f}\t'
                             f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f}\t'
                             f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                             f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             + '=' * 100)

        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)

        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)

        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mrr_list.append(epoch_train_mrr)

        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mrr_list.append(epoch_val_mrr)

        monitor_loss = epoch_val_loss
        monitor_score = np.mean(epoch_val_top1_acc + epoch_val_top5_acc + epoch_val_top10_acc + epoch_val_top20_acc +
                                epoch_val_mrr)

        lr_scheduler.step(monitor_loss)

        # 打印一个 epoch 的训练结果
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f}")

        if args.save_weights:
            state_dict = {
                'epoch': epoch,
                'predict_model': predict_model.state_dict(),
                'checkin_embedding_model': checkin_embedding_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'user_id2idx_dict': user_id2idx_dict,
                'poi_id2idx_dict': poi_id2idx_dict,
                'hotness2idx_dict': hotness2idx_dict,
                'region2idx_dict': region2idx_dict,
                'args': args,
                'epoch_train_metrics': {
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_train_top1_acc': epoch_train_top1_acc,
                    'epoch_train_top5_acc': epoch_train_top5_acc,
                    'epoch_train_top10_acc': epoch_train_top10_acc,
                    'epoch_train_top20_acc': epoch_train_top20_acc,
                    'epoch_train_mrr': epoch_train_mrr
                },
                'epoch_val_metrics': {
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_top1_acc': epoch_val_top1_acc,
                    'epoch_val_top5_acc': epoch_val_top5_acc,
                    'epoch_val_top10_acc': epoch_val_top10_acc,
                    'epoch_val_top20_acc': epoch_val_top20_acc,
                    'epoch_val_mrr': epoch_val_mrr
                }
            }
            model_save_dir = os.path.join(args.save_dir, 'checkpoints')

            if monitor_score >= max_val_score:
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                    print(state_dict['epoch_val_metrics'], file=f)
                max_val_score = monitor_score
                args.last_improve = epoch
                best = datetime.datetime.now()

        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}',
                  file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}',
                  file=f)
            print(
                f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                file=f)
            print(
                f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}',
                  file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}',
                  file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}',
                  file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}',
                  file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)

        model_save_dir = os.path.join(args.save_dir, 'checkpoints/best_epoch.state.pt')
        # 加载
        checkpoint = torch.load(model_save_dir)
        best_val_loss = checkpoint["epoch_val_metrics"]['epoch_val_loss']
        best_val_top_1 = checkpoint["epoch_val_metrics"]['epoch_val_top1_acc']
        best_val_top_5 = checkpoint["epoch_val_metrics"]['epoch_val_top5_acc']
        best_val_top_10 = checkpoint["epoch_val_metrics"]['epoch_val_top10_acc']
        best_val_top_20 = checkpoint["epoch_val_metrics"]['epoch_val_top20_acc']
        best_val_mrr = checkpoint["epoch_val_metrics"]['epoch_val_mrr']

        print(f"\nCurrent best performance: val_loss：{float(f'{best_val_loss:.4f}')}")
        print(f'\tAcc@1：{float(f"{best_val_top_1:.4f}")}, Acc@5：{float(f"{best_val_top_5:.4f}")}\n'
              f'\tAcc@10：{float(f"{best_val_top_10:.4f}")}, Acc@20：{float(f"{best_val_top_20:.4f}")}, '
              f'MRR：{float(f"{best_val_mrr:.4f}")}')
        print(f'\tlast_improve：{int(f"{args.last_improve}")}, total_epoch：{int(f"{epoch}")}\n')

        if epoch - checkpoint['epoch'] > args.early_stop:
            if epoch < args.performance_check:
                continue
            print(f'Over {int(f"{args.early_stop}")} epochs with no performance improvement, start validating...')
            predict_model.load_state_dict(checkpoint['predict_model'])
            checkin_embedding_model.load_state_dict(checkpoint['checkin_embedding_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            predict_model.eval()
            checkin_embedding_model.eval()

            test_batches_top1_acc_list = []
            test_batches_top5_acc_list = []
            test_batches_top10_acc_list = []
            test_batches_top20_acc_list = []
            test_batches_mrr_list = []
            test_batches_loss_list = []

            test_check_sequence_mask = predict_model.generate_square_subsequent_mask(args.batch).to(args.device)

            for test_idx, batch in enumerate(test_loader):
                if len(batch) != args.batch:
                    test_check_sequence_mask = predict_model.generate_square_subsequent_mask(len(batch)).to(
                        args.device)

                test_batch_input_checkin_embedding_seqs = []
                test_batch_seq_lens = []
                test_batch_seq_labels_p_idx_seqs = []

                for trajectory_data in batch:
                    test_label_poi_idx, test_checkin_embedding_seq = get_trajectory_embeddings(
                        args, trajectory_data, user_id2idx_dict, all_user_embeddings, all_region_embeddings,
                        all_hotness_embeddings, checkin_embedding_model
                    )

                    test_batch_input_checkin_embedding_seqs.append(test_checkin_embedding_seq)
                    test_batch_seq_lens.append(test_checkin_embedding_seq.shape[0])
                    test_batch_seq_labels_p_idx_seqs.append(torch.LongTensor(test_label_poi_idx))

                test_batch_checkin_embedding_padded = pad_sequence(test_batch_input_checkin_embedding_seqs,
                                                                   batch_first=True, padding_value=-1)
                test_label_p_idx_padded = pad_sequence(test_batch_seq_labels_p_idx_seqs, batch_first=True,
                                                       padding_value=-1)
                test_x_checkin_embedding = test_batch_checkin_embedding_padded.to(device=args.device, dtype=torch.float)
                test_y_p_idx = test_label_p_idx_padded.to(device=args.device, dtype=torch.long)
                test_y_pred_p = predict_model(test_x_checkin_embedding, test_check_sequence_mask)
                test_loss = criterion_poi(test_y_pred_p.transpose(1, 2), test_y_p_idx)

                top1_acc = 0
                top5_acc = 0
                top10_acc = 0
                top20_acc = 0
                mrr = 0

                test_batch_label_p_idx = test_y_p_idx.detach().cpu().numpy()
                test_batch_pred_p_idx = test_y_pred_p.detach().cpu().numpy()

                for label_pois, pred_pois, seq_len in zip(test_batch_label_p_idx, test_batch_pred_p_idx,
                                                          test_batch_seq_lens):
                    label_pois = label_pois[:seq_len]
                    pred_pois = pred_pois[:seq_len, :]

                    top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                    top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                    top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                    top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                    mrr += MRR_metric_last_timestep(label_pois, pred_pois)

                test_batches_top1_acc_list.append(top1_acc / len(test_batch_label_p_idx))
                test_batches_top5_acc_list.append(top5_acc / len(test_batch_label_p_idx))
                test_batches_top10_acc_list.append(top10_acc / len(test_batch_label_p_idx))
                test_batches_top20_acc_list.append(top20_acc / len(test_batch_label_p_idx))
                test_batches_mrr_list.append(mrr / len(test_batch_label_p_idx))
                test_batches_loss_list.append(test_loss.detach().cpu().numpy())

                if (test_idx % (args.batch * 2)) == 0:
                    logging.info(f'Current batch {test_idx} result of test dataset: \t'
                                 f'Loss: {test_loss.item():.2f}\n'
                                 f'Batch ACC@1：{np.mean(test_batches_top1_acc_list):.4f}\n'
                                 f'Batch ACC@5：{np.mean(test_batches_top5_acc_list):.4f}\n'
                                 f'Batch ACC@10：{np.mean(test_batches_top10_acc_list):.4f}\n'
                                 f'Batch ACC@20：{np.mean(test_batches_top20_acc_list):.4f} \n'
                                 f'Batch MRR：{np.mean(test_batches_mrr_list):.4f} \n' + '=' * 100)

            logging.info(f"Result of test dataset: epoch = {checkpoint['epoch']}\n"
                         f"test_loss: {np.mean(test_batches_loss_list):.4f}\n"
                         f"test_top1_acc：{np.mean(test_batches_top1_acc_list):.4f}, "
                         f"test_top5_acc：{np.mean(test_batches_top5_acc_list):.4f}, \n"
                         f"test_top10_acc：{np.mean(test_batches_top10_acc_list):.4f}, "
                         f"test_top20_acc：{np.mean(test_batches_top20_acc_list):.4f}, \n"
                         f"test_mrr：{np.mean(test_batches_mrr_list):.4f}")
            with open(os.path.join(args.save_dir, f'{args.name}-test-result.txt'), "w") as f:
                print(f"Result of test dataset: epoch = {checkpoint['epoch']}\n"
                      f"test_loss: {np.mean(test_batches_loss_list):.4f}\n"
                      f"test_top1_acc：{np.mean(test_batches_top1_acc_list):.4f}, "
                      f"test_top5_acc：{np.mean(test_batches_top5_acc_list):.4f}, \n"
                      f"test_top10_acc：{np.mean(test_batches_top10_acc_list):.4f}, "
                      f"test_top20_acc：{np.mean(test_batches_top20_acc_list):.4f}, \n"
                      f"test_mrr：{np.mean(test_batches_mrr_list):.4f}", file=f)
            end = datetime.datetime.now()
            print("=" * 100)
            print(f"Time cost to obtain the best performance: {(best - start).seconds} 秒")
            print(f"Total time cost: {(end - start).seconds} 秒")
            break

if __name__ == '__main__':
    args = parameter_parser()

    _init_fn(args.seed)

    args.name = "FPGT-NYC-G"

    args.checkin_embed_dim = args.user_embed_dim + args.hotness_embed_dim + args.region_embed_dim
    print(f"checkin_embed_dim: {args.checkin_embed_dim}, run_name:{args.name}")
    train(args)
