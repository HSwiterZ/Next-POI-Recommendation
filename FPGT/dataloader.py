import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


def get_map_dict(path):
    data = pd.read_csv(path)
    return data.set_index('item')['idx'].to_dict()


def get_poi_features(path):
    poi_features = pd.read_csv(path)
    poi_features.set_index('poi_idx', inplace=True)
    poi_features_list = []
    for poi_idx in poi_features.index:
        poi_feature = poi_features.loc[poi_idx]
        # features = [poi_idx, poi_feature['cat_idx'], poi_feature['region_idx'], poi_feature['hotness_idx']]
        features = [poi_idx, poi_feature['region_idx'], poi_feature['hotness_idx']]
        poi_features_list.append(features)
    return poi_features_list


class TrajectoryDatasetTrain(Dataset):
    def __init__(self, args, train_data, poi_id2idx_dict, poi_features):
        self.df = train_data

        train_data_grouped = train_data.groupby('trajectory_id')

        self.trajectory_seqs = []
        self.input_seqs = []
        self.label_seqs = []

        for trajectory_id, trajectory_data in tqdm(train_data_grouped):
            poi_idx_list = [poi_id2idx_dict[poi] for poi in list(trajectory_data['POI_id'])]
            region_idx_list = [poi_features[poi_idx][1] for poi_idx in poi_idx_list]
            hotness_idx_list = [poi_features[poi_idx][2] for poi_idx in poi_idx_list]

            input_seq = list(zip(region_idx_list[:-1], hotness_idx_list[:-1]))
            label_seq = poi_idx_list[1:]

            if len(poi_idx_list) >= 2:
                self.trajectory_seqs.append(trajectory_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

    def __len__(self):
        return len(self.trajectory_seqs)

    def __getitem__(self, index):
        return self.trajectory_seqs[index], self.input_seqs[index], self.label_seqs[index]


class TrajectoryDatasetVal(Dataset):
    def __init__(self, args, df, user_list, poi_id2idx_dict, poi_features):
        self.df = df
        self.trajectory_seqs = []
        self.input_seqs = []
        self.label_seqs = []
        val_data_grouped = df.groupby('trajectory_id')

        for trajectory_id, trajectory_data in tqdm(val_data_grouped):
            user_id = trajectory_id.split('_')[0]
            if int(user_id) not in user_list:
                continue

            poi_idx_list = []
            region_idx_list = []
            hotness_idx_list = []
            for poi in list(trajectory_data['POI_id']):
                if poi not in poi_id2idx_dict.keys():
                    continue
                else:
                    poi_idx = poi_id2idx_dict[poi]
                    poi_idx_list.append(poi_idx)

                    region_idx_list.append(poi_features[poi_idx][1])
                    hotness_idx_list.append(poi_features[poi_id2idx_dict[poi]][2])

            input_seq = list(zip(region_idx_list[:-1], hotness_idx_list[:-1]))
            label_seq = poi_idx_list[1:]

            if len(input_seq) >= 2:
                self.trajectory_seqs.append(trajectory_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

    def __len__(self):
        return len(self.trajectory_seqs)

    def __getitem__(self, index):
        return self.trajectory_seqs[index], self.input_seqs[index], self.label_seqs[index]


class TrajectoryDatasetTest(Dataset):
    def __init__(self, args, df, user_list, poi_id2idx_dict, poi_features):
        self.df = df
        self.trajectory_seqs = []
        self.input_seqs = []
        self.label_seqs = []
        test_data_grouped = df.groupby('trajectory_id')

        for trajectory_id, trajectory_data in tqdm(test_data_grouped):
            user_id = trajectory_id.split('_')[0]

            if int(user_id) not in user_list:
                continue

            poi_idx_list = []
            region_idx_list = []
            hotness_idx_list = []
            for poi in list(trajectory_data['POI_id']):
                if poi not in poi_id2idx_dict.keys():
                    continue
                else:
                    poi_idx = poi_id2idx_dict[poi]
                    poi_idx_list.append(poi_idx)

                    region_idx_list.append(poi_features[poi_idx][1])
                    hotness_idx_list.append(poi_features[poi_id2idx_dict[poi]][2])

            input_seq = list(zip(region_idx_list[:-1], hotness_idx_list[:-1]))
            label_seq = poi_idx_list[1:]

            if len(input_seq) >= 2:
                self.trajectory_seqs.append(trajectory_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

    def __len__(self):
        return len(self.trajectory_seqs)

    def __getitem__(self, index):
        return self.trajectory_seqs[index], self.input_seqs[index], self.label_seqs[index]