import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

def save_map_dict(args, dict_name_seq=None):
    if dict_name_seq is None:
        dict_name_seq = ["user", "poi", "time", "hot", "reg", "cat"]
    data = pd.read_csv(args.data_train)
    # obtain POI id list
    poi_id_list = sorted(list(set(data["POI_id"].tolist())))

    if "user" in dict_name_seq:
        # obtain user id list
        user_id_list = list(set(data["user_id"].tolist()))
        # transfer user id to index
        user_id2idx_dict = dict(zip(user_id_list, range(len(user_id_list))))

        with open(os.path.join(args.map_path, 'user_id2idx_dict.csv'), 'w') as f:
            print(f'item,idx', file=f)
            for user_id in tqdm(user_id2idx_dict.keys()):
                print(f'{user_id},{user_id2idx_dict[user_id]}', file=f)

    if "poi" in dict_name_seq:
        # transfer POI id to index
        poi_id2idx_dict = dict(zip(poi_id_list, range(len(poi_id_list))))

        with open(os.path.join(args.map_path, 'poi_id2idx_dict.csv'), 'w') as f:
            print(f'item,idx', file=f)
            for poi_id in tqdm(poi_id2idx_dict.keys()):
                print(f'{poi_id},{poi_id2idx_dict[poi_id]}', file=f)

    if "hot" in dict_name_seq:
        hotness_list = []
        result = data['POI_id'].value_counts()
        for poi in tqdm(poi_id_list):
            hotness = math.ceil(result.get(poi, 0) / args.hot_split)
            hotness_list.append(hotness)
        hotness_list = sorted(list(set(hotness_list)))
        hotness2idx_dict = dict(zip(hotness_list, range(len(hotness_list))))

        with open(os.path.join(args.map_path, 'hotness2idx_dict.csv'), 'w') as f:
            print(f'item,idx', file=f)
            for hotness in tqdm(hotness2idx_dict.keys()):
                print(f'{hotness},{hotness2idx_dict[hotness]}', file=f)

    if "reg" in dict_name_seq:
        region_list = []
        for longitude, latitude in tqdm(zip(list(data['longitude']), list(data['latitude']))):
            new_longitude = int(longitude * args.reg_split) / args.reg_split
            new_latitude = int(latitude * args.reg_split) / args.reg_split
            key = f'{new_longitude}_{new_latitude}'
            if key not in region_list:
                region_list.append(key)
        region_list = sorted(list(set(region_list)))
        region2idx_dict = dict(zip(region_list, range(len(region_list))))

        with open(os.path.join(args.map_path, 'region2idx_dict.csv'), 'w') as f:
            print(f'item,idx', file=f)
            for loc in tqdm(region2idx_dict.keys()):
                print(f'{loc},{region2idx_dict[loc]}', file=f)


def save_poi_features(args):
    print("Loading maps...")
    poi_id2idx_dict = get_map_dict(args.poi_map)
    cat_id2idx_dict = get_map_dict(args.cat_map)
    region2idx_dict = get_map_dict(args.reg_map)
    hotness2idx_dict = get_map_dict(args.hot_map)

    print("Loading data of training set...")
    data = pd.read_csv(args.data_train)
    data_copy = data.copy()
    print("Dropping duplicates of training set...")
    data_copy.drop_duplicates(subset='POI_id', keep='first', inplace=True)

    print("Sorting training set...")
    data_copy.sort_values(by=["POI_id"], inplace=True)
    print(len(data_copy), len(data))

    result = data['POI_id'].value_counts()

    with open(os.path.join(args.map_path, 'poi_features2idx.csv'), 'w') as f:
        print(f'poi_idx,region_idx,hotness_idx', file=f)
        for i, row in tqdm(data_copy.iterrows()):

            poi_idx = poi_id2idx_dict[row["POI_id"]]

            new_longitude = int(row["longitude"] * args.reg_split) / args.reg_split
            new_latitude = int(row["latitude"] * args.reg_split) / args.reg_split
            key = f'{new_longitude}_{new_latitude}'
            region_idx = region2idx_dict[key]

            hotness = math.ceil(result.get(row["POI_id"], 0) / args.hot_split)
            hotness_idx = hotness2idx_dict[hotness]
            print(f'{poi_idx},{region_idx},{hotness_idx}', file=f)


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
