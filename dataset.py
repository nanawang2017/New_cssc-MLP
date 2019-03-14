"""
Created on Nov 17, 2018
从user_cluster.pkl得到进行谱聚类后的用户，本py文件主要完成的就是对一类user构造得到训练集和测试集
在主函数中修改dataset_name，
当选择dataset_name = 'Yelp'，记得修改self.users_sim = user_cluster[0]
dataset_name =Gowalla，记得修改self.users_sim = user_cluster[1]（在109行）

"""
import os
import numpy as np
import pickle


class Dataset(object):

    def __init__(self, dataset_name, negative=10, split=0.7):
        """
        train_data:是用户签到训练集 全为用户真实签到的数据构成 用于训练模型
        test_data:模型输入进行评估的数据集
        user_test_poi：是进行评估时使用的用户测试集 用于求命中率
        :param dataset_name:
        :param prefix:
        :param negative:
        :param split:
        """
        self.users_sim = []
        # self.test_data = {}
        self.test_set = []
        self.test_negatives = []
        self.train_data = {}

        self.user_enum, self.spot_enum = 0, 0

        self.negative = negative
        self.split = split

        self.project_path = os.path.dirname(os.getcwd())
        self.checkins_dict_path = self.project_path + '/data/' + dataset_name + '/inter/user_poi.pkl'
        self.social_dict_path = self.project_path + '/data/' + dataset_name + '/inter/user_friend.pkl'
        self.user_cluster_path = self.project_path + '/data/' + dataset_name + '/inter/user_cluster.pkl'
        self.train_path = self.project_path + '/data/' + dataset_name + '/inter/train.pkl'
        # self.test_cluster_path = self.project_path + '/data/' + dataset_name + '/inter/' + 'test' \
        #                          + self.prefix + '.pkl'
        self.inter_path = self.project_path + '/data/' + dataset_name + '/inter/inter.pkl'
        self.test_set_path = self.project_path + '/data/' + dataset_name + '/inter/test_set.pkl'
        self.test_negatives_path = self.project_path + '/data/' + dataset_name + '/inter/test_negatives.pkl'

        self.generate()

    def generate(self):
        writeToFile = False
        try:
            f = open(self.inter_path, 'rb')
            f.close()
        except IOError:
            writeToFile = True
        if not writeToFile:
            with open(self.inter_path, 'rb') as f:
                inter_data = pickle.load(f)
            with open(self.train_path, 'rb') as f:
                self.train_data = pickle.load(f)
            with open(self.test_set_path, 'rb') as f:
                self.test_set = pickle.load(f)
            with open(self.test_negatives_path, 'rb') as f:
                self.test_negatives = pickle.load(f)
            self.user_enum = inter_data['user_enum']
            self.spot_enum = inter_data['spot_enum']
            print(str(len(self.user_enum)) + ' users in enum loaded')
            print(str(len(self.spot_enum)) + ' spots in enum loaded')
            print(str(len(self.train_data['user'])) + ' training labels loaded')
            print(str(len(self.test_set)) + ' test set loaded')
            print(str(len(self.test_negatives)) + ' test  negatives loaded')
        else:
            inter_data = {}
            self.train_data['user'] = []
            self.train_data['spot'] = []
            self.train_data['label'] = []
            # self.test_data['user'] = []
            # self.test_data['spot'] = []
            # self.test_data['label'] = []

            self.user_enum, self.spot_enum = self.getCrossLabels()
            inter_data['user_enum'] = self.user_enum
            inter_data['spot_enum'] = self.spot_enum
            with open(self.inter_path, 'wb') as f:
                pickle.dump(inter_data, f)

            print('Writing ' + str(len(self.train_data['user'])) + ' training data to file')
            with open(self.train_path, 'wb') as f:
                pickle.dump(self.train_data, f)

            print('Writing ' + str(len(self.test_set)) + ' testing data to file')
            with open(self.test_set_path, 'wb') as f:
                pickle.dump(self.test_set, f)

            print('Writing ' + str(len(self.test_negatives)) + ' testing  negatives to file')
            with open(self.test_negatives_path, 'wb') as f:
                pickle.dump(self.test_negatives, f)

    def getCrossLabels(self):

        user_dict = {}  # user_id:[place_id_1, place_id_2, ...]
        spot_dict = {}  # place_id:[user_id_1,...]
        split_portion = self.split  # 训练及测试集通过随机抽取比例进行分割

        with open(self.user_cluster_path, 'rb')as f:
            user_cluster = pickle.load(f)
        # 当数据集为Gowalla是 选择user_cluster[1] 即第二组，当数据集为Yelp时，选择user_cluster[0] 即第一组
        self.users_sim = user_cluster[1]

        with open(self.checkins_dict_path, 'rb') as f:
            user_dict = pickle.load(f)
            for user in list(user_dict.keys()):
                poi_array = user_dict[user]
                for poi in poi_array:
                    if poi not in spot_dict.keys():
                        spot_dict[poi] = []
                    spot_dict[poi].append(user)

            print('Generating labels')
            user_enum = {}
            spot_enum = {}

            # 用于给user_id重新编号
            u_counter = 0
            s_counter = 0

            all_pois = set(spot_dict.keys())
            for user in self.users_sim:

                train_pois = np.random.choice(list(user_dict[user]), size=int(len(user_dict[user]) * split_portion),
                                              replace=False)
                test_pois = list(set(user_dict[user]) - set(train_pois))
                all_neg_pois = list(all_pois - set(user_dict[user]))
                # neg_pois = np.random.choice(all_neg_pois, size=self.negative, replace=False)
                # neg_pois = np.random.choice(all_neg_pois, size=self.negative * len(test_pois), replace=False)

                user_enum[user] = u_counter

                u_counter += 1
                # 构造训练集
                for poi in train_pois:
                    if poi in spot_dict.keys():
                        if poi not in spot_enum:
                            spot_enum[poi] = s_counter
                            s_counter += 1
                        self.train_data['user'].append(user_enum[user])
                        self.train_data['spot'].append(spot_enum[poi])
                        self.train_data['label'].append(1)
                # 为每一个用户 构造测试集 和 负样本
                user_test_pois = []
                neg = []
                for poi in test_pois:
                    if poi in spot_dict.keys():
                        if poi not in spot_enum:
                            spot_enum[poi] = s_counter
                            s_counter += 1
                        user_test_pois.append(spot_enum[poi])
                    neg_pois = np.random.choice(all_neg_pois, size=self.negative, replace=False)
                    for k in neg_pois:
                        if k in spot_dict.keys():
                            if k not in spot_enum:
                                spot_enum[k] = s_counter
                                s_counter += 1
                            neg.append(spot_enum[k])
                if len(user_test_pois)==0 or len(neg)==0:
                    continue
                else:

                    self.test_set.append([user_enum[user], user_test_pois])
                    self.test_negatives.append(neg)

                # for poi in test_pois:
                #     if poi in spot_dict.keys():
                #         if poi not in spot_enum:
                #             spot_enum[poi] = s_counter
                #             s_counter += 1
                #         self.test_data['user'].append(user_enum[user])
                #         self.test_data['spot'].append(spot_enum[poi])
                #         self.test_data['label'].append(1)
                #
                #         for k in neg_pois:
                #             if k in spot_dict.keys():
                #                 if k not in spot_enum:
                #                     spot_enum[k] = s_counter
                #                     s_counter += 1
                #                 self.test_data['user'].append(user_enum[user])
                #                 self.test_data['spot'].append(spot_enum[k])
                #                 self.test_data['label'].append(0)

        return user_enum, spot_enum


if __name__ == "__main__":
    dataset_name = 'Gowalla'
    Dataset(dataset_name)
