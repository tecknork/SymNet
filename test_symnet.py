from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from utils.base_solver import BaseSolver

import os, logging, importlib, re, copy, random, tqdm, argparse
import os.path as osp
#import cPickle as pickle
from pprint import pprint
from datetime import datetime
import numpy as np
from collections import defaultdict


import tensorflow as tf
import tensorflow.contrib.slim as slim
import torch

from utils import config as cfg
from utils import dataset, utils
from utils.evaluator import CZSL_Evaluator


from run_symnet import make_parser


def main():
    logger = logging.getLogger('MAIN')

    parser = make_parser()
    args = parser.parse_args()
    utils.display_args(args, logger)


    logger.info("Loading dataset")
    test_dataloader = dataset.get_dataloader(args.data, 'test',
        batchsize=args.test_bz, obj_pred=args.obj_pred)
    


    logger.info("Loading network and solver")
    network = importlib.import_module('network.'+args.network)
    net = network.Network(test_dataloader, args)
    

    with utils.create_session() as sess:
        sw = SolverWrapper(net, test_dataloader, args)
        sw.trainval_model(sess, args.epoch)




################################################################################



class SolverWrapper(BaseSolver):
    def __init__(self, network, test_dataloader, args):
        logger = self.logger("init")
        self.network = network

        self.test_dataloader = test_dataloader
        self.weight_dir = osp.join(cfg.WEIGHT_ROOT_DIR, args.name)
        self.args = args


        self.trained_weight = os.path.join(cfg.WEIGHT_ROOT_DIR, args.name, "snapshot_epoch_%d.ckpt"%args.epoch)
        self.logger("init").info("pretrained model <= "+self.trained_weight)
            

        

    def construct_graph(self, sess):
        logger = self.logger('construct_graph')

        with sess.graph.as_default():
            if cfg.RANDOM_SEED is not None:
                tf.set_random_seed(cfg.RANDOM_SEED)

            loss_op, score_op, train_summary_op,image_embeddings = self.network.build_network()
            
            global_step = tf.Variable(self.args.epoch, trainable=False)

        return score_op, train_summary_op,image_embeddings

        

    def trainval_model(self, sess, max_epoch):
        logger = self.logger('train_model')
        logger.info('Begin training')

        score_op, train_summary_op,img_embeddings = self.construct_graph(sess)
        #for x in tf.global_variables():
        #    print(x.name)

        self.initialize(sess)
        sess.graph.finalize()

        evaluator = CZSL_Evaluator(self.test_dataloader.dataset, self.network)


        ############################## test czsl ################################

        accuracies_pair = defaultdict(list)
        accuracies_attr = defaultdict(list)
        accuracies_obj = defaultdict(list)
        image_reterival_top_nn = []
        test_query_embeddings_list= []
        image_reterival_image_labels_nn = []
        image_embeddings = self.network.get_img_embeddings(sess,img_embeddings)
        #print(image_embeddings.shape)
        #print(image_embeddings)

        for image_ind, batch in tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), postfix='test'):

            predictions = self.network.test_step(sess, batch, score_op)  # ordereddict of [score_pair, score_a, score_o]

            attr_truth, obj_truth = batch[1], batch[2]
            attr_truth, obj_truth = torch.from_numpy(attr_truth), torch.from_numpy(obj_truth)

            def get_ground_label_for_image_ids(image_ids):
                lables_for_batch = []
                for image_id in image_ids:
                    image_data = self.test_dataloader.dataset.test_data[image_id]
                    lables_for_batch.append((image_data[3], image_data[4]))
                return lables_for_batch
            match_stats = []
            for key in score_op.keys():
                if key != 'nearest_neighbour':
                    p_pair, p_a, p_o = predictions[key]
                    pair_results = evaluator.score_model(p_pair, obj_truth)
                    match_stats = evaluator.evaluate_predictions(pair_results, attr_truth, obj_truth)
                    accuracies_pair[key].append(match_stats)  # 0/1 sequence of t/f

                    a_match, o_match = evaluator.evaluate_only_attr_obj(p_a, attr_truth, p_o, obj_truth)

                    accuracies_attr[key].append(a_match)
                    accuracies_obj[key].append(o_match)

                else:
                    top_nn = predictions[key]
                    test_query_embeddings_list.extend(top_nn)
                   # print(top_nn)
                    #print(len(top_nn))
                    #print(top_nn.shape)
                    # image_reterival_top_nn.extend([image_id for row in top_nn for image_id in row])
                    # image_reterival_image_labels_nn.extend(
                    #     [get_ground_label_for_image_ids(image_id) for row in top_nn for image_id in row])

        #print(test_query_embeddings_list)
        test_query_embeddings_list= np.array(test_query_embeddings_list)

        #print(test_query_embeddings_list.shape)

        for i in range(image_embeddings.shape[0]):
            image_embeddings[i, :] /= np.linalg.norm(image_embeddings[i, :])
        for i in range(test_query_embeddings_list.shape[0]):
            test_query_embeddings_list[i, :] /= np.linalg.norm(test_query_embeddings_list[i, :])

        sims = test_query_embeddings_list.dot(image_embeddings.T)
        nn_result_labels = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]
        nn_result_labels = [get_ground_label_for_image_ids(data) for data in nn_result_labels]
        ############ image_reterival_score ########################
        target_labels_for_each_query = [(data[6], data[7]) for data in self.test_dataloader.dataset.data]
        # recall_k = defaultdict(list)
        # for k in [1, 5, 10, 50, 100]:
        #             r = 0.0
        #             r_a = 0.0
        #             r_o = 0.0
        #             for query_result_image_ids, query_result_image_labels, query_target_labels in zip(
        #                     image_reterival_top_nn, image_reterival_image_labels_nn, target_labels_for_each_query):
        #                 if query_target_labels in query_result_image_labels[:k]:
        #                     r += 1
        #                 if query_target_labels[0] in [x[0] for x in query_result_image_labels[:k]]:
        #                     r_a += 1
        #                 if query_target_labels[1] in [x[1] for x in query_result_image_labels[:k]]:
        #                     r_o += 1
        #             r /= len(target_labels_for_each_query)
        #             r_a /= len(target_labels_for_each_query)
        #             r_o /= len(target_labels_for_each_query)
        #             recall_k[k].append([r, r_a, r_o])
        #             print("k:%d recall_compositon:%s recall_attribue:%s recall_object:%s" % (k, r, r_a, r_o))
        # ###################### image reterival #################################################
        out = []
        for k in [1, 5, 10,50,100]:
            r = 0.0
            for target_query, nns in zip(target_labels_for_each_query, nn_result_labels):
                if target_query in nns[:k]:
                    r += 1
            r /= len(nn_result_labels)
            out += [('recall_top' + str(k) + '_correct_composition', r)]

            r = 0.0
            for target_query, nns in zip(target_labels_for_each_query, nn_result_labels):
                if target_query[0] in [x[0] for x in nns[:k]]:
                    r += 1
            r /= len(nn_result_labels)
            out += [('recall_top' + str(k) + '_correct_adj', r)]


            r = 0.0
            for target_query, nns in zip(target_labels_for_each_query, nn_result_labels):
                if target_query[1] in [x[1] for x in nns[:k]]:
                    r += 1
            r /= len(nn_result_labels)
            out += [('recall_top' + str(k) + '_correct_noun', r)]

        print(out)


        for name in accuracies_pair.keys():
            accuracies = accuracies_pair[name]
            accuracies = zip(*accuracies)
            accuracies = map(torch.mean, map(torch.cat, accuracies))
            attr_acc, obj_acc, closed_1_acc, closed_2_acc, closed_3_acc, _, objoracle_acc = map(lambda x:x.item(), accuracies)

            real_attr_acc = torch.mean(torch.cat(accuracies_attr[name])).item()
            real_obj_acc = torch.mean(torch.cat(accuracies_obj[name])).item()

            report_dict = {
                'real_attr_acc':real_attr_acc,
                'real_obj_acc': real_obj_acc,
                'top1_acc':     closed_1_acc,
                'top2_acc':     closed_2_acc,
                'top3_acc':     closed_3_acc,
                'name':         self.args.name,
                'epoch':        self.args.epoch,
                #'ir_recall':    recall_k,
            }

            print(name + ": " + utils.formated_czsl_result(report_dict))
                    

        logger.info('Finished.')






if __name__=="__main__":
    main()
