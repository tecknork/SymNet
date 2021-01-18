import tensorflow as tf
import time, cv2, os, argparse
import os.path as osp
from PIL import Image
import numpy as np

from . import config as cfg
from . import aux_data

################################################################################
#                               tools for solvers                              #
################################################################################

def create_session():
    """create tensorflow session"""
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "lin-razam:8890")
    return sess


def display_args(args, logger, verbose=False):
    """print some essential arguments"""
    if verbose:
        ignore = []
        for k,v in args.__dict__.items():
            if not callable(v) and not k.startswith('__') and k not in ignore:
                logger.info("{:30s}{}".format(k,v))
    else:
        logger.info('Name:       %s'%args.name)
        logger.info('Network:    %s'%args.network)
        logger.info('Data:       %s'%args.data)
        logger.info('FC layers:  At {fc_att}, Cm {fc_compress}, Cls {fc_cls}'.format(
            **args.__dict__))



def duplication_check(args):
    if args.force:
        return
    elif args.trained_weight is None or args.trained_weight.split('/')[0] != args.name:
        assert not osp.exists(osp.join(cfg.WEIGHT_ROOT_DIR, args.name)), \
            "weight dir with same name exists (%s)"%(args.name)
        assert not osp.exists(osp.join(cfg.LOG_ROOT_DIR, args.name)), \
            "log dir with same name exists (%s)"%(args.name)
        

def formated_czsl_result(report):
    fstr = '[{name}/{epoch}] rA:{real_attr_acc:.4f}|rO:{real_obj_acc:.4f}|Cl/T1:{top1_acc:.4f}|T2:{top2_acc:.4f}|T3:{top3_acc:.4f}'

    if 'ir_recall' in report:
        fstr_recall = '|IR_R' \
                      '/1R:{r1}|1RA:{r1a}|1RO:{r1o}' \
                      '/5R:{r5}|5RA:{r5a}|5RO:{r5o}' \
                      '/10R:{r10}|10RA:{r10a}|10RO:{r10o}' \
                      '/50R:{r50}|50RA:{r50a}|50RO:{r50o}' \
                      '/100R:{r100}|100RA:{r100a}|100RO:{r100o}'.format(
                      r1=report['ir_recall'][1][0][0],
                      r1a=report['ir_recall'][1][0][1],
                      r1o=report['ir_recall'][1][0][2],
                      r5=report['ir_recall'][5][0][0],
                      r5a=report['ir_recall'][5][0][1],
                      r5o=report['ir_recall'][5][0][2],
                      r10=report['ir_recall'][10][0][0],
                      r10a=report['ir_recall'][10][0][1],
                      r10o=report['ir_recall'][10][0][2],
                      r50=report['ir_recall'][50][0][0],
                      r50a=report['ir_recall'][50][0][1],
                      r50o=report['ir_recall'][50][0][2],
                      r100=report['ir_recall'][100][0][0],
                      r100a=report['ir_recall'][100][0][1],
                      r100o=report['ir_recall'][100][0][2],
        )
        return fstr.format(**report) + fstr_recall

    else:
        return fstr.format(**report)

################################################################################
#                                glove embedder                                #
################################################################################

class Embedder:
    """word embedder (for various vector type)
    __init__(self)
    """

    def __init__(self, vec_type, vocab, data):
        self.vec_type = vec_type

        if vec_type != 'onehot':
            self.embeds = self.load_word_embeddings(vec_type, vocab, data)
            self.emb_dim = self.embeds.shape[1]
        else:
            self.emb_dim = len(vocab)
    
    def get_embedding(self, i):
        """actually implements __getitem__() function"""
        if self.vec_type == 'onehot':
            return tf.one_hot(i, depth=self.emb_dim, axis=1)
        else:
            i_onehot = tf.one_hot(i, depth=self.embeds.shape[0], axis=1)
            return tf.matmul(i_onehot, self.embeds)


    def load_word_embeddings(self, vec_type, vocab, data):
        tmp = aux_data.load_wordvec_dict(data, vec_type)
        if type(tmp) == tuple:
            attr_dict, obj_dict = tmp
            attr_dict.update(obj_dict)
            embeds = attr_dict
        else:
            embeds = tmp

        embeds_list = []
        for k in vocab:
            if k in embeds:
                embeds_list.append(embeds[k])
            else:
                raise NotImplementedError('some vocabs are not in dictionary: %s'%k)

        embeds = np.array(embeds_list, dtype=np.float32)

        print ('Embeddings shape = %s'%str(embeds.shape))
        return embeds





################################################################################
#                                network utils                                 #
################################################################################


def repeat_tensor(tensor, axis, multiple):
    """e.g. (1,2,3)x3 = (1,1,1,2,2,2,3,3,3)"""
    
    result_shape = tensor.shape.as_list()
    for i,v in enumerate(result_shape):
        if v is None:
            result_shape[i] = tf.shape(tensor)[i]
    result_shape[axis] *= multiple

    tensor = tf.expand_dims(tensor, axis+1)
    mul = [1]*len(tensor.shape)
    mul[axis+1] = multiple
    tensor = tf.tile(tensor, mul)
    tensor = tf.reshape(tensor, result_shape)

    return tensor


def tile_tensor(tensor, axis, multiple):
    """e.g. (1,2,3)x3 = (1,2,3,1,2,3,1,2,3)"""
    mul = [1]*len(tensor.shape)
    mul[axis] = multiple

    return tf.tile(tensor, mul)


def activation_func(name):
    if name == "none":
        return (lambda x:x)
    elif name == "sigmoid":
        return tf.sigmoid
    elif name == "relu":
        return tf.nn.relu
    else:
        raise NotImplementedError("activation function %s not implemented"%name)

