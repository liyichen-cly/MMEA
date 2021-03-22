import math
import multiprocessing as mp
import numpy as np
import random
import gc
from sklearn import preprocessing
from datetime import datetime

import openea.modules.train.batch as bat
from openea.modules.base.initializers import xavier_init

from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import *
from openea.modules.base.initializers import init_embeddings
from openea.modules.load.read import generate_sup_attribute_triples
from openea.modules.base.losses import *
from openea.modules.finding.evaluation import early_stop

from openea.models.basic_model import BasicModel
from openea.approaches.literal_encoder import LiteralEncoder
import openea.modules.finding.evaluation as eva


def valid_temp(model, embed_choice='avg', w=(1, 1)):
    if embed_choice == 'rv':
        ent_embeds = model.rv_ent_embeds.eval(session=model.session)
    elif embed_choice == 'iv':
        ent_embeds = model.iv_ent_embeds.eval(session=model.session)
    elif embed_choice == 'av':
        ent_embeds = model.av_ent_embeds.eval(session=model.session)
    elif embed_choice == 'final':
        ent_embeds = model.ent_embeds.eval(session=model.session)
    elif embed_choice == 'avg':
        ent_embeds = w[0] * model.rv_ent_embeds.eval(session=model.session) + \
                     w[1] * model.iv_ent_embeds.eval(session=model.session)
    else:  # 'final'
        ent_embeds = model.ent_embeds
    print(embed_choice, 'valid results:')
    embeds1 = ent_embeds[model.kgs.valid_entities1,]
    embeds2 = ent_embeds[model.kgs.valid_entities2 + model.kgs.test_entities2,]
    hits1_12, mrr_12 = eva.valid(embeds1, embeds2, None, model.args.top_k, model.args.test_threads_num,
                                 normalize=True)
    del embeds1, embeds2
    gc.collect()
    return mrr_12


def attr_conv(attr_hs, attr_as, attr_vs, dim, feature_map_size=2, kernel_size=[2, 4], activation=tf.nn.tanh,
              layer_num=2):
    attr_as = tf.reshape(attr_as, [-1, 1, dim])
    attr_vs = tf.reshape(attr_vs, [-1, 1, dim])

    input_avs = tf.concat([attr_as, attr_vs], 1)
    input_shape = input_avs.shape.as_list()
    input_layer = tf.reshape(input_avs, [-1, input_shape[1], input_shape[2], 1])
    _conv = input_layer
    _conv = tf.layers.batch_normalization(_conv, 2)
    for i in range(layer_num):
        _conv = tf.layers.conv2d(inputs=_conv,
                                 filters=feature_map_size,
                                 kernel_size=kernel_size,
                                 strides=[1, 1],
                                 padding="same",
                                 activation=activation)
    _conv = tf.nn.l2_normalize(_conv, 2)
    _shape = _conv.shape.as_list()
    _flat = tf.reshape(_conv, [-1, _shape[1] * _shape[2] * _shape[3]])
    dense = tf.layers.dense(inputs=_flat, units=dim, activation=activation)
    dense = tf.nn.l2_normalize(dense)  # important!!
    score = -tf.reduce_sum(tf.square(attr_hs - dense), 1)
    return score


def read_word2vec(file_path, vector_dimension):
    print('\n', file_path)
    word2vec = dict()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split(' ')
            if len(line) != vector_dimension + 1:
                continue
            v = np.array(list(map(float, line[1:])), dtype=np.float32)
            word2vec[line[0]] = v
    file.close()
    return word2vec


def clear_attribute_triples(attribute_triples):
    print('\nbefore clear:', len(attribute_triples))
    # step 1
    # only use >= 10 attr.
    attribute_triples_new = set()
    attr_num = {}
    for (e, a, _) in attribute_triples:
        ent_num = 1
        if a in attr_num:
            ent_num += attr_num[a]
        attr_num[a] = ent_num
    attr_set = set(attr_num.keys())
    attr_set_new = set()
    for a in attr_set:
        if attr_num[a] >= 10:
            attr_set_new.add(a)
    for (e, a, v) in attribute_triples:
        if a in attr_set_new:
            attribute_triples_new.add((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 1:', len(attribute_triples))

    # step 2
    attribute_triples_new = []
    literals_number, literals_string = [], []
    for (e, a, v) in attribute_triples:
        if '"^^' in v:
            v = v[:v.index('"^^')]
        if v.endswith('"@en'):
            v = v[:v.index('"@en')]
        if is_number(v):
            literals_number.append(v)
        else:
            literals_string.append(v)
        v = v.replace('(', '').replace(')', '').replace(',', '').replace('"', '')
        v = v.replace('_', ' ').replace('-', ' ').replace('/', ' ')
        if 'http' in v:
            continue
        attribute_triples_new.append((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 2:', len(attribute_triples))
    return attribute_triples, literals_number, literals_string


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def generate_neg_attribute_triples(pos_batch, all_triples_set, entity_list, neg_triples_num, neighbor=None):
    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    for head, attribute, value in pos_batch:
        for i in range(neg_triples_num):
            while True:
                neg_head = random.choice(neighbor.get(head, entity_list))
                if (neg_head, attribute, value) not in all_triples_set:
                    break
            neg_batch.append((neg_head, attribute, value))
    assert len(neg_batch) == neg_triples_num * len(pos_batch)
    return neg_batch


def generate_attribute_triple_batch_queue(triple_list1, triple_list2, triple_set1, triple_set2, entity_list1,
                                          entity_list2, batch_size, steps, out_queue, neighbor1, neighbor2,
                                          neg_triples_num):
    for step in steps:
        pos_batch, neg_batch = generate_attribute_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                                               entity_list1, entity_list2, batch_size,
                                                               step, neighbor1, neighbor2, neg_triples_num)
        out_queue.put((pos_batch, neg_batch))
    exit(0)


def generate_attribute_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                    entity_list1, entity_list2, batch_size,
                                    step, neighbor1, neighbor2, neg_triples_num):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    pos_batch1 = bat.generate_pos_triples(triple_list1, batch_size1, step)
    pos_batch2 = bat.generate_pos_triples(triple_list2, batch_size2, step)
    neg_batch1 = generate_neg_attribute_triples(pos_batch1, triple_set1, entity_list1,
                                                neg_triples_num, neighbor=neighbor1)
    neg_batch2 = generate_neg_attribute_triples(pos_batch2, triple_set2, entity_list2,
                                                neg_triples_num, neighbor=neighbor2)
    return pos_batch1 + pos_batch2, neg_batch1 + neg_batch2


def alignment_loss(ents1, ents2):
    distance = ents1 - ents2
    loss = tf.reduce_sum(tf.reduce_sum(tf.square(distance), axis=1))
    return loss



class MMEA(BasicModel):

    def __init__(self):
        super().__init__()

    def init(self):
        self._define_img_embedds()

        self._define_variables()
        self._define_only_relation_graph()
        self._define_only_image_graph()
        self._define_only_attribute_graph()
        self._define_ra_common_space_learning_graph()

        self._define_common_space_learning_graph()
        self._define_unify_entity_mapping_graph()

        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)
        # self.merged = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter("logs/" + datetime.now().strftime("%Y%m%d%H%M%S"), self.session.graph)

    def _define_img_embedds(self):
        id_entity_dict1 = {v: k for k, v in self.kgs.kg1.entities_id_dict.items()}
        id_entity_dict2 = {v: k for k, v in self.kgs.kg2.entities_id_dict.items()}
        img_embedds = []
        for i in range(self.kgs.entities_num):
            if i in id_entity_dict1.keys():
                img_embedds.append(self.kgs.kg1.images_id_dict[i])
            elif i in id_entity_dict2.keys():
                img_embedds.append(self.kgs.kg2.images_id_dict[i])
        assert len(img_embedds) == self.kgs.entities_num
        img_embedds = np.array(img_embedds, np.float32)
        self.img_embeds = img_embedds

    def _define_variables(self):
        with tf.variable_scope('relation_view' + 'embeddings'):
            self.rv_ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'rv_ent_embeds',
                                                 self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)

        with tf.variable_scope('image_view' + 'embeddings'):
            self.iv_ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'iv_ent_embeds',
                                                 self.args.init, self.args.ent_l2_norm)
            self.iv_ent_mapping = init_embeddings([self.args.vgg_dim, self.args.dim], 'iv_ent_mapping',
                                                  self.args.init, self.args.ent_l2_norm)
            self.image_embeds = tf.constant(self.img_embeds, dtype=tf.float32)

        with tf.variable_scope('attribute_view' + 'embeddings'):
            self.av_ent_embeds = xavier_init([self.kgs.entities_num, self.args.dim], 'av_ent_embeds', True)
            self.av_c = xavier_init([self.kgs.attributes_num, self.args.rbf_dim], 'av_c', True)
            self.av_delta = xavier_init([self.kgs.attributes_num, self.args.rbf_dim], 'av_delta', True)
            self.av_W = xavier_init([self.args.rbf_dim, self.args.dim], 'av_W', True)
            self.av_b = xavier_init([1, self.args.dim], 'av_b', True)
            # False important!
            self.attr_embeds = xavier_init([self.kgs.attributes_num, self.args.dim], 'attr_embeds', False)

        with tf.variable_scope('shared' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)

        with tf.variable_scope('shared' + 'combination'):
            self.rv_mapping = tf.get_variable('rv_mapping', shape=[self.args.dim, self.args.dim],
                                              initializer=tf.initializers.orthogonal())
            self.iv_mapping = tf.get_variable('iv_mapping', shape=[self.args.dim, self.args.dim],
                                              initializer=tf.initializers.orthogonal())
            self.eye_mat = tf.constant(np.eye(self.args.dim), dtype=tf.float32, name='eye')


    def _define_only_relation_graph(self):
        with tf.name_scope('only_relation_placeholder'):
            self.rel_p_hs = tf.placeholder(tf.int32, shape=[None])
            self.rel_p_rs = tf.placeholder(tf.int32, shape=[None])
            self.rel_p_ts = tf.placeholder(tf.int32, shape=[None])
            self.rel_n_hs = tf.placeholder(tf.int32, shape=[None])
            self.rel_n_rs = tf.placeholder(tf.int32, shape=[None])
            self.rel_n_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('only_relation_lookup'):
            r_phs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_p_hs)
            r_prs = tf.nn.embedding_lookup(self.rel_embeds, self.rel_p_rs)
            r_pts = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_p_ts)
            r_nhs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_n_hs)
            r_nrs = tf.nn.embedding_lookup(self.rel_embeds, self.rel_n_rs)
            r_nts = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_n_ts)
        with tf.name_scope('only_relation_loss'):
            self.only_relation_loss = get_loss_func(r_phs, r_prs, r_pts, r_nhs, r_nrs, r_nts, self.args)
            # tf.summary.scalar('rel. loss', self.only_relation_loss)
            self.only_relation_optimizer = generate_optimizer(self.only_relation_loss, self.args.learning_rate,
                                                              opt=self.args.optimizer)

    def _define_only_image_graph(self):
        with tf.name_scope('images_embedding_placeholder'):
            self.img_p_es = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('images_embedding_lookup'):
            i_pes = tf.nn.embedding_lookup(self.iv_ent_embeds, self.img_p_es)
            i_pis = tf.nn.embedding_lookup(self.image_embeds, self.img_p_es)
        with tf.variable_scope('images_cnn'):
            output_layer = tf.layers.dense(inputs=i_pis, units=self.args.dim, activation=tf.nn.tanh)
            dense = tf.layers.dropout(output_layer, rate=0.5)
            dense = tf.nn.l2_normalize(dense)  # important!!
            pos_score = -tf.reduce_sum(tf.square(i_pes - dense), 1)
            pos_score = tf.log(1 + tf.exp(-pos_score))
            self.only_image_loss = tf.reduce_sum(pos_score)
            # add loss weight
            self.only_image_loss *= self.args.only_image_loss_weight
            self.only_image_optimizer = generate_optimizer(self.only_image_loss, self.args.learning_rate,
                                                           opt=self.args.optimizer)

    def _define_only_attribute_graph(self):
        with tf.name_scope('attribute_triple_placeholder'):
            self.attr_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.attr_pos_as = tf.placeholder(tf.int32, shape=[None])
            self.attr_pos_vs = tf.placeholder(tf.float32, shape=[None])

        with tf.name_scope('attribute_triple_lookup'):
            attr_phs = tf.nn.embedding_lookup(self.av_ent_embeds, self.attr_pos_hs)
            attr_pas = tf.nn.embedding_lookup(self.attr_embeds, self.attr_pos_as)
            attr_pc = tf.nn.embedding_lookup(self.av_c, self.attr_pos_as)
            attr_pdelta = tf.nn.embedding_lookup(self.av_delta, self.attr_pos_as)

        with tf.variable_scope('attribute_cnn'):
            a_pos_vs = tf.reshape(self.attr_pos_vs, [-1, 1])
            dist = -tf.square(tf.subtract(tf.tile(a_pos_vs, [1, self.args.rbf_dim]), attr_pc))
            delta2 = tf.square(attr_pdelta)
            RBF_out = tf.exp(tf.divide(dist, delta2))
            attr_pvs = tf.matmul(RBF_out, self.av_W) + self.av_b

            pos_score = attr_conv(attr_phs, attr_pas, attr_pvs, self.args.dim)
            pos_score = tf.log(1 + tf.exp(-pos_score))
            pos_loss = tf.reduce_sum(pos_score)
            self.attribute_loss = pos_loss
            tf.summary.scalar('attr.loss', self.attribute_loss)
            self.attribute_optimizer = generate_optimizer(self.attribute_loss, self.args.learning_rate,
                                                          opt=self.args.optimizer)

    def _define_unify_entity_mapping_graph(self):
        with tf.name_scope('entity_seed_links_placeholder'):
            self.seed_entities1 = tf.placeholder(tf.int32, shape=[None])
            self.seed_entities2 = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('entity_seed_links_lookup'):
            tes1 = tf.nn.embedding_lookup(self.ent_embeds, self.seed_entities1)
            tes2 = tf.nn.embedding_lookup(self.ent_embeds, self.seed_entities2)
        with tf.name_scope('entity_mapping_loss'):
            self.entity_mapping_loss = alignment_loss(tes1, tes2)
            self.entity_mapping_optimizer = generate_optimizer(self.entity_mapping_loss, self.args.learning_rate,
                                                               opt=self.args.optimizer)

    def _define_common_space_learning_graph(self):
        with tf.name_scope('cross_name_view_placeholder'):
            self.cn_hs = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('cross_name_view_lookup'):
            final_cn_phs = tf.nn.embedding_lookup(self.ent_embeds, self.cn_hs)
            cr_hs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.cn_hs)
            ci_hs = tf.nn.embedding_lookup(self.iv_ent_embeds, self.cn_hs)
            ca_hs = tf.nn.embedding_lookup(self.av_ent_embeds, self.cn_hs)
        with tf.name_scope('cross_name_view_loss'):
            self.cross_name_loss = self.args.relation_loss_weight * alignment_loss(final_cn_phs, cr_hs)
            self.cross_name_loss += self.args.image_loss_weight * alignment_loss(final_cn_phs, ci_hs)
            self.cross_name_loss += self.args.attr_loss_weight * alignment_loss(final_cn_phs, ca_hs)
            # tf.summary.scalar('common space loss', self.cross_name_loss)
            self.cross_name_optimizer = generate_optimizer(self.args.cv_weight * self.cross_name_loss,
                                                           self.args.ITC_learning_rate, opt=self.args.optimizer)


    def train_only_relation_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_relation_triple_batch_queue,
                       args=(self.kgs.kg1.relation_triples_list, self.kgs.kg2.relation_triples_list,
                             self.kgs.kg1.relation_triples_set, self.kgs.kg2.relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.only_relation_loss, self.only_relation_optimizer],
                                             feed_dict={self.rel_p_hs: [x[0] for x in batch_pos],
                                                        self.rel_p_rs: [x[1] for x in batch_pos],
                                                        self.rel_p_ts: [x[2] for x in batch_pos],
                                                        self.rel_n_hs: [x[0] for x in batch_neg],
                                                        self.rel_n_rs: [x[1] for x in batch_neg],
                                                        self.rel_n_ts: [x[2] for x in batch_neg]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        # self.writer.add_summary(rs, epoch)
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.relation_triples_list)
        random.shuffle(self.kgs.kg2.relation_triples_list)
        end = time.time()
        print('epoch {} of only rel., avg. loss: {:.8f}, time: {:.4f}s'.format(epoch, epoch_loss, end - start))


    def train_only_image_1epo(self, epoch, entities):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(entities) / self.args.entity_batch_size))
        batch_size = self.args.entity_batch_size if steps > 1 else len(entities)
        for i in range(steps):
            batch_pos = random.sample(entities, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.only_image_loss, self.only_image_optimizer],
                                             feed_dict={self.img_p_es: batch_pos})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        # self.writer.add_summary(rs, epoch)
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of only img., avg. loss: {:.8f}, time: {:.4f}s'.format(epoch, epoch_loss, end - start))


    def train_only_attribute_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        rs = None
        for steps_task in steps_tasks:
            mp.Process(target=generate_attribute_triple_batch_queue,
                       args=(self.kgs.kg1.attribute_triples_list, self.kgs.kg2.attribute_triples_list,
                             self.kgs.kg1.attribute_triples_set, self.kgs.kg2.attribute_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.attribute_batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, 0)).start()
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            batch_loss, _ = self.session.run(
                fetches=[self.attribute_loss, self.attribute_optimizer],
                feed_dict={self.attr_pos_hs: [x[0] for x in batch_pos],
                           self.attr_pos_as: [x[1] for x in batch_pos],
                           self.attr_pos_vs: [x[2] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        # self.writer.add_summary(rs, epoch)
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.attribute_triples_list)
        random.shuffle(self.kgs.kg2.attribute_triples_list)
        end = time.time()
        print('epoch {} of only att., avg. loss: {:.8f}, time: {:.4f}s'.format(epoch, epoch_loss, end - start))

    def train_entity_mapping_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            links_batch = random.sample(self.kgs.train_links, len(self.kgs.train_links) // triple_steps)
            batch_loss, _ = self.session.run(fetches=[self.entity_mapping_loss, self.entity_mapping_optimizer],
                                             feed_dict={self.seed_entities1: [x[0] for x in links_batch],
                                                        self.seed_entities2: [x[1] for x in links_batch]})
            epoch_loss += batch_loss
            trained_samples_num += len(links_batch)
        epoch_loss /= trained_samples_num
        print('epoch {} of entity avg. mapping loss: {:.8f}, cost time: {:.4f}s'.format(epoch, epoch_loss,
                                                                                        time.time() - start))

    def train_common_space_learning_1epo(self, epoch, entities):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(entities) / self.args.entity_batch_size))
        batch_size = self.args.entity_batch_size if steps > 1 else len(entities)
        for i in range(steps):
            batch_pos = random.sample(entities, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.cross_name_loss, self.cross_name_optimizer],
                                             feed_dict={self.cn_hs: batch_pos})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        # self.writer.add_summary(rs, epoch)
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of common space learning, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss,
                                                                                           end - start))

    def run(self):
        t = time.time()
        relation_triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        attribute_triples_num = self.kgs.kg1.local_attribute_triples_num + self.kgs.kg2.local_attribute_triples_num
        relation_triple_steps = int(math.ceil(relation_triples_num / self.args.batch_size))
        attribute_triple_steps = int(math.ceil(attribute_triples_num / self.args.batch_size))
        relation_step_tasks = task_divide(list(range(relation_triple_steps)), self.args.batch_threads_num)
        attribute_step_tasks = task_divide(list(range(attribute_triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        relation_batch_queue = manager.Queue()
        attribute_batch_queue = manager.Queue()

        neighbors1, neighbors2 = None, None
        entity_list = self.kgs.kg1.entities_list + self.kgs.kg2.entities_list

        for i in range(1, self.args.max_epoch + 1):
            print('epoch {}:'.format(i))
            # relation
            self.train_only_relation_1epo(i, relation_triple_steps, relation_step_tasks, relation_batch_queue,
                                          neighbors1, neighbors2)
            # image
            self.train_only_image_1epo(i, entity_list)
            # attribute
            self.train_only_attribute_1epo(i, attribute_triple_steps, attribute_step_tasks, attribute_batch_queue,
                                           neighbors1, neighbors2)
            # common
            self.train_common_space_learning_1epo(i, entity_list)
            self.train_entity_mapping_1epo(i, relation_triple_steps)

            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                valid_temp(self, embed_choice='rv')
                valid_temp(self, embed_choice='iv')
                valid_temp(self, embed_choice='av')
                # valid_temp(self, embed_choice='final')
                # valid_temp(self, embed_choice='avg')
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.args.early_stop and (self.early_stop or i == self.args.max_epoch):
                    break

        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
        # self.writer.close()
