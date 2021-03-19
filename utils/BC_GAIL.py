from stable_baselines.gail import GAIL
import gym
import tensorflow as tf
import os
import math


class BC_GAIL(GAIL):
    """
    Behavior Cloning (BC) + Generative Adversarial Imitation Learning (GAIL)
    Perform BC before GAIL

    Reimplement pretrain function from base_class.py for Falcon training and evaluation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4,
                 adam_epsilon=1e-8, val_interval=None, save_path=None):
        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param dataset: (ExpertDataset) Dataset manager
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """

        max_acc = 0
        max_acc2 = 0
        max_acc3 = 0

        continuous_actions = isinstance(self.action_space, gym.spaces.Box)
        discrete_actions = isinstance(self.action_space, gym.spaces.Discrete)

        assert discrete_actions or continuous_actions, 'Only Discrete and Box action spaces are supported'

        # Validate the model every 10% of the total number of iteration
        if val_interval is None:
            # Prevent modulo by zero
            if n_epochs < 10:
                val_interval = 1
            else:
                val_interval = int(n_epochs / 10)

        with self.graph.as_default():
            with tf.variable_scope('pretrain'):
                if continuous_actions:
                    obs_ph, actions_ph, deterministic_actions_ph = self._get_pretrain_placeholders()
                    loss = tf.reduce_mean(tf.square(actions_ph - deterministic_actions_ph))
                else:
                    obs_ph, actions_ph, actions_logits_ph = self._get_pretrain_placeholders()
                    # actions_ph has a shape if (n_batch,), we reshape it to (n_batch, 1)
                    # so no additional changes is needed in the dataloader
                    actions_ph = tf.expand_dims(actions_ph, axis=1)
                    one_hot_actions = tf.one_hot(actions_ph, self.action_space.n)
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=actions_logits_ph,
                        labels=tf.stop_gradient(one_hot_actions)
                    )
                    loss = tf.reduce_mean(loss)

                    real_actions = tf.cast(tf.squeeze(actions_ph), tf.int32)
                    predicted_actions = tf.cast(tf.squeeze(tf.argmax(actions_logits_ph, axis=1)),
                                                tf.int32)  # return top 1 actions
                    predicted_actions2 = tf.cast(tf.math.top_k(actions_logits_ph, k=2)[1],
                                                 tf.int32)  # return top 2 actions
                    predicted_actions3 = tf.cast(tf.math.top_k(actions_logits_ph, k=3)[1],
                                                 tf.int32)  # return top 3 actions
                    real_actions_reshaped = tf.cast(tf.reshape(actions_ph, [-1, 1]), tf.int32)
                    ret = tf.equal(predicted_actions, real_actions)
                    accuracy = tf.reduce_sum(tf.cast(tf.equal(ret, True), tf.int32))
                    ret2 = tf.reduce_any(tf.equal(predicted_actions2, real_actions_reshaped), -1)
                    accuracy2 = tf.reduce_sum(tf.cast(tf.equal(ret2, True), tf.int32))
                    ret3 = tf.reduce_any(tf.equal(predicted_actions3, real_actions_reshaped), -1)
                    accuracy3 = tf.reduce_sum(tf.cast(tf.equal(ret3, True), tf.int32))

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)
                optim_op = optimizer.minimize(loss, var_list=self.params)

            self.sess.run(tf.global_variables_initializer())

        if self.verbose > 0:
            print("Pretraining with Behavior Cloning...")

        training_acc = 0.0
        for epoch_idx in range(int(n_epochs)):
            train_loss = 0.0
            # Full pass on the training set
            for _ in range(len(dataset.train_loader)):
                expert_obs, expert_actions = dataset.get_next_batch('train')
                feed_dict = {
                    obs_ph: expert_obs,
                    actions_ph: expert_actions,
                }
                train_loss_, _, training_acc_ = self.sess.run([loss, optim_op, accuracy], feed_dict)
                train_loss += train_loss_
                training_acc += training_acc_

            train_loss /= len(dataset.train_loader)

            if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
                val_loss = 0.0
                val_acc = 0.0
                val_acc2 = 0.0
                val_acc3 = 0.0
                # Full pass on the validation set
                for _ in range(len(dataset.val_loader)):
                    expert_obs, expert_actions = dataset.get_next_batch('val')
                    val_loss_, val_acc_, val_acc2_, val_acc3_, = self.sess.run([loss, accuracy, accuracy2, accuracy3],
                                                                               {obs_ph: expert_obs,
                                                                                actions_ph: expert_actions})
                    val_loss += val_loss_
                    val_acc += val_acc_
                    val_acc2 += val_acc2_
                    val_acc3 += val_acc3_

                val_loss /= len(dataset.val_loader)
                val_acc /= (len(dataset.val_loader) * dataset.val_loader.batch_size)
                val_acc2 /= (len(dataset.val_loader) * dataset.val_loader.batch_size)
                val_acc3 /= (len(dataset.val_loader) * dataset.val_loader.batch_size)
                # save the bc training models with the highest prediction accuracies
                if training_acc >= 0.99:
                    self.save("{}/{}".format(save_path, 'best_bc_model'))

                if val_acc > max_acc:
                    max_acc = val_acc
                    self.save("{}/{}".format(save_path, 'pretrained_bc_model'))

                if val_acc2 > max_acc2:
                    max_acc2 = val_acc2
                    self.save("{}/{}".format(save_path, 'pretrained_bc_model2'))

                if val_acc3 > max_acc3:
                    max_acc3 = val_acc3
                    self.save("{}/{}".format(save_path, 'pretrained_bc_model3'))

                if self.verbose > 0:
                    training_acc /= (len(dataset.train_loader) * dataset.train_loader.batch_size * val_interval)
                    print("==== Training progress {:.2f}% ====".format(100 * (epoch_idx + 1) / n_epochs))
                    print('Epoch {}'.format(epoch_idx + 1))
                    print(
                        "Training loss: {:.6f}, Validation loss: {:.6f}, training accuracy: {:.6f}, validation accuracy: {:.6f}, validation accuracy2: {:.6f}, validation accuracy3: {:.6f}".format(
                            train_loss, val_loss, training_acc, val_acc, val_acc2, val_acc3))
                    print(
                        "best validation accuracy: {:.6f}, best validation accuracy2: {:.6f}, best validation accuracy3: {:.6f}".format(
                            max_acc, max_acc2, max_acc3))
                    print()
                    training_acc = 0.0

            # Free memory
            del expert_obs, expert_actions
        if self.verbose > 0:
            print("Pretraining done.")
        return self
