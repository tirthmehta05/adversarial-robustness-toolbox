# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements Smooth Adversarial Attack using PGD and DDN.

| Paper link: https://arxiv.org/pdf/1906.04584.pdf
| Authors' implementation: https://github.com/Hadisalman/smoothing-adversarial
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.config import ART_NUMPY_DTYPE
import torch

if TYPE_CHECKING:
    # pylint: disable=C0412
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)



def fit_pytorch(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        print('Inside macer fit method')
        import torch 
        import torch.nn.functional as F
        from torch.distributions.normal import Normal
        import time
        import os
        import random

        x = x.astype(ART_NUMPY_DTYPE)
        m = Normal(torch.tensor([0.0]).to(self._device), torch.tensor([1.0]).to(self._device))
        cl_total = 0.0
        rl_total = 0.0
        input_total = 0
        start_epoch = 0

        # Put the model in the training mode
        self.model.train()
        print(type(self))

        if self.optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")
        if self.scheduler is None:  # pragma: no cover
            raise ValueError("A scheduler is needed to train the model, but none for provided.")

        if kwargs.get('checkpoint') is not None:
          chkpt = kwargs.get('checkpoint')
          cpoint = torch.load(chkpt)
          self.model.load_state_dict(cpoint['net'])
          start_epoch = cpoint['epoch']
          self.scheduler.step(start_epoch)
          print('Loading model from epoch {} and checkpoint {}'.format(str(start_epoch), str(chkpt)))
        num_batch = int(np.ceil(len(x) / float(batch_size)))
        ind = np.arange(len(x))

        # Start training
        for epoch_num in range(start_epoch+1, nb_epochs+1):
            # Shuffle the examples
            random.shuffle(ind)
            t1 = time.time()

            print('Epoch: {} and learning rate: {}'.format(epoch_num, self.optimizer.state_dict().get('param_groups')[0].get('lr')))
            i = 0
            # Train for one epoch
            for nb in range(num_batch):
              print(i)
      
              i_batch = torch.from_numpy(x[ind[nb * batch_size : (nb + 1) * batch_size]]).to(self.device)
              o_batch = torch.from_numpy(y[ind[nb * batch_size : (nb + 1) * batch_size]]).to(self.device)
              input_size = len(i_batch)
              input_total += input_size

              new_shape = [input_size * self.gauss_num]
              new_shape.extend(i_batch[0].shape)
              i_batch = i_batch.repeat((1, self.gauss_num, 1, 1)).view(new_shape)
              noise = torch.randn_like(i_batch, device=self.device) * self.scale
              noisy_inputs = i_batch + noise
              outputs = self.model(noisy_inputs)
              # outputs = outputs[-1]
              outputs = outputs.reshape((input_size, self.gauss_num, self.nb_classes))

              # Classification loss
              outputs_softmax = F.softmax(outputs, dim=2).mean(1)
              outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
              #print(o_batch, "****", o_batch.shape)
              classification_loss = F.nll_loss(
                  outputs_logsoftmax, o_batch, reduction='sum')

              cl_total += classification_loss.item()

              # Robustness loss
              beta_outputs = outputs * self.beta  # only apply beta to the robustness loss
              beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
              top2 = torch.topk(beta_outputs_softmax, 2)
              top2_score = top2[0]
              top2_idx = top2[1]
              indices_correct = (top2_idx[:, 0] == o_batch)  # G_theta
              out0, out1 = top2_score[indices_correct,
                                      0], top2_score[indices_correct, 1]
              robustness_loss = m.icdf(out1) - m.icdf(out0)
              indices = ~torch.isnan(robustness_loss) & ~torch.isinf(
                  robustness_loss) & (torch.abs(robustness_loss) <= self.gamma)  # hinge
              out0, out1 = out0[indices], out1[indices]
              robustness_loss = m.icdf(out1) - m.icdf(out0) + self.gamma
              robustness_loss = robustness_loss.sum() * self.scale / 2
              rl_total += robustness_loss.item()
              # Final objective function
              loss = classification_loss + self.lbd * robustness_loss
              loss /= input_size
              self.optimizer.zero_grad()
              loss.backward()
              self.optimizer.step()
              i+=1

            self.scheduler.step()

            cl_total /= input_total
            rl_total /= input_total
            print('CLoss: {} and RLoss: {} after epoch: {}'.format(cl_total, rl_total, epoch_num))
            t2 = time.time()
            print('Elapsed time for {} epochs: {}'.format(epoch_num, str(t2 - t1)))
            if epoch_num % 5 == 0:
              state = {
                'net': self.model.state_dict(),
                'epoch': epoch_num,
                'optimizer_state_dict': self.optimizer.state_dict()
              }
              torch.save(state, '{}/{}_LR_{}.pth'.format(os.path.join('../'),'MACER_ART_'+str(epoch_num), self.optimizer.state_dict().get('param_groups')[0].get('lr')))
              
              

def fit_tensorflow(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        import tensorflow as tf
        import tensorflow_probability as tfp
        import time
        import os
        import math
        loc_norm = tf.constant([0.0])
        scale_norm = tf.constant([1.0])
        m = tfp.distributions.Normal(loc_norm, scale_norm)
        cl_total = 0.0
        rl_total = 0.0
        input_total = 0
        start_epoch = 0
        train_ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(batch_size)
        if kwargs.get('checkpoint') is not None:
            chkpt = kwargs.get('checkpoint')
            print(chkpt,"*")
            self.model.load_weights(chkpt)
            start_epoch = int(chkpt.split("/")[-1].split('_')[3])
            print(start_epoch,"**")
            print('Loading model from epoch {} and checkpoint {}'.format(str(start_epoch), str(chkpt)))
      
        for epoch_num in range(start_epoch+1, nb_epochs+1):
            print(epoch_num)
            t1 = time.time()
            i = 0
            for images, labels in train_ds:
                input_size = len(images)
                input_total += input_size
                new_shape = [input_size * self.gauss_num]
                new_shape.extend(images[0].shape)
                i_batch = tf.reshape(tf.tile(images, (1,self.gauss_num,1,1)),new_shape)
                noise = tf.random.normal(i_batch.shape, 0, 1, tf.float32) * self.scale
                noisy_inputs = i_batch + noise
                
                with tf.GradientTape() as tape:
                    outputs = self.model(noisy_inputs, training=True)
                    outputs = tf.reshape(outputs, [input_size, self.gauss_num, self.nb_classes])
                    # Classification loss
                    #outputs_softmax = F.softmax(outputs, dim=2).mean(1)
                    outputs_softmax = tf.reduce_mean(tf.nn.softmax(outputs, axis = 2),axis = 1)
                    #outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
                    outputs_logsoftmax = tf.math.log(outputs_softmax + 1e-10)  # avoid nan
                    # classification_loss = F.nll_loss(
                    #     outputs_logsoftmax, o_batch, reduction='sum')
                    classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, outputs_logsoftmax)
                    classification_loss = tf.reduce_sum(classification_loss)
                    cl_total += tf.get_static_value(classification_loss)

                    # Robustness loss
                    beta_outputs = outputs * self.beta  # only apply beta to the robustness loss
                    #beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
                    #print(tf.nn.softmax(beta_outputs, axis = 2),"!!!!!")
                    beta_outputs_softmax = tf.reduce_mean(tf.nn.softmax(beta_outputs, axis = 2),axis = 1)
                    #top2 = torch.topk(beta_outputs_softmax, 2)
                    top2 = tf.math.top_k(beta_outputs_softmax, k = 2)
                    top2_score = top2[0]
                    top2_idx = top2[1]
                    indices_correct = (top2_idx[:, 0] == labels)  # G_theta
                    out = tf.boolean_mask(top2_score,indices_correct)
                    out0, out1 = out[:,0], out[:,1] 
                    icdf_out1 = loc_norm + scale_norm * tf.math.erfinv(2 * out1 - 1) * math.sqrt(2)
                    icdf_out0 = loc_norm + scale_norm * tf.math.erfinv(2 * out0 - 1) * math.sqrt(2)
                    #robustness_loss = m.icdf(out1) - m.icdf(out0)
                    robustness_loss = icdf_out1 - icdf_out0
                    #indices = ~torch.isnan(robustness_loss) & ~torch.isinf(robustness_loss) & (torch.abs(robustness_loss) <= self.gamma)  # hinge
                    indices = ~tf.math.is_nan(robustness_loss) & ~tf.math.is_inf(robustness_loss) & (tf.abs(robustness_loss)<=self.gamma)
                    out0, out1 = out0[indices], out1[indices]
                    icdf_out1 = loc_norm + scale_norm * tf.math.erfinv(2 * out1 - 1) * math.sqrt(2)
                    icdf_out0 = loc_norm + scale_norm * tf.math.erfinv(2 * out0 - 1) * math.sqrt(2)
                    #robustness_loss = m.icdf(out1) - m.icdf(out0) + self.gamma
                    robustness_loss = icdf_out1 - icdf_out0 + self.gamma
                    #robustness_loss = robustness_loss.sum() * self.scale / 2
                    robustness_loss = tf.reduce_sum(robustness_loss)* self.scale / 2
                    rl_total += tf.get_static_value(robustness_loss)
                    # Final objective function
                    loss = classification_loss + self.lbd * robustness_loss
                    loss /= input_size
                    #self.optimizer.zero_grad()
                    #loss.backward()
                    #self.optimizer.step()

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                i+=1
      
            print("Classification Total Loss: ",cl_total)
            print("Robustness Total Loss: ",rl_total)
            self.optimizer.learning_rate = self.scheduler(epoch_num-1)
            cl_total /= input_total
            rl_total /= input_total
            print('CLoss: {} and RLoss: {} after epoch: {}'.format(cl_total, rl_total, epoch_num))
            t2 = time.time()
            print('Elapsed time for {} epochs: {}'.format(epoch_num, str(t2 - t1)))
            if epoch_num % 1 == 0:
                # state = {
                #     'net': self.model.state_dict(),
                #     'epoch': epoch_num,
                #     'optimizer_state_dict': self.optimizer.state_dict()}
                self.model.save_weights('{}/{}_LR_{}'.format(os.path.join('../'),'MACER_ART_5_'+str(epoch_num), self.optimizer._decayed_lr('float32').numpy()))
    