'''
tensorflow implement Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks,https://arxiv.org/abs/1808.06866
'''
import tensorflow as tf
import numpy as np

import operator

class FilterPruner(object):
    def __init__(self,compression_factor):
        self.compression_factor = compression_factor

    def compress(self,sess):
        print("FilterPruner compress")
        for var in tf.trainable_variables():
            if 'weights' in var.name:
                print(var.name)

                var_T = tf.transpose(var, perm=[3, 2, 0, 1])

                var_T = sess.run(var_T)

                totalNum = var_T.shape[0]
                prunerNumber = self.compression_factor*totalNum
                prunerNumber = int(prunerNumber)
                print("prunerNumber=",prunerNumber,'totalNum=',totalNum)

                result = {}
                for i in range(0,totalNum):
                    data = var_T[i]
                    data = np.linalg.norm(data)

                    result[i] = data

                sorted_result = sorted(result.items(), key=operator.itemgetter(1), reverse=False)

                result = {}
                for i in range(0,prunerNumber):
                    result[i] = sorted_result[i][0]
                    # print(i,"=",sorted_result[i][0],sorted_result[i][1],type(sorted_result[i]))

                for i in range(0,prunerNumber):
                    data = var_T[result[i]]
                    zeroData = np.zeros(data.shape)
                    var_T[result[i]] = zeroData

                var_T = np.transpose(var_T,[2,3,1,0]) 

                sess.run(tf.assign(var,var_T))






